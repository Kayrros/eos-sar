import numpy as np
import time
from ortools.graph.python import min_cost_flow
from numpy.typing import NDArray
from typing import Any
import logging

from eos.sar.utils import wrap

RealArray = NDArray["np.floating[Any]"]


def round_assert_almost_int(array_to_round: RealArray, atol=1e-3):
    """
    Round an array and check that it was already mostly and integer up to a
    specified tolerance.

    Parameters
    ----------
    array_to_round : RealArray
        Array that contains floats that are almost integers.
    atol : float, optional
        Tolerance for the accepted deviation from the integer value. The default is 1e-3.

    Returns
    -------
    rounded : RealArray
        The rounded array, of the same type as the input array.

    Raises
    ------
    AssertionError (if not integer)

    """

    rounded = np.round(array_to_round)

    # profiling shows that this is a bit costly, 50 ms approx for 2048 x 2048 array
    # however, for now, I think it is worth it
    np.testing.assert_allclose(rounded, array_to_round, atol=atol)

    return rounded


def compute_residue(vert_grad1: RealArray, horiz_grad2: RealArray):
    """
    Computes the residue of two gradient matrices.

    Parameters
    ----------
    vert_grad1 : RealArray shape (N-1, M)
        Vertical gradient(axis=0) of a 2D (N, M) image.
    horiz_grad2 : RealArray shape (N, M-1)
        Horizontal gradient(axis=1) of a 2D (N, M) image.

    Returns
    -------
    residue : NDArray[np.int8] shape (N -1 , M - 1)
        Computed residue.

    Raises
    ------
    AssertionError (if not integer and not belonging to [-1, 0, 1])

    """
    residue = np.diff(vert_grad1, axis=1) - np.diff(horiz_grad2, axis=0)
    residue = - residue / (2 * np.pi)
    residue = round_assert_almost_int(residue)

    in_mask = np.isin(residue, [-1, 0, 1])

    assert np.all(in_mask), f"Unexpected residue value(s) of {residue[~in_mask]}"

    return residue.astype(np.int8)


def get_nodes(N_minus_1: int, M_minus_1: int):
    """
    Get the node indices for the connections in the graph of the MCF problem.
    Each node is in the middle of a cycle in the original image of size (N, M).
    Therefore, we have N - 1 x M - 1 node graph (similarly to the residues) + earth node.
    All nodes are connected horizontally an vertically (like a typical raster)
    and the nodes on the edges are connected to the earth node.
    This function gives the connections represented as the start node indices and the
    end node indices, in this order (left -> right, right -> left, up -> down, down -> up)

    Parameters
    ----------
    N_minus_1 : int
        Original image height - 1.
    M_minus_1 : int
        Original image width - 1.

    Returns
    -------
    start_nodes : NDArray[np.int64]
        Indices of the start nodes for the connections.
        They are given in the following order (the size is in parentheses):
            left (N-1 x M), right (N-1 x M), up (N x M-1), down (N x M-1).
    end_nodes :  NDArray[np.int64]
        Indices of the end nodes for the connections.
        They are given in the following order (the size is in parentheses):
            right (N-1 x M), left (N-1 x M), down (N x M-1), up (N x M-1).

    """
    # start_nodes et end_nodes
    # node indices at the center of each cycle
    num_data_nodes = N_minus_1 * M_minus_1
    indices = np.arange(num_data_nodes).reshape(N_minus_1, M_minus_1)

    # add earth node at the edges
    indices = np.pad(indices, ((1, 1), (1, 1)), constant_values=num_data_nodes)

    """
    if   a = 1 2 3
             4 5 6
    then append an earth node of index 7 around the edges
              7 7 7 7 7
              7 1 2 3 7
              7 4 5 6 7
              7 7 7 7 7
    """

    N = N_minus_1 + 1
    M = M_minus_1 + 1

    # horiz connections
    left_index = indices[1:N, :M].ravel()
    right_index = indices[1:N, 1:].ravel()

    # vertical connections
    up_index = indices[:N, 1:M].ravel()
    down_index = indices[1:, 1:M].ravel()

    start_nodes = np.concatenate((left_index, right_index, up_index, down_index))
    end_nodes = np.concatenate((right_index, left_index, down_index, up_index))

    return start_nodes, end_nodes


def solve_smcf(residue: NDArray[np.int8]):
    """
    Solves the minimum cost flow problem, that takes as input the residues of an
    unprecise gradient field, and gives flows that would correct it to have a true
    gradient field. Residues are interpreted as suplies on each node.
    Each flow corresponds to a gradient in the original image,
    i.e. a horizontal flow corresponds to a vertical gradient.
    Thus, we can assign costs for each flow depending on our certainty of the corresponding
    gradient estimation (the more certain, the higher the cost of having a flow).
    For now, the cost is uniform=1.

    Attention! min_cost_flow.SimpleMinCostFlow() seems to consider that everyting is integer
    even you give a float for suplies for ex., it seems to give integer solutions...
    In our case, this is not a big problem because even if we solve for a floating solution
    (if we want floating flows), the theory says that it ends up being an integer solution,
    because the residue is integer...
    Theorem (Integrality): Any minimum cost network flow problem instance whose demands are
    all integers has an optimal solution with integer flow on each edge.

    Anyway, care must be taken for the costs in the future, not to use floats, quantize in integers...

    Parameters
    ----------
    residue : NDArray[np.int8]
        Residue array of shape (N-1, M-1) for an (N, M) image.

    Returns
    -------
    flows : NDArray[np.int64]
        Optimal flows.

    """
    N_minus_1, M_minus_1 = residue.shape

    start_nodes, end_nodes = get_nodes(N_minus_1, M_minus_1)

    n_arc = len(start_nodes)

    # capacities, 1000 to allow for a big number of flows
    capacities = np.full(n_arc, 1000)

    # costs
    unit_costs = np.ones(n_arc)

    # supplies
    out = np.sum(residue)

    # add earth node supply
    # earth node contains all remaining unbalanced residues
    supplies = np.append(residue.ravel(), -out)

    time_smcf = time.time()

    # construct object
    smcf = min_cost_flow.SimpleMinCostFlow()
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs)
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    # solve
    status = smcf.solve()

    end_time_smcf = time.time() - time_smcf

    logging.info(status)
    logging.info(f"smcf time: {end_time_smcf}")

    flows = smcf.flows(all_arcs)

    return flows


def ambiguity_from_flows(flows: NDArray[np.int64], N: int, M: int):
    """
    Get the integer 2pi ambiguities from the flow arrays, assuming the
    flows are ordered [x1_plus, x1_minus, x2_minus, x2_plus]

    Parameters
    ----------
    flows : NDArray[np.int64]
        Flows for the arcs as defined previously in get_nodes.
    N : int
        Height of original image.
    M : int
        Width of original image.

    Returns
    -------
    K1 : NDArray[np.int64] (N-1, M)
        Vertical gradient ambiguities.
    K2 : NDArray[np.int64] (N, M -1)
        Horizontal gradient ambiguities.

    Raises
    ------
    AssertionError: Arcs have been constructed as bidirectionnal, i.e. if A, B are connected
    if A -> B, also B->A. In the solution, check that at least one of the two flows is zero.
    Otherwise raise an error.
    """
    # horizontal flows correspond to vertical gradients

    end = (N - 1) * M
    # left -> right
    x1_plus = flows[:end].reshape(N - 1, M)

    start = end
    end = 2 * start
    # right -> left
    x1_minus = flows[start:end].reshape(N - 1, M)

    # vertical flows correspond to horizontal gradients

    start = end
    end = start + N * (M - 1)
    # up -> down
    x2_minus = flows[start:end].reshape(N, M - 1)

    start = end
    # down -> up
    x2_plus = flows[start:].reshape(N, M - 1)

    # some assertions

    """
    if you change the mcf solver, such that the solution is not int64
    you need to check that it is integer
    for instance do the following
    x1_plus = round_assert_almost_int(x1_plus)
    x1_minus = round_assert_almost_int(x1_minus)
    x2_plus = round_assert_almost_int(x2_plus)
    x2_minus = round_assert_almost_int(x2_minus)
    """

    # assert at least one is 0
    assert np.all(np.logical_or(x1_plus == 0, x1_minus == 0)
                  ), "horizontal flow error: at leat one flow from + and - should be zero"
    assert np.all(np.logical_or(x2_plus == 0, x2_minus == 0)
                  ), "vertical flow error: at leat one flow from + and - should be zero"

    K1 = x1_plus - x1_minus
    K2 = x2_plus - x2_minus

    return K1, K2


def mcf_estim_unwrapped_gradients(wrapped_phase: RealArray):
    """
    From the wrapped phase, estimate the unwrapped gradient field using the MCF method.

    Parameters
    ----------
    wrapped_phase : RealArray
        Wrapped phase image, shape (N, M).

    Returns
    -------
    grad1_unwrapped_phase : RealArray
        Unwrapped gradient estimation in vertical direction (axis=0).
        Same type as wrapped phase. shape (N-1, M).
    grad2_unwrapped_phase : RealArray
        Unwrapped gradient estimation in horizontal direction (axis=1).
        Same type as wrapped phase. shape (N, M-1).

    Raises
    ------
    AssertionError:
        - If the Residue is not an integer and not belonging to [-1, 0, 1]
        - If not at least one flow in a bidirectional arc is 0
        - If the retrieved gradient has a non zero residue at any pixel

    """
    N = np.shape(wrapped_phase)[0]
    M = np.shape(wrapped_phase)[1]

    w_vert_grad_array1 = wrap(np.diff(wrapped_phase, axis=0))
    w_horiz_grad_array2 = wrap(np.diff(wrapped_phase, axis=1))

    # residue matrix
    residue = compute_residue(w_vert_grad_array1, w_horiz_grad_array2)

    num_non_zero = np.sum(residue != 0)
    tot = (N - 1) * (M - 1)
    perctg = (num_non_zero / tot) * 100
    logging.info(
        f"Number of non zero residue : {num_non_zero} / {tot} = {perctg} %")

    # constructing MCF graph and solving for the flows
    flows = solve_smcf(residue)

    # release memory
    del residue

    # interpreting the flows as integer ambiguities on gradients
    K1, K2 = ambiguity_from_flows(flows, N, M)

    # release memory
    del flows

    # adding ambiguities to get unwrapped gradients
    grad1_unwrapped_phase = (w_vert_grad_array1 +
                             2 * np.pi * K1).astype(wrapped_phase.dtype)
    grad2_unwrapped_phase = (w_horiz_grad_array2 +
                             2 * np.pi * K2).astype(wrapped_phase.dtype)

    # assert true gradient field
    true_grad = np.all(compute_residue(
        grad1_unwrapped_phase, grad2_unwrapped_phase) == 0)

    assert true_grad, "non zero residue detected: the gradient estimated by MCF is not a true gradient field"
    logging.info("The gradient estimated by MCF is a true gradient field")

    return grad1_unwrapped_phase, grad2_unwrapped_phase


def integrate_gradient_field(grad1: RealArray, grad2: RealArray, upper_left_val: float = 0.):
    """
    Integrate a gradient field (rasters of vertical and horizontal gradients) to get the image. We start
    from the upper left value and go from left to right, down one line, then right to left, down one line, until
    the end of the image.

    Parameters
    ----------
    grad1 : RealArray
        Vertical gradient (axis=0) of shape (N-1, M).
    grad2 : RealArray
        Horizontal gradient (axis=1) of shape (N, M-1).
    upper_left_val : float, optional
        Upper left corner value for the final image from which we start integrating the gradients.
        The default is 0.

    Returns
    -------
    integrated : RealArray
        Integrated image result.

    Raises
    ------
    AssertionError: If the gradient shapes are not compatible.
    """

    N_minus_1, M = grad1.shape
    N, M_minus_1 = grad2.shape

    assert N == N_minus_1 + 1, "Gradient shape mismatch in axis 0"
    assert M == M_minus_1 + 1, "Gradient shape mismatch in axis 1"

    integrated = np.zeros((N, M), dtype=grad1.dtype)

    last_cumsum = upper_left_val
    integrated[0, 0] = upper_left_val
    # integrate by going from left to right, up to bottom on all gradients
    for i in range(N):
        is_odd = i % 2

        if i:  # do not go vertically for the first iteration
            # vertical gradient
            # either first column or last column
            col = M_minus_1 if is_odd else 0

            # increment last cumsum
            last_cumsum += grad1[i - 1, col]

            # write to unwrapped array
            integrated[i, col] = last_cumsum

        if is_odd:
            # go from right to left, need to also invert gradient sign
            cumsum = last_cumsum + np.cumsum(- grad2[i, ::-1])

            # set horizontal row from right to left, skipping first element
            integrated[i, M - 2::-1] = cumsum

            # update cumsum
            last_cumsum = cumsum[-1]
        else:
            # go from left to right, the sign is correct
            cumsum = last_cumsum + np.cumsum(grad2[i])

            # set horizontal row from left to right, skipping first element
            integrated[i, 1:M] = cumsum

            # update cumsum
            last_cumsum = cumsum[-1]

    return integrated


def mcf(wrapped_phase: RealArray):
    """
    Unwrap a wrapped phase image with the Minimum Cost Flow (MCF) [1] method.
    For now, we set the costs to 1 everywhere.

    Parameters
    ----------
    wrapped_phase : RealArray
        Wrapped phase image of shape (N, M), observed in the [-pi, pi] interval.

    Returns
    -------
    unwrapped_phase : RealArray
        Unwrapped result, obtained by addind integer 2 * np.pi to the wrapped image, i.e.
        congruence is verified for the solution (if you re-wrap, you obtain the wrapped input).

    Notes
    -----
    [1] M. Costantini, “A novel phase unwrapping method based on network programming”,
    IEEE Trans. Geosci. Remote Sens., vol. 36, no. 3, pp. 813–821, 1998.

    Raises
    ------
    AssertionError:
            - If the Residue is not an integer and not belonging to [-1, 0, 1]
            - If not at least one flow in a bidirectional arc is 0
            - If the retrieved gradient has a non zero residue at any pixel
            - If the unwrapped image has a non integer ambiguity with respect to the
            wrapped image (atol 1e-2).

    """
    # estimate unwrapped phase gradients with MCF method
    grad1_unwrapped, grad2_unwrapped = mcf_estim_unwrapped_gradients(
        wrapped_phase)

    unwrapped_phase = integrate_gradient_field(grad1_unwrapped, grad2_unwrapped,
                                               wrapped_phase[0, 0])

    # avoid cumulating small floating precision errors during gradient integration
    # this seems to bring a very small overhead from profiling
    # around 7 ms for 2048 x 2048 array, except rounding wich is 50 ms
    amb = (unwrapped_phase - wrapped_phase) / (2 * np.pi)
    amb = round_assert_almost_int(amb, atol=1e-2).astype(wrapped_phase.dtype)

    unwrapped_phase = wrapped_phase + 2 * np.pi * amb

    return unwrapped_phase
