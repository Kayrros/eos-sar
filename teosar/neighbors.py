import numpy as np
import scipy
import scipy.sparse
from numpy.typing import NDArray

from teosar import psutils


def get_neighbors(
    ps_col: NDArray[np.int32],
    ps_row: NDArray[np.int32],
    distance_threshold: float,
    resolution_x: float,
    resolution_y: float,
) -> NDArray[np.bool_]:
    num_PS = len(ps_col)
    # find neighbors
    distance_x = np.abs(ps_col.reshape([1, -1]) - ps_col.reshape([-1, 1]))
    distance_y = np.abs(ps_row.reshape([1, -1]) - ps_row.reshape([-1, 1]))

    distance_in_meters = np.sqrt(
        (resolution_x * distance_x) ** 2 + (resolution_y * distance_y) ** 2
    )
    is_at_ok_distance = distance_in_meters < distance_threshold

    # remove yourself
    for i in range(num_PS):
        is_at_ok_distance[i, i] = False

    return is_at_ok_distance


def phi_ps_neighbors(phi_sparse, is_at_ok_distance):
    output = np.full(phi_sparse.shape, np.nan)
    for k in range(phi_sparse.shape[1]):
        neighbors = is_at_ok_distance.indices[
            is_at_ok_distance.indptr[k] : is_at_ok_distance.indptr[k + 1]
        ]
        phi_neighbors = phi_sparse[:, neighbors]

        if len(neighbors):
            phi_neighbors = np.angle(np.nansum(np.exp(1j * phi_neighbors), axis=-1))
            output[:, k] = phi_neighbors

    return output


def compute_phi_neighbors(
    ps_col, ps_row, phi_sparse_ts, resolution_x, resolution_y, distance_threshold=300
):
    is_at_ok_distance = scipy.sparse.csr_array(
        get_neighbors(ps_col, ps_row, distance_threshold, resolution_x, resolution_y)
    )

    phi_sparse_ts = np.array(phi_sparse_ts)
    phi_neighbors = phi_ps_neighbors(phi_sparse_ts, is_at_ok_distance)

    return phi_neighbors


def compute_ps_vs_neighbors(
    ps_col, ps_row, phi_sparse_ts, resolution_x, resolution_y, distance_threshold=300
):
    phi_sparse_ts = np.array(phi_sparse_ts)
    phi_neighbors = compute_phi_neighbors(
        ps_col,
        ps_row,
        phi_sparse_ts,
        resolution_x,
        resolution_y,
        distance_threshold=distance_threshold,
    )
    outs = psutils.wrap(phi_sparse_ts - phi_neighbors)
    return outs
