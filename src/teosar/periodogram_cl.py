from __future__ import annotations

import math
from importlib import resources
from typing import (
    Any,  # noqa
    Optional,
)

import numpy as np
import pyopencl as cl
from numpy.typing import NDArray

RealArray = NDArray["np.floating[Any]"]

mf = cl.mem_flags


def profile(events, f_res, name, is_kernel):
    """
    Add an event to the profile. Events are organized by types.
    """
    if events is None:
        return
    evt = f_res
    if name in events:
        events[name].append(evt)
    else:
        events[name] = [evt]
    if "evt_all" in events:
        events["evt_all"].append(evt)
    else:
        events["evt_all"] = [evt]
    if is_kernel:
        if "evt_all_kernel" in events:
            events["evt_all_kernel"].append(evt)
        else:
            events["evt_all_kernel"] = [evt]
    else:
        if "evt_all_other" in events:
            events["evt_all_other"].append(evt)
        else:
            events["evt_all_other"] = [evt]


def print_profile_info(events):
    """
    Print a profile history of the code (with computation time).
    """
    if events is None:
        return
    print("Profiling Information:")
    tot_time = (
        events["evt_all"][-1].profile.end - events["evt_all"][0].profile.start
    ) * 1e-6
    print("Time between first and last OpenCL event: %f ms" % tot_time)
    sum_time = (
        sum([(evt.profile.end - evt.profile.start) for evt in events["evt_all"]]) * 1e-6
    )
    print("Time spent in OpenCL events: %f ms" % sum_time)
    print("Thus a CPU overhead of: %f ms\n" % (tot_time - sum_time))

    sum_time_k = (
        sum([(evt.profile.end - evt.profile.start) for evt in events["evt_all_kernel"]])
        * 1e-6
    )
    print("Time spent executing kernels: %f ms" % sum_time_k)
    print(
        "Time spent in transfers and synchronizations: %f ms\n"
        % (sum_time - sum_time_k)
    )

    print("Total cumulated time spent per kernel/OpenCL operation:")
    for key in sorted(events.keys()):
        if key[:4] == "evt_":
            continue
        print(
            "%s: %f ms"
            % (
                key,
                sum([(evt.profile.end - evt.profile.start) for evt in events[key]])
                * 1e-6,
            )
        )
    print("Note: enqueue_copy corresponds to transfers CPU<->OpenCL (CPU or GPU)")


def DIVUP(a, b):
    """
    Integer division with rounding to the upper value.
    """
    return int(math.ceil(float(a) / float(b)))


class PeriodogramCL:
    def __init__(
        self,
        ctx: cl.Context = None,
        queue: cl.CommandQueue = None,
        use_double_precision: bool = False,
        interactive_device_selection: bool = False,
        enable_profile: bool = False,
        num_constants_per_sum_term: int = 3,
    ):
        # Create an OpenCL context if None given
        if ctx is None:
            # If interactive is False, the first device available is selected
            # use env var PYOPENCL_CTX to enforce a given device
            ctx = cl.create_some_context(interactive=interactive_device_selection)
            # A queue is related to a ctx, we cannot reuse one if we create
            # a new ctx
            if queue is not None:
                queue = None

        # Create the OpenCL command queue
        if queue is None:
            # Activate profiling if requested.
            if enable_profile:
                queue = cl.CommandQueue(
                    ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
                )
                events: Optional[dict[str, Any]] = {}
            else:
                queue = cl.CommandQueue(ctx)
                events = None

        if enable_profile:
            print("Using device " + ctx.devices[0].get_info(cl.device_info.NAME))

        # Most common thread group size that fit all GPUs are 64, 128, 256
        self.num_threads_per_ps = 256

        with resources.files("teosar").joinpath("periodogram.cl").open("r") as f:
            fstr = "".join(f.readlines())

        build_options = "-DUSE_DOUBLE" if use_double_precision else ""
        build_options += " -DLOCAL_SIZE=%d" % self.num_threads_per_ps
        build_options += " -DCONSTANTS_PER_SUM_TERM=%d" % num_constants_per_sum_term

        self.num_constants_per_sum_term = num_constants_per_sum_term
        self.num_variables_per_sum_term = num_constants_per_sum_term - 1
        # See code on how to compile for other periodogram computations

        program = cl.Program(ctx, fstr).build(options=build_options)
        compute_periodograms = program.compute_periodograms
        compute_periodograms.set_scalar_arg_dtypes(
            [None, None, None, None, np.int32, np.int32, np.int32]
        )
        self.compute_periodograms = compute_periodograms

        self.ctx = ctx
        self.queue = queue
        self.events = events
        self.dtype = np.float64 if use_double_precision else np.float32
        self.dtype_size = 8 if use_double_precision else 4

    def find_maximum_on_grid(
        self,
        constants: RealArray,
        variables: RealArray,
        weights: Optional[RealArray] = None,
    ) -> RealArray:
        """
        Inputs:
        . constants: 3D array [num_ps, sum_size, num_constants_per_sum_term]
        . variables: 2D array [num_values_to_test, num_variables_per_sum_term]
        . weights: 1D array [sum_size]

        Outputs:
        . results: 2D array [num_ps, 2] where results[i] contains for the i-th PS
          the maximum and the index (1st dimension) of the values in the variables
          table that attain this maximum.
          The maximized function is the selected periodogram function.
          The periodogram function is:
          gamma = || sum_n w[n] * exp_imag(c[p,n,0] + c[p,n,1] * v[p, 0] + ... c[p,n,j] * v[p, j]) || / sum_n w[n]

        """
        num_ps = constants.shape[0]
        sum_size = constants.shape[1]
        num_values_to_test = variables.shape[0]

        if (
            len(constants.shape) != 3
            or constants.shape[2] != self.num_constants_per_sum_term
        ):
            raise WrongShape.from_msg(
                "constants",
                f"(num_ps, sum_size, {self.num_constants_per_sum_term})",
                constants.shape,
            )

        if (
            len(variables.shape) != 2
            or variables.shape[1] != self.num_variables_per_sum_term
        ):
            raise WrongShape.from_msg(
                "variables",
                f"(num_values_to_test, {self.num_variables_per_sum_term})",
                variables.shape,
            )

        if weights is None:
            weights = np.ones((sum_size,), dtype=self.dtype)
        elif len(weights.shape) != 1 or weights.shape[0] != sum_size:
            raise WrongShape.from_msg("weights", f"({sum_size},) ", weights.shape)

        # Ensure we have contiguous allocation and the correct dtype
        constants = np.ascontiguousarray(constants, dtype=self.dtype)
        variables = np.ascontiguousarray(variables, dtype=self.dtype)
        weights: RealArray = np.ascontiguousarray(weights, dtype=self.dtype)
        weights /= np.sum(weights)

        # Allocate the OpenCL buffers
        constants_cl = cl.Buffer(
            self.ctx,
            mf.READ_ONLY,
            num_ps * sum_size * self.num_constants_per_sum_term * self.dtype_size,
        )
        variables_cl = cl.Buffer(
            self.ctx,
            mf.READ_ONLY,
            num_values_to_test * self.num_variables_per_sum_term * self.dtype_size,
        )
        weights_cl = cl.Buffer(self.ctx, mf.READ_ONLY, sum_size * self.dtype_size)
        results_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, num_ps * 2 * self.dtype_size)

        # Upload the inputs
        profile(
            self.events,
            cl.enqueue_copy(
                self.queue, constants_cl, constants.flatten(), device_offset=0
            ),
            "enqueue_copy",
            False,
        )
        profile(
            self.events,
            cl.enqueue_copy(
                self.queue, variables_cl, variables.flatten(), device_offset=0
            ),
            "enqueue_copy",
            False,
        )
        profile(
            self.events,
            cl.enqueue_copy(self.queue, weights_cl, weights.flatten(), device_offset=0),
            "enqueue_copy",
            False,
        )

        # Determine the thread distribution: num_threads_per_ps times num_ps threads
        global_size = [self.num_threads_per_ps, num_ps]
        # Allocate the threads by groups of num_threads_per_ps
        local_size = [self.num_threads_per_ps, 1]

        # Launch the OpenCL kernel
        profile(
            self.events,
            self.compute_periodograms(
                self.queue,
                global_size,
                local_size,
                results_cl,
                constants_cl,
                variables_cl,
                weights_cl,
                num_ps,
                num_values_to_test,
                sum_size,
            ),
            "compute_periodograms",
            True,
        )

        # Wait the results and download them
        results: RealArray = np.empty([num_ps * 2], dtype=self.dtype, order="C")
        profile(
            self.events,
            cl.enqueue_copy(self.queue, results, results_cl),
            "enqueue_copy",
            False,
        )
        results = results.reshape([num_ps, 2])

        # print profiling info if requested
        print_profile_info(self.events)

        return results


class WrongShape(ValueError):
    @staticmethod
    def from_msg(
        varname: str, expected_shape: str, var_shape: tuple[int, ...]
    ) -> WrongShape:
        msg = f"Wrong shape for {varname}. Expected shape: {expected_shape}, got {var_shape}"
        return WrongShape(msg)


def create_constants(
    num_ps: int,
    sum_size: int,
    phi_ps_mat: RealArray,
    const_list: list[RealArray],
    dtype=np.float32,
) -> RealArray:
    constants = np.zeros([num_ps, sum_size, len(const_list) + 1], dtype=dtype)

    if phi_ps_mat.shape == (num_ps, sum_size):
        constants[:, :, 0] = phi_ps_mat
    elif phi_ps_mat.shape == (sum_size, num_ps):
        constants[:, :, 0] = phi_ps_mat.T
    else:
        raise WrongShape.from_msg(
            "phi_ps_mat", str((num_ps, sum_size)), phi_ps_mat.shape
        )

    for i, const in enumerate(const_list):
        try:
            const_view = np.broadcast_to(const, (num_ps, sum_size))
        except ValueError as e:
            raise WrongShape(
                f"Could not broadcast const {i} of shape {const.shape} to (num_ps, sum_size)\n{e}"
            )
        else:
            constants[:, :, 1 + i] = const_view
    return constants


def create_variables(test_vars_list: list[RealArray], dtype=np.float32) -> RealArray:
    variables = np.stack(np.meshgrid(*test_vars_list), axis=-1, dtype=dtype)
    variables = np.reshape(variables, [-1, len(test_vars_list)])
    return variables
