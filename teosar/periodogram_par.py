import functools
from dataclasses import dataclass
from typing import Any, Optional  # noqa

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from numpy.typing import NDArray

from teosar.periodogram_cl import PeriodogramCL, RealArray, WrongShape


class PeriodogramTF:
    def __init__(
        self, num_constants_per_sum_term: int = 3, batch_size: int = 1024, ncpu: int = 1
    ):
        self.num_constants_per_sum_term = num_constants_per_sum_term
        self.num_variables_per_sum_term = num_constants_per_sum_term - 1
        self.batch_size = batch_size
        self.ncpu = ncpu

    def find_maximum(
        self,
        constants: RealArray,
        variables_init: RealArray,
        weights: Optional[RealArray] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        The maximized function is the selected periodogram function.
        The periodogram gamma function is:
        gamma = || sum_n w[n] * exp_imag(c[p,n,0] + c[p,n,1] * v[p, 0] + ... c[p,n,j] * v[p, j]) || / sum_n w[n]

        Inputs:
         . constants: 3D array [num_ps, sum_size, num_constants_per_sum_term]
         . weights: 1D array [sum_size]
         . variables_init: 2D array [num_ps, num_variables_per_sum_term]
             varibles_init[i] contains initialization variables for the ith PS.

        Outputs:
         . opt_variables: 2D array [num_ps, num_variables_per_sum_term]
                 opt_variables[i] contains for the i-th PS
                 the values of the variables that optimize the objective function
         . opt_gammas: 1D array [num_ps,]
                 Maximal gamma at optimum

        """
        num_ps, sum_size, num_constants_per_sum_term = constants.shape

        if num_constants_per_sum_term != self.num_constants_per_sum_term:
            raise WrongShape.from_msg(
                "constants",
                f"(num_ps, sum_size, {self.num_constants_per_sum_term})",
                constants.shape,
            )
        if variables_init.shape != (num_ps, self.num_variables_per_sum_term):
            raise WrongShape.from_msg(
                "variables_init",
                f"({num_ps}, {self.num_variables_per_sum_term})",
                variables_init.shape,
            )

        if weights is None:
            weights = np.ones((sum_size,), dtype=np.float64)
        elif weights.shape != (sum_size,):
            raise WrongShape.from_msg("weights", f"({sum_size},)", weights.shape)

        weights = weights / np.sum(weights)

        tf_const = tf.cast(constants, tf.float64)
        tf_init = tf.cast(variables_init, tf.float64)

        def make_val_and_grad_fn(value_fn):
            @functools.wraps(value_fn)
            def val_and_grad(x):
                return tfp.math.value_and_gradient(value_fn, x)

            return val_and_grad

        # Defined for a single element/PS
        def periodogram_to_optimize(tf_const_e, variables_e):
            res = weights * tf.exp(
                tf.dtypes.complex(
                    tf.zeros(sum_size, tf.float64),
                    (
                        tf_const_e[:, 0]
                        + tf.reduce_sum(tf_const_e[:, 1:] * variables_e, axis=1)
                    ),
                )
            )
            return tf.reduce_sum(res)

        # Encapsulating the lbfgs_minimize call means the
        # lbfgs optimisation itself will get compiled too
        def optimize_for_element(args):
            tf_const_e, tf_init_e = args

            @make_val_and_grad_fn
            def loss(x):
                per = periodogram_to_optimize(tf_const_e, x)
                return -tf.math.abs(per)

            res = tfp.optimizer.lbfgs_minimize(
                loss,
                initial_position=tf_init_e,
                tolerance=1e-15,
                max_iterations=10,
            )
            return tf.concat(
                [tf.expand_dims(res.objective_value, axis=0), res.position], axis=0
            )

        # Compile a tensorflow graph that runs the optimisation iteratively
        # for each PS. tf.function is used to compile the graph.
        # jit_compile means XLA is used to compile the graph and gives a huge
        # speedup in our case.
        # TODO: parallel_iterations is supposed to distribute computation
        # accross CPUs, but in practice only one is used.
        # Investigate what needs to be done to distribute the work.
        @tf.function(jit_compile=True)
        def optimize_for_slice(
            tf_const_slice,
            tf_init_slice,
        ):
            return tf.map_fn(
                optimize_for_element,
                (
                    tf_const_slice,
                    tf_init_slice,
                ),
                fn_output_signature=tf.TensorSpec(
                    self.num_constants_per_sum_term, dtype=tf.dtypes.float64
                ),
                parallel_iterations=self.ncpu,
            )

        if num_ps <= self.batch_size:
            start_indices = np.array([0])
            end_indices = np.array([num_ps])
        else:
            start_indices = np.arange(0, num_ps, self.batch_size)
            end_indices = np.append(start_indices[1:], num_ps)

        opt_variables = np.zeros(
            (num_ps, self.num_variables_per_sum_term), dtype=np.float64
        )
        opt_gammas = np.zeros((num_ps,), dtype=np.float64)

        for s, e in tqdm.tqdm(
            zip(start_indices, end_indices), total=len(start_indices)
        ):
            tf_const_slice = tf_const[s:e]
            tf_init_slice = tf_init[s:e, :]
            x = optimize_for_slice(
                tf_const_slice,
                tf_init_slice,
            )

            opt_gammas[s:e] = -x[:, 0].numpy()
            opt_variables[s:e, :] = x[:, 1:].numpy()

        return opt_variables, opt_gammas


@dataclass(frozen=True)
class PeriodogramPar:
    periodo_cl: PeriodogramCL
    periodo_tf: PeriodogramTF

    def __post_init__(self):
        assert (
            self.periodo_cl.num_constants_per_sum_term
            == self.periodo_tf.num_constants_per_sum_term
        )

    def find_maximum(
        self,
        constants: RealArray,
        variables: RealArray,
        weights: Optional[RealArray] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        The maximized function is the selected periodogram function.
        The periodogram gamma function is:
        gamma = || sum_n w[n] * exp_imag(c[p,n,0] + c[p,n,1] * v[p, 0] + ... c[p,n,j] * v[p, j]) || / sum_n w[n]


         Inputs:
         . constants: 3D array [num_ps, sum_size, num_constants_per_sum_term]
         . variables: 2D array [num_values_to_test, num_variables_per_sum_term]
         . weights: 1D array [sum_size]

        Outputs:
         . opt_variables: 2D array [num_ps, 2]
                 opt_variables[i] contains for the i-th PS
                 the values of the variables that optimize the objective function
         . opt_gammas: 1D array [num_ps,]
                 Maximal gamma at optimum

        """
        results = self.periodo_cl.find_maximum_on_grid(constants, variables, weights)

        maximums_indices = np.array(results[:, 1], dtype=np.int32)
        variables_init = variables[maximums_indices]
        return self.periodo_tf.find_maximum(constants, variables_init, weights)
