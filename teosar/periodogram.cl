
#ifdef USE_DOUBLE
#ifdef cl_khr_fp64 
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#endif
#define TYPE double
#else /* USE_DOUBLE */
#define TYPE float
#endif /* USE_DOUBLE */

/* Use ifdefs to correctly set these constants */
#ifndef CONSTANTS_PER_SUM_TERM
#define CONSTANTS_PER_SUM_TERM 3
#endif

#define VARIABLES_PER_SUM_TERM (CONSTANTS_PER_SUM_TERM - 1)

/***
 * Efficient periodogram computation with OpenCL for GPU or CPU.
 * 
 * The goal is to be able to compute efficiently a significant number of
 * periodograms for many PS pairs in parallel. We assume each PS pair
 * could use hundreds of dates, and thus have hundreds of elements to sum.
 * 
 * For each PS p, the periodogram is of the form:
 * | sum_n w[n] exp(i * ( c0[p, n] + c1[p, n] * x1 + c2[p, n] * x2 + ... + ck[p, n] * xk) ) |
 * 
 * where w is a weight, c* are constants and x1, x2, .., xk the variables.
 * 
 * As a result there is a lot of variables to load, and at least c0 is specific to each
 * individual element of the loop.
 * 
 * Finally as there are hundreds of elements to sum, there are too many variables
 * to cache in registers, local memory or L1 cache, except if each variable load
 * is compensated with computing various (x1, x2, ..xk) at the same time.
 * 
 * This can be achieved with two different ways: either have several threads
 * compute in parallel the results for the same PS and different (x1, x2, ..xk),
 * or have each thread accumulate the same time the results of the sum
 * for various (x1, x2, ..xk).
 * Both can also be done at the same time.
 * 
 * As all parameters for a given PS should fit L1 cache, we will always have
 * low latency and fast bandwidth to load the variables. However only treating
 * one (x1, x2, ..xk) pair per thread will be severely cache bandwidth limited,
 * due to the significant cache line size.
 * 
 * To reduce this issue we must have each thread accumulate at the same time
 * the results for various (x1, x2, ..xk). In addition it will help that c0, c1, c2,...ck
 * can be accessed in a single memory call and thus should be stored contiguously
 * in memory.
 * 
 * There is a third way that could lead to better performance but is complex:
 * The sum could be divided among threads. Then each thread computes its share
 * of the sum for various (x1, x2, ..xk). Finally the partial results are combined.
 * This technique would reduce the number of memory access calls, but is quite
 * complex to write efficiently.
 * 
 * In addition saving the results of the computations for all (x1, x2, ..xk) pairs might
 * take a significant amount of space, thus reduction amount threads working on
 * the same PS before saving the result could help greatly.
 * 
 */

inline void periodogram_sum_term(__private TYPE *constants,
                                 __private TYPE *variables,
                                 TYPE weight,
                                 __private TYPE *accumulator_real,
                                 __private TYPE *accumulator_imag)
{
    TYPE real_part, imag_part;
    /* HERE use ifdefs to choose what the periodogram should look like */
    TYPE inside_exp = constants[0];
    # pragma unroll
    for (int j=0; j<VARIABLES_PER_SUM_TERM; j++){
         inside_exp += constants[j + 1] * variables[j];
     }
     
    imag_part = sincos(inside_exp, &real_part); // imag_part gets the sin and real_part the cos
    real_part = real_part * weight;
    imag_part = imag_part * weight;
    *accumulator_real = *accumulator_real + real_part;
    *accumulator_imag = *accumulator_imag + imag_part;
}


/* Number of sums to compute per thread at a given time.
 * Reduce this in case of register spilling.
 * Register spilling can be detected by very slow performance,
 * or compiler warning. */
#define NUM_SUMS_PER_THREAD 8

/* CALLER MUST DEFINE LOCAL_SIZE */
#define NUM_THREADS_PER_PS LOCAL_SIZE

__kernel void compute_periodograms(__global TYPE * restrict result,
                                   __global const TYPE * restrict constants,
                                   __global const TYPE * restrict variables,
                                   __global const TYPE * restrict weights,
                                   int num_ps,
                                   int num_variables_tuples_to_test_per_ps,
                                   int sum_size)
{
    /* Required layout:
     * constants: flattened 3D table constants[p, n, k] with p the PS index, n the sum index and k the tuple element index
     * variables: flattened 2D table variables[i, j] with i the tested variable tuple and j the tuple element index
     * weights: 1D table of size sum_size that should verify sum(weights) = 1
     * result: flattened 2D table result[p, i] with p the PS index and i the tuple index where
     *         the tuple is (found_maximum, variable tuple index)
     * 0 <= p < num_ps
     * 0 <= n < sum_size
     * 0 <= k < CONSTANTS_PER_SUM_TERM
     * 0 <= i < num_variables_tuples_to_test_per_ps
     * 0 <= j < VARIABLES_PER_SUM_TERM
     */
    
    /* Retrieve on which ps and tuples this thread should be working on. */
    int thread_id = get_global_id(0);
    //int num_threads_for_ps = get_local_size(0);
    int ps_id = get_global_id(1);

    __private TYPE constants_preloaded[CONSTANTS_PER_SUM_TERM] = {0}; /* initializing with 0, despite being useless, helps compilers */
    __private TYPE variables_preloaded[VARIABLES_PER_SUM_TERM * NUM_SUMS_PER_THREAD] = {0};
    __private TYPE accumulator_real[NUM_SUMS_PER_THREAD] = {0};
    __private TYPE accumulator_imag[NUM_SUMS_PER_THREAD] = {0};
    __local TYPE tmp[NUM_THREADS_PER_PS]; /* NUM_THREADS_PER_PS vs num_threads_for_ps: some compilers really behave bad without passing a define */
    __local int tmp2[NUM_THREADS_PER_PS];
    TYPE weight, value;
    TYPE best_maximum = 0.f; /* 0 <= periodogram <= 1. */
    int best_index = 0, best_tmp_index = 0;

    int variable_id0, i, j, k, m, n;
    for (variable_id0 = thread_id; variable_id0 < num_variables_tuples_to_test_per_ps; variable_id0 += NUM_THREADS_PER_PS*NUM_SUMS_PER_THREAD) {
        /* Load the variable values we test in this loop in this thread.
         * Those are the range variable_id0 to variable_id0 + NUM_SUMS_PER_THREAD - 1 */
        # pragma unroll   /* We must ALWAYS unroll a loop whenever the loop index is used to access a __private table as it is stored in registers. Not doing so generates slow code */
        for (i = 0; i < NUM_SUMS_PER_THREAD; i++) {
            /* The min is in case num_variables_tuples_to_test_per_ps is not
             * a multiple of NUM_SUMS_PER_THREAD * NUM_THREADS_PER_PS.
             * In that case we compute for the last element several times.
             * An alternative is to use if conditions but this can easily
             * generate bad code if not done right. */
            int variable_id = min(num_variables_tuples_to_test_per_ps-1, variable_id0 + i);
            # pragma unroll
            for (j = 0; j < VARIABLES_PER_SUM_TERM; j++) {
                variables_preloaded[i * VARIABLES_PER_SUM_TERM + j] = variables[variable_id * VARIABLES_PER_SUM_TERM + j];
            }
        }

        /* Performs the sum */

        /* Initialize accumulators */
        # pragma unroll
        for (i = 0; i < NUM_SUMS_PER_THREAD; i++) {
            accumulator_real[i] = 0;
            accumulator_imag[i] = 0;
        }
        /* Accumulate */
        for (n = 0; n < sum_size; n++) {
            weight = weights[n];
            # pragma unroll
            for (k = 0; k < CONSTANTS_PER_SUM_TERM; k++)
                constants_preloaded[k] = constants[(ps_id * sum_size + n) * CONSTANTS_PER_SUM_TERM + k];
            # pragma unroll
            for (i = 0; i < NUM_SUMS_PER_THREAD; i++) {
                periodogram_sum_term(constants_preloaded,
                                     &variables_preloaded[i * VARIABLES_PER_SUM_TERM],
                                     weight,
                                     &accumulator_real[i],
                                     &accumulator_imag[i]);
            }
        }
        /* Optimization Note: We could avoid the division and sqrt call as it doesn't impact the maximization result.
         * The min call can be optimized out of the loop as well and might not even be needed because of the W. */
        # pragma unroll
        for (i = 0; i < NUM_SUMS_PER_THREAD; i++) {
            value = sqrt(accumulator_real[i] * accumulator_real[i] + accumulator_imag[i] * accumulator_imag[i]);
            /* Each thread computes its own maximum */
            if (value > best_maximum) {
                best_maximum = value;
                best_index = min(num_variables_tuples_to_test_per_ps-1, variable_id0 + i); 
            }
        }
    }
    /* At this point this thread has computed the value and position of the maximum
     * of the periodogram for a given ps and a subset of tested variables.
     * Now we must merge the results with other threads in order to finish */

    /* The following code is naïve and could be improved a bit, but this should have
     * a small impact */

    /* All threads write their result */
    tmp[thread_id] = best_maximum;
    tmp2[thread_id] = best_index;

    /* Wait all threads have finished.
     * Note that ALL threads must reach this
     * section. We cannot early exit before */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Now quit all threads except the first one */
    if (thread_id != 0)
        return;
    /* best_maximum = tmp[0], best_tmp_index = 0 */
    for (m = 1; m < NUM_THREADS_PER_PS; m++) {
        if (tmp[m] > best_maximum) {
            best_maximum = tmp[m];
            best_tmp_index = m;
        }
    }
    best_index = tmp2[best_tmp_index];

    /* Write the result */
    result[2*ps_id] = best_maximum;
    result[2*ps_id+1] = (TYPE)best_index;
}
                                   
