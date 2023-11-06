import functools
import os
from typing import Optional

import numpy as np
import scipy
import tifffile
from numpy.typing import NDArray

from teosar import periodogram, psc, psutils

"""
Ferreti 2001
Needs to work on small areas (< 5x5km according to Ferreti 2000)
The work consists in:
. Estimate a first draw of PS using the DA > 0.25 test
. Use an iterative algorithm to estimate PS velocity and image APS
 using the strong assumptions that the PS motion is linear with time,
 and that the APS is an affine plane for each image (thus the need
 for the area to be small)
---- end of what we implemented
---- the rest needs to be added
. Extrapolate APS on the whole grid, taking into account residual
 errors in the estimation model (which we model as APS + random error)
 and smoothing spatially these errors
. Use the periodogram to estimate new PS now that APS has been removed (threshold of 0.75)
"""


def save_debug_image(
    path, PS_X_coordinates, PS_Y_coordinates, parent_shape, sparse_data, as_complex=True
):
    data_full = psutils.sparse_data_to_raster(
        sparse_data, PS_Y_coordinates, PS_X_coordinates, parent_shape
    )
    if as_complex:
        data_full = np.exp(1j * data_full).astype(np.complex64)
    tifffile.imwrite(path, data_full)


def iterative_alternate_periodogram(
    PS_X_coordinates,
    PS_Y_coordinates,
    Delta_phi_against_ref,
    bperp,
    inc,
    rng,
    dates,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    wavelength=5.5465763 * 1e-2,
    debug_path=None,
    *,
    use_tensorflow=True,
):
    # Estimate LOS velocity, DEM errors and APSs on the sparse grid
    # Solving system 13 of Ferreti 2001
    # We start with errors = delta_phi, APS = 0, DEM errors = 0, velocity = 0
    # Hypothesis: velocity is PS dependent, but is constant over time
    # APS is affine on the image plane

    # Convert dates to deltas in day
    times_differences_against_ref = [
        (d - dates[0]).days / 365.25 for d in dates[1:]
    ]  # total_seconds() to get seconds
    times_differences_against_ref = np.array(times_differences_against_ref)

    num_PS = len(PS_X_coordinates)

    rng_PS = rng[PS_X_coordinates]
    inc_PS = inc[PS_Y_coordinates, PS_X_coordinates]
    date_normal_baseline = bperp[:, PS_Y_coordinates, PS_X_coordinates]
    PS_Delta_phi_against_ref = Delta_phi_against_ref[
        :, PS_Y_coordinates, PS_X_coordinates
    ]

    # some constants
    Cq = -4 * np.pi / (wavelength * rng_PS[np.newaxis, :] * np.sin(inc_PS))
    Cv = -4 * np.pi / (wavelength * 1e3)  # 1e-3 to have mm/year

    num_dates = len(dates) - 1  # ignoring the reference

    # init variables
    q_estimation = np.zeros([num_PS], dtype=np.float32)  # constant dem error
    v_estimation = np.zeros([num_PS], dtype=np.float32)  # constant velocity
    delta_q = delta_v = 0

    APS_dzeta_model = periodogram.LinearTermModel(
        1.0, PS_Y_coordinates, np.linspace(-0.1, 0.1, 11)
    )  # odd boundaries to have 0. tested
    APS_eta_model = periodogram.LinearTermModel(
        1.0, PS_X_coordinates, np.linspace(-0.1, 0.1, 11)
    )
    atmo_model = periodogram.CompoundModel([APS_dzeta_model, APS_eta_model])
    atmo_grid = atmo_model.predict_grid()

    for iteration in range(max_iterations):
        # (a) Update estimation of altitude and velocity with estimated residuals
        q_estimation += np.asarray(delta_q)  # error in altitude estimation
        v_estimation += np.asarray(delta_v)  # linear slant range velocities

        # (b) Ferreti 2001 does stop automatically if there are no more changes
        if (
            iteration > 1
            and max(abs(delta_q)) < threshold_q
            and max(abs(delta_v)) < threshold_v
        ):
            break

        # (c) Update Zero-Baseline Steering (Delta_phi)
        # Note: In the original paper date_normal_baseline is supposed constant on the area,
        # but they mention as improvement not to do this supposition. Here we use the non constant estimation.
        Delta_phi_no_q_v_estimation = get_phi_no_q_v_estimation(
            PS_Delta_phi_against_ref,
            Cq,
            date_normal_baseline,
            q_estimation,
            Cv,
            times_differences_against_ref,
            v_estimation,
        )

        # (d) Estimate APS+residual phase on the remaining delta phi with current estimation of q and v removed
        # The minimization is independant for each date.
        p_dzeta = np.empty([num_dates], dtype=np.float32)
        p_eta = np.empty([num_dates], dtype=np.float32)
        for i in range(num_dates):
            period = periodogram.Periodogram(Delta_phi_no_q_v_estimation[i, :])
            exhaustive = period.exhaustive_gamma(atmo_grid)
            x, _ = period.refinement(atmo_model, exhaustive, no_failure=True)
            p_dzeta[i] = x[0]
            p_eta[i] = x[1]

        periodogram_for_each_date = np.exp(
            1j
            * (
                Delta_phi_no_q_v_estimation[:, :]
                - p_dzeta[:, np.newaxis] * PS_Y_coordinates[np.newaxis, :]
                - p_eta[:, np.newaxis] * PS_X_coordinates[np.newaxis, :]
            )
        )
        periodogram_for_each_date = np.sum(periodogram_for_each_date, axis=1)
        periodogram_for_each_date /= num_PS
        ak = np.angle(periodogram_for_each_date)  # constant phase values

        APS_estimated = (
            ak[:, np.newaxis]
            + p_dzeta[:, np.newaxis] * PS_Y_coordinates[np.newaxis, :]
            + p_eta[:, np.newaxis] * PS_X_coordinates[np.newaxis, :]
        )

        # (e)
        Delta_phi_estimation_noplane = Delta_phi_no_q_v_estimation - APS_estimated

        if debug_path is not None:
            os.makedirs(debug_path, exist_ok=True)
            # Save debug infos for last image of the serie
            i = Delta_phi_against_ref.shape[0] - 1
            parent_shape = Delta_phi_against_ref[0].shape
            save_debug_image(
                os.path.join(debug_path, f"APS_{iteration}_{i}.tiff"),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                APS_estimated[i, :],
            )
            save_debug_image(
                os.path.join(debug_path, f"DPHI_NOMVT_NOTOPO_{iteration}_{i}.tiff"),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                Delta_phi_no_q_v_estimation[i, :],
            )
            save_debug_image(
                os.path.join(debug_path, f"DPHI_TOPO_{iteration}_{i}.tiff"),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                (-Cq * date_normal_baseline * q_estimation[np.newaxis, :])[i, :],
            )
            save_debug_image(
                os.path.join(debug_path, f"DPHI_MVT_{iteration}_{i}.tiff"),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                (
                    Cv
                    * times_differences_against_ref[:, np.newaxis]
                    * v_estimation[np.newaxis, :]
                )[i, :],
            )
            save_debug_image(
                os.path.join(
                    debug_path, f"DPHI_NOAPS_NOMVT_NOTOPO_{iteration}_{i}.tiff"
                ),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                Delta_phi_estimation_noplane[i, :],
            )

        # (f) Extract velocity and altitude residuals
        date_coefs = np.abs(periodogram_for_each_date)
        date_coefs /= np.sum(date_coefs)
        date_coefs = np.asarray(date_coefs, dtype=np.float64)

        delta_q, delta_v, _ = velo_topo_periodogram(
            Delta_phi_estimation_noplane,
            Cq,
            date_normal_baseline,
            Cv,
            times_differences_against_ref,
            date_coefs,
            use_tensorflow=use_tensorflow,
        )
        print(f"iteration: {iteration} dq {max(abs(delta_q))} dv {max(abs(delta_v))}")

    # final residual

    Delta_phi_no_q_v_estimation = get_phi_no_q_v_estimation(
        PS_Delta_phi_against_ref,
        Cq,
        date_normal_baseline,
        q_estimation,
        Cv,
        times_differences_against_ref,
        v_estimation,
    )
    residual = psutils.wrap(Delta_phi_no_q_v_estimation - APS_estimated)
    return q_estimation, v_estimation, APS_estimated, ak, p_dzeta, p_eta, residual


# Here I add stuff to complete ferreti2001 quickly but not necessarily in a clean manner


def velo_topo_periodogram(
    phi_ps_mat,
    Cq,
    date_normal_baseline,
    Cv,
    times_differences_against_ref,
    weights_per_date=None,
    *,
    use_tensorflow=True,
):
    num_dates, num_PS = phi_ps_mat.shape

    if use_tensorflow:
        import tensorflow as tf
        import tensorflow_probability as tfp

        if weights_per_date is None:
            weights_per_date = np.ones((num_dates,), dtype=np.float64) / num_dates
        else:
            weights_per_date = np.array(weights_per_date, dtype=np.float64)
            # normalize weights
            weights_per_date /= np.sum(weights_per_date)

        # Here we use tensorflow, which has a fixed cost when calling (graph compilation),
        # but then is much faster. We have a lot of variables to minimize, thus it makes sense.
        tf_phi_ps_mat = tf.cast(phi_ps_mat, tf.float64)

        def periodogram_to_optimize(x):
            res = 0
            q = x[0:num_PS]
            v = x[num_PS:]
            for k in range(num_dates):
                res += weights_per_date[k] * tf.exp(
                    tf.dtypes.complex(
                        tf.cast(0.0, tf.float64),
                        (
                            tf_phi_ps_mat[k, :]
                            - tf.cast(Cq * date_normal_baseline[k], tf.float64) * q
                            - tf.cast(Cv * times_differences_against_ref[k], tf.float64)
                            * v
                        ),
                    )
                )
            return res

        def make_val_and_grad_fn(value_fn):
            @functools.wraps(value_fn)
            def val_and_grad(x):
                return tfp.math.value_and_gradient(value_fn, x)

            return val_and_grad

        @make_val_and_grad_fn
        def loss(x):
            per = periodogram_to_optimize(x)
            return -tf.reduce_sum(tf.math.abs(per))

        res = tfp.optimizer.lbfgs_minimize(
            loss,
            initial_position=tf.constant(np.zeros(2 * num_PS), dtype=tf.float64),
            tolerance=1e-15,
            max_iterations=10,
        )
        x = res.position
        q = x[0:num_PS].numpy()
        v = x[num_PS:].numpy()
        gammas = tf.math.abs(periodogram_to_optimize(x)).numpy().squeeze()
    else:
        q = np.zeros([num_PS], dtype=np.float32)  # constant dem error
        v = np.zeros([num_PS], dtype=np.float32)  # constant velocity
        gammas = np.zeros([num_PS], dtype=np.float32)  # temporal coherence
        v_test = periodogram.get_test_vals(300, 10)
        q_test = periodogram.get_test_vals(80, 10)
        lin_defo_model = periodogram.LinearTermModel(
            Cv, times_differences_against_ref, v_test
        )
        for h in range(num_PS):
            topo_model = periodogram.LinearTermModel(
                Cq[0, h], date_normal_baseline[:, h], q_test
            )
            defo_topo_model = periodogram.CompoundModel([lin_defo_model, topo_model])
            defo_topo_grid = defo_topo_model.predict_grid()
            period = periodogram.Periodogram(phi_ps_mat[:, h], weights_per_date)
            exhaustive = period.exhaustive_gamma(defo_topo_grid)
            x, gamma_opt = period.refinement(
                defo_topo_model, exhaustive, no_failure=True
            )
            v[h] = x[0]
            q[h] = x[1]
            gammas[h] = gamma_opt

    return q, v, gammas


def spatial_low_pass_interpolate_atmo(
    residual,
    PS_X_coordinates,
    PS_Y_coordinates,
    parent_shape,
    weights: Optional[NDArray] = None,
):
    h, w = parent_shape
    interpolated = []
    for res in residual:
        tck = scipy.interpolate.bisplrep(
            PS_Y_coordinates, PS_X_coordinates, res, weights
        )
        interp = scipy.interpolate.bisplev(np.arange(h), np.arange(w), tck)
        interpolated.append(interp)
    return interpolated


def get_atmo_full(interpolated, ak, pdzeta, peta, parent_shape):
    interpolated = np.array(interpolated)
    h, w = parent_shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    atmos = (
        ak[:, None, None]
        + pdzeta[:, None, None] * yy
        + peta[:, None, None] * xx
        + interpolated
    )
    return atmos


def final_periodogram(
    phi_ts_raster,
    atmos,
    dates,
    rng,
    inc,
    bperp,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
):
    times_differences_against_ref = [
        (d - dates[0]).days / 365.25 for d in dates[1:]
    ]  # total_seconds() to get seconds
    times_differences_against_ref = np.array(times_differences_against_ref)

    phi_no_atmo = psutils.wrap(phi_ts_raster - atmos)
    mask = ~np.isnan(phi_no_atmo[0])
    mask_sparse = scipy.sparse.coo_array(mask)
    row = mask_sparse.row
    col = mask_sparse.col
    phi_no_atmo_sparse = phi_no_atmo[:, row, col]

    rng_PS = rng[col]
    inc_PS = inc[row, col]
    date_normal_baseline = bperp[:, row, col]

    # some constants
    Cq = -4 * np.pi / (wavelength * rng_PS[np.newaxis, :] * np.sin(inc_PS))
    Cv = -4 * np.pi / (wavelength * 1e3)  # 1e-3 to have mm/year

    q, v, gammas = velo_topo_periodogram(
        phi_no_atmo_sparse,
        Cq,
        date_normal_baseline,
        Cv,
        times_differences_against_ref,
        use_tensorflow=use_tensorflow,
    )
    phi_residual = get_phi_no_q_v_estimation(
        phi_no_atmo_sparse,
        Cq,
        date_normal_baseline,
        q,
        Cv,
        times_differences_against_ref,
        v,
    )
    # assume wrapped is good estimation
    phi_residual = psutils.wrap(phi_residual)  # non linear deformation phase
    # now add back linear trend deformation
    defo_nonlinear = phi_residual / Cv  # (mm)
    defo_linear = (
        Cv * times_differences_against_ref[:, np.newaxis] * v[np.newaxis, :]
        + Cq * date_normal_baseline * q[np.newaxis, :]
    )  # (radians)
    return q, v, gammas, col, row, defo_nonlinear, defo_linear


def get_phi_no_q_v_estimation(
    phi_ps_mat, Cq, date_normal_baseline, q, Cv, times_differences_against_ref, v
):
    phi_no_q_v_estimation = (
        phi_ps_mat
        - Cq * date_normal_baseline * q[np.newaxis, :]
        - Cv * times_differences_against_ref[:, np.newaxis] * v[np.newaxis, :]
    )
    return phi_no_q_v_estimation


def full_pipeline(
    amps,
    phi_ts,
    bperp,
    inc,
    rng,
    dates,
    da_threshold=0.25,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    first_gamma_threshold=0.8,
    second_gamma_threshold: float = 0.9,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
):
    (q, v, gammas, defo_nonlinear, defo_linear, atmos) = full_pipeline_nosparse(
        amps,
        phi_ts,
        bperp,
        inc,
        rng,
        dates,
        da_threshold=da_threshold,
        max_iterations=max_iterations,
        threshold_q=threshold_q,
        threshold_v=threshold_v,
        first_gamma_threshold=first_gamma_threshold,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
    )

    # keep only good ps
    final_ps_mask = gammas > second_gamma_threshold

    h, w = final_ps_mask.shape
    col, row = np.meshgrid(np.arange(w), np.arange(h))

    return (
        q[final_ps_mask],
        v[final_ps_mask],
        gammas[final_ps_mask],
        col[final_ps_mask],
        row[final_ps_mask],
        defo_nonlinear[:, final_ps_mask],
        defo_linear[:, final_ps_mask],
    )


def full_pipeline_nosparse(
    amps,
    phi_ts,
    bperp,
    inc,
    rng,
    dates,
    da_threshold=0.25,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    first_gamma_threshold=0.8,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
):  # no clean, some things hardcoded
    print("ps candidates selection")
    # ps candidates
    _, PS_candidates_basic, PS_candidates_mask = psc.get_PS_candidates_DA(
        amps, da_threshold
    )
    PS_candidates_mask_sparse = psutils.dense_mask_to_sparse(PS_candidates_basic)
    PS_X_coordinates = PS_candidates_mask_sparse.col.reshape([-1])
    PS_Y_coordinates = PS_candidates_mask_sparse.row.reshape([-1])

    print("iterative periodogram")
    # estimate atmosphere on candidates
    Delta_phi_against_ref = np.array(phi_ts)
    (
        q_estimation,
        v_estimation,
        APS_estimated,
        ak,
        pdzeta,
        peta,
        residual,
    ) = iterative_alternate_periodogram(
        PS_X_coordinates,
        PS_Y_coordinates,
        Delta_phi_against_ref,
        bperp,
        inc,
        rng,
        dates,
        max_iterations=max_iterations,
        threshold_q=threshold_q,
        threshold_v=threshold_v,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
    )

    # filter bad candidates
    gammas_approx = np.abs(np.mean(np.exp(1j * residual), axis=0))
    good_ps = gammas_approx > first_gamma_threshold

    print("interpolate atmo")
    # interpolate atmosphere in residual to a regular grid,
    # it uses the coherence (gamma) as weights for the spline fitting
    interpolated = spatial_low_pass_interpolate_atmo(
        residual[:, good_ps],
        PS_X_coordinates[good_ps],
        PS_Y_coordinates[good_ps],
        parent_shape=Delta_phi_against_ref[0].shape,
        weights=gammas_approx[good_ps],
    )

    # add affine planes to interpolated to get the full atmo
    atmos = get_atmo_full(
        interpolated, ak, pdzeta, peta, parent_shape=Delta_phi_against_ref[0].shape
    )

    print("final periodogram")
    # do final periodogram
    q, v, gammas, col, row, defo_nonlinear, defo_linear = final_periodogram(
        Delta_phi_against_ref,
        atmos,
        dates,
        rng,
        inc,
        bperp,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
    )

    def make_array(arr):
        if arr.ndim == 2:
            return np.asarray([make_array(a) for a in arr])
        return psutils.sparse_data_to_raster(
            arr, row, col, Delta_phi_against_ref[0].shape
        )

    return (
        make_array(q),
        make_array(v),
        make_array(gammas),
        make_array(defo_nonlinear),
        make_array(defo_linear),
        atmos,
    )
