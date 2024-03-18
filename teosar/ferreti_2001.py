import concurrent.futures
import functools
import multiprocessing
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
import tifffile
import tqdm
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
. Extrapolate APS on the whole grid, taking into account residual
 errors in the estimation model (which we model as APS + random error)
 and smoothing spatially these errors
. Use the periodogram to estimate new PS now that APS has been removed (threshold of 0.75)
"""


@dataclass(frozen=True)
class Ferreti2001Result:
    """Periodogram model estimated by the Ferreti method.

    The model is: (radians)
        observed = (aps + q * Cq * bperp + v * Cv * years_since_ref + residuals) mod 2pi
    """

    observations: np.ndarray
    """ (t, h, w), in radians; observed signal """
    aps: np.ndarray
    """ (t, h, w), in radians; atmosphere """
    q: np.ndarray
    """ (h, w), in m; refined topography """
    v: np.ndarray
    """ (h, w), in mm/year; linear deformation rate"""
    c0: np.ndarray
    """(h, w), in radians, constant per pixel to have a centered model"""
    bperp: np.ndarray
    """ (t, h, w), in meters; normal baseline per date """
    Cq: np.ndarray
    """ (h, w), meters^2 to radians """
    Cv: float
    """ mm to radians """
    gammas: np.ndarray
    """ (h, w), temporal coherence (in [0,1]) """
    years_since_ref: np.ndarray
    """ (t), number of years since the reference date """

    @property
    def linear_deformation_in_mm(self) -> np.ndarray:
        """(t, h, w), linear part of the deformation, in mm"""
        return self.years_since_ref[:, None, None] * self.v[None, :]

    @property
    def affine_deformation_in_mm(self) -> np.ndarray:
        """(t, h, w), affine part of the deformation, in mm"""
        return self.c0 / self.Cv + self.years_since_ref[:, None, None] * self.v[None, :]

    @property
    def residuals_in_mm(self) -> np.ndarray:
        """(t, h, w), residual signal after removing the atmosphere, the topographic component and the affine deformation. in mm"""
        in_radians = psutils.wrap(
            self.observations
            - self.aps
            - self.Cq * self.bperp * self.q
            - self.Cv * self.linear_deformation_in_mm
            - self.c0
        )
        return in_radians / self.Cv


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
    years_since_ref,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    wavelength=5.5465763 * 1e-2,
    debug_path=None,
    *,
    use_tensorflow=True,
    ncpu=1,
    batch_size=128,
):
    # Estimate LOS velocity, DEM errors and APSs on the sparse grid
    # Solving system 13 of Ferreti 2001
    # We start with errors = delta_phi, APS = 0, DEM errors = 0, velocity = 0
    # Hypothesis: velocity is PS dependent, but is constant over time
    # APS is affine on the image plane

    # Convert dates to deltas in day
    num_PS = len(PS_X_coordinates)

    rng_PS = rng[PS_X_coordinates]
    inc_PS = inc[PS_Y_coordinates, PS_X_coordinates]
    date_normal_baseline = bperp[:, PS_Y_coordinates, PS_X_coordinates]
    PS_Delta_phi_against_ref = Delta_phi_against_ref[
        :, PS_Y_coordinates, PS_X_coordinates
    ]

    # some constants
    Cq = -4 * np.pi / (wavelength * rng_PS[np.newaxis, :] * np.sin(inc_PS))
    Cq = Cq.flatten()
    Cv = -4 * np.pi / (wavelength * 1e3)  # 1e-3 to have mm/year

    num_dates = len(years_since_ref)

    # init variables
    q_estimation = np.zeros([num_PS], dtype=np.float32)  # constant dem error
    v_estimation = np.zeros([num_PS], dtype=np.float32)  # constant velocity
    delta_q = delta_v = None

    APS_dzeta_model = periodogram.LinearTermModel(
        1.0, PS_Y_coordinates, np.linspace(-0.1, 0.1, 11).tolist()
    )  # odd boundaries to have 0. tested
    APS_eta_model = periodogram.LinearTermModel(
        1.0, PS_X_coordinates, np.linspace(-0.1, 0.1, 11).tolist()
    )
    atmo_model = periodogram.CompoundModel([APS_dzeta_model, APS_eta_model])
    atmo_grid = atmo_model.predict_grid()

    for iteration in range(max_iterations):
        if iteration > 1:
            assert delta_q is not None
            assert delta_v is not None

            # (a) Update estimation of altitude and velocity with estimated residuals
            q_estimation += np.asarray(delta_q)  # error in altitude estimation
            v_estimation += np.asarray(delta_v)  # linear slant range velocities

            # (b) Ferreti 2001 does stop automatically if there are no more changes
            if max(abs(delta_q)) < threshold_q and max(abs(delta_v)) < threshold_v:
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
            years_since_ref,
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
            parent_shape = Delta_phi_against_ref[0].shape
            # Save debug infos for last image of the serie
            save_debug_image(
                os.path.join(debug_path, "APS_%d.tiff" % iteration),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                APS_estimated[-1, :],
            )
            save_debug_image(
                os.path.join(debug_path, "DPHI_NOMVT_NOTOPO_%d.tiff" % iteration),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                Delta_phi_no_q_v_estimation[-1, :],
            )
            save_debug_image(
                os.path.join(debug_path, "DPHI_TOPO_%d.tiff" % iteration),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                (-Cq * date_normal_baseline * q_estimation[np.newaxis, :])[-1, :],
            )
            save_debug_image(
                os.path.join(debug_path, "DPHI_MVT_%d.tiff" % iteration),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                (Cv * years_since_ref[:, np.newaxis] * v_estimation[np.newaxis, :])[
                    -1, :
                ],
            )
            save_debug_image(
                os.path.join(debug_path, "DPHI_NOAPS_NOMVT_NOTOPO_%d.tiff" % iteration),
                PS_X_coordinates,
                PS_Y_coordinates,
                parent_shape,
                Delta_phi_estimation_noplane[-1, :],
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
            years_since_ref,
            date_coefs,
            use_tensorflow=use_tensorflow,
            ncpu=ncpu,
            batch_size=batch_size,
        )
        print(f"iteration: {iteration} dq {max(abs(delta_q))} dv {max(abs(delta_v))}")

    # final residual

    Delta_phi_no_q_v_estimation = get_phi_no_q_v_estimation(
        PS_Delta_phi_against_ref,
        Cq,
        date_normal_baseline,
        q_estimation,
        Cv,
        years_since_ref,
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
    years_since_ref,
    weights_per_date=None,
    *,
    use_tensorflow=True,
    ncpu=1,
    batch_size=128,
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
        tf_Cq = tf.cast(Cq, tf.float64)
        tf_date_normal_baseline = tf.cast(date_normal_baseline, tf.float64)

        @tf.function(jit_compile=True)
        def periodogram_to_optimize(
            q_b, v_b, tf_Cq_b, tf_date_normal_baseline_b, tf_phi_ps_mat_b
        ):  # _b stands for batch
            res = 0
            for k in range(num_dates):
                res += weights_per_date[k] * tf.exp(
                    tf.dtypes.complex(
                        tf.cast(0.0, tf.float64),
                        (
                            tf_phi_ps_mat_b[k, :]
                            - tf_Cq_b * tf_date_normal_baseline_b[k] * q_b
                            - tf.cast(Cv * years_since_ref[k], tf.float64) * v_b
                        ),
                    )
                )
            return res

        def make_val_and_grad_fn(value_fn):
            @functools.wraps(value_fn)
            def val_and_grad(x):
                return tfp.math.value_and_gradient(value_fn, x)

            return val_and_grad

        @tf.function(jit_compile=True)
        def subloss(q_b, v_b, tf_Cq_slice, tf_date_slice, tf_phi_ps_slice):
            per = periodogram_to_optimize(
                q_b, v_b, tf_Cq_slice, tf_date_slice, tf_phi_ps_slice
            )
            return -tf.reduce_sum(tf.math.abs(per))

        # slice in batches
        start_indices = np.arange(0, num_PS, batch_size)
        end_indices = np.append(start_indices[1:], num_PS)
        q = np.zeros((num_PS,), dtype=np.float64)
        v = np.zeros((num_PS,), dtype=np.float64)
        for s, e in tqdm.tqdm(
            zip(start_indices, end_indices), total=len(start_indices)
        ):
            nps = e - s
            tf_Cq_slice = tf_Cq[s:e]
            tf_date_slice = tf_date_normal_baseline[:, s:e]
            tf_phi_ps_slice = tf_phi_ps_mat[:, s:e]

            @make_val_and_grad_fn
            def loss(x):
                q_b = x[0:nps]
                v_b = x[nps:]
                return subloss(q_b, v_b, tf_Cq_slice, tf_date_slice, tf_phi_ps_slice)

            res = tfp.optimizer.lbfgs_minimize(
                loss,
                initial_position=tf.constant(np.zeros(2 * nps), dtype=tf.float64),
                tolerance=1e-15,
                max_iterations=10,
            )

            x = res.position
            q[s:e] = x[0:nps].numpy()
            v[s:e] = x[nps:].numpy()

        gammas = (
            tf.math.abs(
                periodogram_to_optimize(
                    q, v, tf_Cq, tf_date_normal_baseline, tf_phi_ps_mat
                )
            )
            .numpy()
            .squeeze()
        )
    else:
        q = np.zeros([num_PS], dtype=np.float32)  # constant dem error
        v = np.zeros([num_PS], dtype=np.float32)  # constant velocity
        gammas = np.zeros([num_PS], dtype=np.float32)  # temporal coherence
        v_test = periodogram.get_test_vals(300, 10)
        q_test = periodogram.get_test_vals(80, 10)
        lin_defo_model = periodogram.LinearTermModel(Cv, years_since_ref, v_test)
        mp_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=ncpu, mp_context=mp_context
        ) as executor:
            future_to_ps = {
                executor.submit(
                    process_ps,
                    Cq[h],
                    date_normal_baseline[:, h],
                    q_test,
                    lin_defo_model,
                    phi_ps_mat[:, h],
                    weights_per_date,
                ): h
                for h in range(num_PS)
            }

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_ps), total=num_PS
            ):
                h = future_to_ps[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f"PS {h} generated an exception: {exc}")
                else:
                    _v, _q, gamma_opt = data
                    v[h] = _v
                    q[h] = _q
                    gammas[h] = gamma_opt
    return q, v, gammas


def process_ps(
    Cq_ps, date_normal_baseline_ps, q_test, lin_defo_model, phi_ps, weights_per_date
):
    topo_model = periodogram.LinearTermModel(Cq_ps, date_normal_baseline_ps, q_test)
    defo_topo_model = periodogram.CompoundModel([lin_defo_model, topo_model])
    defo_topo_grid = defo_topo_model.predict_grid()
    period = periodogram.Periodogram(phi_ps, weights_per_date)
    exhaustive = period.exhaustive_gamma(defo_topo_grid)
    x, gamma_opt = period.refinement(defo_topo_model, exhaustive, no_failure=True)
    _v = x[0]
    _q = x[1]
    return (_v, _q, gamma_opt)


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
    years_since_ref,
    rng,
    inc,
    bperp,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
    ncpu=1,
    batch_size=128,
):
    n, h, w = phi_ts_raster.shape

    phi_no_atmo = psutils.wrap(phi_ts_raster - atmos)

    # some constants
    Cq = -4 * np.pi / (wavelength * rng[np.newaxis, :] * np.sin(inc))
    Cv = -4 * np.pi / (wavelength * 1e3)  # 1e-3 to have mm/year

    Cq = Cq.reshape((-1))
    phi_no_atmo = phi_no_atmo.reshape((n, -1))
    bperp = bperp.reshape((n, -1))

    q, v, gammas = velo_topo_periodogram(
        phi_no_atmo,
        Cq,
        bperp,
        Cv,
        years_since_ref,
        use_tensorflow=use_tensorflow,
        ncpu=ncpu,
        batch_size=batch_size,
    )

    q = q.reshape((h, w))
    v = v.reshape((h, w))
    gammas = gammas.reshape((h, w))
    return q, v, gammas


def get_phi_no_q_v_estimation(
    phi_ps_mat, Cq, date_normal_baseline, q, Cv, years_since_ref, v
):
    phi_no_q_v_estimation = (
        phi_ps_mat
        - Cq[np.newaxis, :] * date_normal_baseline * q[np.newaxis, :]
        - Cv * years_since_ref[:, np.newaxis] * v[np.newaxis, :]
    )
    return phi_no_q_v_estimation


def full_pipeline(
    amps,
    Delta_phi_against_ref,
    bperp,
    inc,
    rng,
    years_since_ref,
    da_threshold=0.25,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    first_gamma_threshold=0.8,
    second_gamma_threshold: float = 0.9,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
    ncpu=1,
    batch_size=128,
):
    result = run(
        amps,
        Delta_phi_against_ref,
        bperp,
        inc,
        rng,
        years_since_ref,
        da_threshold=da_threshold,
        max_iterations=max_iterations,
        threshold_q=threshold_q,
        threshold_v=threshold_v,
        first_gamma_threshold=first_gamma_threshold,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
        ncpu=ncpu,
        batch_size=batch_size,
    )

    # keep only good ps
    final_ps_mask = result.gammas > second_gamma_threshold

    h, w = final_ps_mask.shape
    col, row = np.meshgrid(np.arange(w), np.arange(h))

    return (
        result.q[final_ps_mask],
        result.v[final_ps_mask],
        result.gammas[final_ps_mask],
        col[final_ps_mask],
        row[final_ps_mask],
        result.residuals_in_mm[:, final_ps_mask],
        result.linear_deformation_in_mm[:, final_ps_mask],
    )


def get_psc_coords(amps, da_threshold):
    _, PS_candidates_basic, _ = psc.get_PS_candidates_DA(amps, da_threshold)
    PS_candidates_mask_sparse = psutils.dense_mask_to_sparse(PS_candidates_basic)

    PS_X_coordinates = PS_candidates_mask_sparse.col.reshape([-1])
    PS_Y_coordinates = PS_candidates_mask_sparse.row.reshape([-1])

    return PS_X_coordinates, PS_Y_coordinates


def get_atmos(
    PS_X_coordinates,
    PS_Y_coordinates,
    Delta_phi_against_ref,
    bperp,
    inc,
    rng,
    years_since_ref,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    first_gamma_threshold=0.8,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
    ncpu=1,
    batch_size=128,
):
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
        years_since_ref,
        max_iterations=max_iterations,
        threshold_q=threshold_q,
        threshold_v=threshold_v,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
        ncpu=ncpu,
        batch_size=batch_size,
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
    return atmos


def run(
    amps,
    Delta_phi_against_ref,
    bperp,
    inc,
    rng,
    years_since_ref,
    da_threshold=0.25,
    max_iterations=10,
    threshold_q=0.7,
    threshold_v=0.1,
    first_gamma_threshold=0.8,
    wavelength=5.5465763 * 1e-2,
    *,
    use_tensorflow=True,
    ncpu=1,
    batch_size=128,
) -> Ferreti2001Result:
    print("ps candidates selection")
    # ps candidates
    PS_X_coordinates, PS_Y_coordinates = get_psc_coords(amps, da_threshold)
    del amps

    print("iterative periodogram")
    # estimate atmosphere on candidates
    Delta_phi_against_ref = np.array(Delta_phi_against_ref)
    atmos = get_atmos(
        PS_X_coordinates,
        PS_Y_coordinates,
        Delta_phi_against_ref,
        bperp,
        inc,
        rng,
        years_since_ref,
        max_iterations=max_iterations,
        threshold_q=threshold_q,
        threshold_v=threshold_v,
        first_gamma_threshold=first_gamma_threshold,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
        ncpu=ncpu,
        batch_size=batch_size,
    )
    del PS_X_coordinates
    del PS_Y_coordinates

    print("final periodogram")
    # do final periodogram
    q, v, gammas = final_periodogram(
        Delta_phi_against_ref,
        atmos,
        years_since_ref,
        rng,
        inc,
        bperp,
        wavelength=wavelength,
        use_tensorflow=use_tensorflow,
        ncpu=ncpu,
        batch_size=batch_size,
    )

    Cq = -4 * np.pi / (wavelength * rng[np.newaxis, :] * np.sin(inc))
    Cv = -4 * np.pi / (wavelength * 1e3)  # 1e-3 to have mm/year

    # compute c0 such that the periodogram is centered on 0
    per = (
        Delta_phi_against_ref
        - atmos
        - Cq * bperp * q
        - Cv * years_since_ref[:, None, None] * v[None, :]
    )
    c0 = np.angle(np.mean(np.exp(1j * per), axis=0))

    return Ferreti2001Result(
        observations=Delta_phi_against_ref,
        aps=atmos,
        q=q,
        v=v,
        c0=c0,
        gammas=gammas,
        years_since_ref=years_since_ref,
        bperp=bperp,
        Cq=Cq,
        Cv=Cv,
    )
