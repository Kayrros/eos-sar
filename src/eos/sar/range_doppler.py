"""Range Doppler physical sensor model for projection and localization."""

import numpy as np

from eos.sar import const, geoconfig
from eos.sar.orbit import Orbit


def iterative_projection(
    orbit: Orbit,
    gx,
    gy,
    gz,
    azt_init=None,
    max_iterations=20,
    tol=1.2 * 1e-7,
    half_wavelength_f_dc=None,
):
    """
    Solves the point of closest approach using the Newton-Raphson algorithm.

    Parameters
    ----------
    orbit: Orbit
        Fitted Orbit instance
    gx, gy, gz: Iterable
        Geocentric coordinates
    azt_init: Iterable
        Initial guess for the azimuth time, same len as x
    max_iterations: int
        Maximum number of iterations for reaching the solution
    tol: float
        Tolerance in seconds of azimuth time precision on the orbit
        below which the iterations stop
    half_wavelength_f_dc: Iterable, Optional
        (lambda / 2) * f_dc, where lambda is the constant carrier wavelength
        and f_dc is the Doppler centroid per point.
        When None (the default behavior), it is assumed to be 0 everywhere, so that the
        function yields the Zero-Doppler time along the orbit.
        When specified, the function will yield the azimuth time along the orbit
        where the Doppler condition is met, i.e.
        V(t).(M - O(t)) -  (lambda / 2) * f_dc * || M - O(t) || = 0
        with O(t) the satellite position and V(t) its speed, M the 3D target on the earth surface,


    Returns
    ------
        azt: ndarray
            time of closest approach
        rng: ndarray
            distance to sensor
        i: ndarray
            incidence angle
    """
    points = np.column_stack((gx, gy, gz))
    n_points = len(points)

    if azt_init is None:
        # determine which state vectors to use
        sv_times = [s.time for s in orbit.sv]
        start = min(sv_times)
        end = max(sv_times)
        # initial guess
        azt_curr = (start + end) / 2 * np.ones((n_points,))
    else:
        azt_curr = np.atleast_1d(np.array(azt_init))
        assert len(azt_curr) == n_points

    # mask on points on which to iterate
    index = np.ones((n_points,), dtype=bool)
    # initialization of step
    dazt = np.ones_like(azt_curr)

    # checking half_wavelength_f_dc
    if half_wavelength_f_dc is not None:
        half_wavelength_f_dc = np.atleast_1d(half_wavelength_f_dc)
        assert len(half_wavelength_f_dc) == n_points

    # Newton-Raphson iterations
    for j in range(max_iterations):
        E, dE = get_E_dE(
            azt_curr[index],
            orbit,
            points[index],
            None if half_wavelength_f_dc is None else half_wavelength_f_dc[index],
        )

        dazt[index] = -E / dE
        azt_curr[index] += dazt[index]
        index = np.abs(dazt) >= tol
        if index.sum() == 0:
            break

    sat_positions = orbit.evaluate(azt_curr)

    # apply the cosine rule to get the incidence angle
    cos_i, rng = geoconfig.compute_cosi_rng(points, sat_positions)
    i = np.arccos(cos_i)

    # support for scalar input
    if len(azt_curr) == 1:
        return azt_curr[0], rng[0], i[0]

    return azt_curr, rng, i


def ascending_node_crossing_time(
    orbit: Orbit, max_iterations: int = 20, tol: float = 1.2 * 1e-7
) -> float:
    """Find the azimuth time solving `orbit(azt).z = 0`.
    The orbit instance should be defined around the solution.

    Parameters
    ----------
    orbit: fitted Orbit instance
    azt_init: float (optional)
           Initial guess for the azimuth time
    max_iterations: int
            Maximum number of iterations for reaching the solution
    tol: float
            Tolerance in seconds of azimuth time precision on the orbit
            below which the iterations stop
    Returns
    ------
    azt: float
        time of crossing the ascending node
    """
    # determine which state vectors to use
    sv_times: list[float] = [s.time for s in orbit.sv]
    start = min(sv_times)
    end = max(sv_times)

    # initial guess
    azt_curr = (start + end) / 2

    # Newton-Raphson iterations
    for _ in range(max_iterations):
        E = orbit.evaluate(azt_curr)[2]
        dE = orbit.evaluate(azt_curr, order=1)[2]

        dazt = E / dE
        azt_curr -= dazt
        if np.abs(dazt) < tol:
            break

    return azt_curr


def get_E_dE(azt, orbit: Orbit, M, half_wavelength_f_dc=None):
    """
    Get the function that needs to be 0 and its derivative.

    Parameters
    ----------
    azt : ndarray (N, )
        Azimuth times along the orbit (N, ) np.ndarray.
    orbit : Orbit instance
    M : ndarray (N, 3 )
        Points in geocentric coordinates.
    half_wavelength_f_dc: Iterable, Optional
        (lambda / 2) * f_dc, where lambda is the constant carrier wavelength
        and f_dc is the Doppler centroid per point.
        When None, it is assumed to be zero (Zero-Doppler condition.)

    Returns
    -------
    E : ndarray (N,)
        Scalar product of the speed with the LOS vector
    dE : ndarray (N, )
        Derivative of E, w.r.t. time

    Notes
    -----
    V(t).(M - O(t)) -  (lambda / 2) * f_dc * || M - O(t) || = 0 is the Doppler condition,
    where O(t) the satellite position and V(t) its speed, M the 3D target on the earth surface.

    In this function is computed
    E = V(t).(M - O(t)) -  (lambda / 2) * f_dc * || M - O(t) ||
    the function that needs to be zero to satisfy the Doppler condition.
    Its derivative is returned as well
    dE/dt  = acc(t).(M - O(t))  - ||V(t)||^2 + (lambda / 2) * f_dc V(t).(M - O(t))/|| M - O(t) ||.
    Formula is taken from [0] (and adapted for Doppler condition) and from [1] (equation 19)

    References
    ----------
    [0] Delft Object-oriented Radar Interferometric Software User manual.
        Available at http://doris.tudelft.nl/usermanual/node195.html
    [1] Small, D. and Schubert, A. (2019) Guide to Sentinel-1 Geocoding,
    Tech. Rep. UZH-S1-GC-AD.
    Available at:
    https://sentinels.copernicus.eu/documents/247904/1653442/Guide-to-Sentinel-1-Geocoding.pdf
    """
    # speed
    V = orbit.evaluate(azt, order=1)
    # LOS vector
    D = M - orbit.evaluate(azt)
    # scalar product
    V_dot_D = np.sum(V * D, axis=1)
    # acceleration
    Acc = orbit.evaluate(azt, order=2)
    # scalar product
    Acc_dot_D = np.sum(D * Acc, axis=1)
    # squared speed norm
    V_dot_V = np.sum(V * V, axis=1)

    E = V_dot_D
    dE_dt = Acc_dot_D - V_dot_V

    if half_wavelength_f_dc is not None:
        norm_D = np.linalg.norm(D, axis=1)
        E = E - half_wavelength_f_dc * norm_D
        dE_dt = dE_dt + half_wavelength_f_dc * V_dot_D / norm_D

    return E, dE_dt


# localization functions


def iterative_localization(
    orbit: Orbit, azt, rng, alt, gxyz_init, max_iterations=10000, tol=0.01
):
    """Solves the Range-Doppler equations for a set of points using \
        the Newton-Raphson method.

    Parameters
    ----------
    orbit: fitted Orbit instance
    azt: ndarray (N,) or float
        Time of closest approach
    rng: ndarray (N,) or float
        Distance to sensor
    alt: ndarray (N,) or float
        Height at which we localize the point
    gxyz_init: tuple (gx (N,), gy (N,), gz(N,))
        Initial point from which to begin iterations
    max_iterations: int
        Maximum number of iterations of Newton-Raphson
    tol: float
        Tolerance on the step in gx, gy, gz (in meters)
        iterations stop when all steps dgx, dgy and dgz are below tol
    Returns:
        gx, gy, gz: ndarray (N,)
           Localized 3D point in geocentric coordinates

    Notes
    -----
    Denote P = (gx, gy, gz) the 3D point in geocentric coordinates
    F(P) = (f(P), g(P), h(P))
    where
        f(P) = satV.(P - satPos)
            denotes the dot product between speed and the LOS
            f(P) = 0 means that the point is in the plane orthogonal to speed
        g(P) = (P - satPos)**2 - rdist**2
            denotes LOS distance squared minus range squared
            g(P) = 0 means LOS distance equals the range
        h(P) = (gx^2 + gy^2)/(a+h)^2 + (gz^2)/(b+h)^2 - 1
            denotes the ellipsoid above the earth with height h
            h(P) = 0 means that the point is on the ellipsoid (at height h)
    To find the 3D position of the point
    We need to find the root where F(P) = 0
    Linearization with Taylor expansion:
         Find the P that solve -F(P) = delta*(P-P0)
         where delta is the derivative matrix

    References
    ----------
    [0] Delft Object-oriented Radar Interferometric Software User manual.
        Available at http://doris.tudelft.nl/usermanual/node195.html
    """
    # deal with scalar case
    azt = np.atleast_1d(azt)
    rng = np.atleast_1d(rng)
    N = len(azt)
    satPos = orbit.evaluate(azt).reshape(N, 3)  # (N, 3)
    satV = orbit.evaluate(azt, order=1).reshape(N, 3)  # (N, 3)
    # P is variable that will change throughout the iterations
    P = np.column_stack(gxyz_init)
    # init
    ell_axis = np.array(
        [
            const.EARTH_WGS84_AXIS_A_M,
            const.EARTH_WGS84_AXIS_A_M,
            const.EARTH_WGS84_AXIS_B_M,
        ]
    ).reshape(1, 3)
    ell_axis = ell_axis + np.reshape(alt, (N, 1))  # (N, 3)
    # mask on points on which to iterate
    index = np.ones((N,), dtype=bool)
    step = np.ones((N, 3))
    # iterate
    for i in range(max_iterations):
        step[index] = get_step(
            P[index], satPos[index], satV[index], rng[index], ell_axis[index]
        )
        P[index] += step[index]
        index = np.any(np.abs(step) > tol, axis=1)
        if index.sum() == 0:
            break
    gx, gy, gz = P.squeeze().T
    return gx, gy, gz


def get_step(P, satPos, satV, rng, ell_axis):
    """Compute the Newton-Raphson step of the Range-Doppler localization\
    algorithm on an array of points.

    Parameters
    ----------
    P : ndarray (N, 3)
        Geocentric coordinates of points. Current solution of
        the Range-Doppler equations
    satPos : ndarray (N, 3)
        Satellite position for each point.
    satV : ndarray (N, 3)
        Satellite velocity for each point.
    rng : ndarray (N, )
        the range of each point.
    ell_axis : ndarray (N, 3)
        Earth ellipsoid axis in the x, y, z direction,
        incremented by the altitude of each point

    Returns
    -------
    step : ndarray (N, 3)
        Step to take in geocentric coordinates to move
        towards the optimal solution.
    """
    N = len(P)
    F = np.zeros((N, 3))
    delta = np.zeros((N, 3, 3))
    # vector between satellite and point
    LOS = P - satPos  # (N, 3)
    # compute F(xyz)
    F[:, 0] = np.sum(satV * LOS, axis=1)
    F[:, 1] = np.linalg.norm(LOS, axis=1) ** 2 - rng**2
    F[:, 2] = np.sum((P / ell_axis) ** 2, axis=1) - 1
    # compute the jacobian matrix
    delta[:, 0, :] = satV
    delta[:, 1, :] = 2 * LOS
    delta[:, 2, :] = 2 * P / (ell_axis**2)
    # find the step in xyz
    step = np.linalg.solve(delta, -F[..., None])[..., 0]
    return step
