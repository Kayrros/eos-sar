import numpy as np
from eos.sar import const


def iterative_projection(orbit, gx, gy, gz, azt_init=None, max_iterations=20, tol=1.2*1e-7):
    """Solves the point of closest approach using the Newton-Raphson algorithm.
    
    Parameters
    ----------
    orbit: fitted Orbit instance 
    gx, gy, gz: Iterable  
        Geocentric coordinates
    azt_init: Iterable
           Initial guess for the azimuth time, same len as x
    max_iterations: int
            Maximum number of iterations for reaching the solution
    tol: float 
            Tolerance in seconds of azimuth time precision on the orbit 
            below which the iterations stop
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
    if azt_init is None:
        # determine which state vectors to use
        sv_times = [s['time'] for s in orbit.sv]
        start = min(sv_times)
        end = max(sv_times)
        # initial guess
        azt_curr = (start + end)/2 * np.ones((len(points),))
    else:
        azt_curr = np.array(azt_init)
    # mask on points on which to iterate
    index = np.ones((len(azt_curr), ), dtype=bool)
    # initialization of step
    dazt = np.ones_like(azt_curr)
    # Newton-Raphson iterations
    for j in range(max_iterations):
        E, dE = get_E_dE(azt_curr[index], orbit, points[index])
        dazt[index] = -E / dE
        azt_curr[index] += dazt[index]
        index = np.abs(dazt) >= tol
        if index.sum() == 0:
            break
    closest_positions = orbit.evaluate(azt_curr).reshape(-1, 3)
    # apply the cosine rule to get the incidence angle
    op = np.linalg.norm(points, axis=1)
    os = np.linalg.norm(closest_positions, axis=1)
    rng = np.linalg.norm(closest_positions - points, axis=1)
    i = np.arccos((os**2 - op**2 - rng**2) / (2 * op * rng))
    
    # support for scalar input
    if len(azt_curr) == 1: 
        return azt_curr[0], rng[0], i[0]
    else: 
        return azt_curr, rng, i


def get_E_dE(azt, orbit, M):
    """
    Parameters
    ----------
    azt : ndarray (N, )
        Azimuth times along the orbit (N, ) np.ndarray.
    orbit : Orbit instance 
    M : ndarray (N, 3 ) 
        Points in geocentric coordinates.

    Returns
    -------
    E : ndarray (N,)
        Scalar product of the speed with the LOS vector
    dE : ndarray (N, )
        Derivative of E, w.r.t. time

    Notes
    -----
    In this function is computed E = V(t).(M - Orbit(t)) the scalar product of the speed
    with the LOS vector and dE/dt  = acc(t).(M - Orbit(t))  - ||V(t)||^2.
    Formula is taken from [0]

    References
    ----------
    [0] Delft Object-oriented Radar Interferometric Software User manual.
        Available at http://doris.tudelft.nl/usermanual/node195.html

    """
    # speed
    V = orbit.evaluate(azt, order=1).reshape(-1, 3)
    # LOS vector
    D = (M - orbit.evaluate(azt)).reshape(-1, 3)
    # scalar product
    E = np.sum(V * D, axis=1).squeeze()
    # acceleration
    Acc = orbit.evaluate(azt, order=2).reshape(-1, 3)
    # scalar product
    term1 = np.sum(D * Acc, axis=1).squeeze()
    # squared speed norm
    term2 = (np.linalg.norm(V, axis=1).squeeze()) ** 2
    # combine
    dE = term1 - term2
    return E, dE

# localization functions


def iterative_localization(orbit, azt, rng, alt, gxyz_init, max_iterations=10000, tol=0.01):
    """Solves the Range-Doppler equations for a set of points using the Newton-Raphson method

    Parameters
    ----------
    orbit: fitted Orbit instance
    azt: ndarray (N,) or float 
        Time of closest approach 
    rng: ndarray (N,) or float 
        Distance to sensor
    alt: ndarray (N,) or float
        Height at which we localize the point 
    gxyz_init: tuple (gx (N,), gy (N,), gz(N,) ) 
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
    ell_axis = np.array([const.EARTH_WGS84_AXIS_A_M,
                         const.EARTH_WGS84_AXIS_A_M,
                         const.EARTH_WGS84_AXIS_B_M]).reshape(1,3)
    ell_axis = ell_axis + np.reshape(alt, (N, 1)) # (N, 3)
    # mask on points on which to iterate
    index = np.ones((N,), dtype=bool)
    step = np.ones((N, 3))
    # iterate
    for i in range(max_iterations):
        step[index] = get_step(P[index], satPos[index],
                               satV[index], rng[index], ell_axis[index])
        P[index] += step[index]
        index = np.any(np.abs(step) > tol, axis=1)
        if index.sum() == 0:
            break
    gx, gy, gz = P.squeeze().T
    return gx, gy, gz


def get_step(P, satPos, satV, rng, ell_axis):
    """Computes the Newton-Raphson step of the Range-Doppler localization
    algorithm on an array of points 

    Parameters
    ----------
    P : ndarray (N, 3)
        Geocentric coordinates of points. Current solution of the Range-Doppler equations 
    satPos : ndarray (N, 3)
        Satellite position for each point.
    satV : ndarray (N, 3)
        Satellite velocity for each point.
    rng : ndarray (N, )
        the range of each point.
    ell_axis : ndarray (N, 3)
        Earth ellipsoid axis in the x, y, z direction, incremented by the altitude of each point

    Returns
    -------
    step : ndarray (N, 3)
        Step to take in geocentric coordinates to move towards the optimal solution.

    """
    N = len(P)
    F = np.zeros((N, 3))
    delta = np.zeros((N, 3, 3))
    # vector between satellite and point
    LOS = P - satPos  # (N, 3)
    # compute F(xyz)
    F[:, 0] = np.sum(satV * LOS, axis=1)
    F[:, 1] = np.linalg.norm(LOS, axis=1) ** 2 - rng ** 2
    F[:, 2] = np.sum((P / ell_axis)**2, axis=1) - 1
    # compute the jacobian matrix
    delta[:, 0, :] = satV
    delta[:, 1, :] = 2 * LOS
    delta[:, 2, :] = 2 * P / (ell_axis ** 2)
    # find the step in xyz
    step = np.linalg.solve(delta, -F)
    return step
