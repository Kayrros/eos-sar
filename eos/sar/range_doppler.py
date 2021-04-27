import numpy as np
from eos.sar import const


def iterative_projection(orbit, x, y, z, tinit=None, max_iterations=20, tol=1.2*1e-7):
    """Solves the point of closest approach using the Newton-Raphson algorithm
    Parameters
    ----------
    orbit: fitted Orbit instance 
    x, y, z: Iterable  
        Geocentric coordinates
    tinit: Iterable
           Initial guess for the azimuth time, same len as x
    max_iterations: int
            Maximum number of iterations for reaching the solution
    tol: float 
            Tolerance in seconds of azimuth time precision on the orbit 
            below which the iterations stop
    Returns
    ------
        t: ndarray 
            time of closest approach 
        r: ndarray 
            distance to sensor
        i: ndarray 
            incidence angle
    """

    points = np.column_stack((x, y, z))
    if tinit is None:
        # determine which state vectors to use
        sv_times = [s['time'] for s in orbit.sv]
        start = min(sv_times)
        end = max(sv_times)
        # initial guess
        tcurr = (start + end)/2 * np.ones((len(points),))
    else:
        tcurr = np.array(tinit)
    # mask on points on which to iterate
    index = np.ones((len(tcurr), ), dtype=bool)
    # initialization of step
    dt = np.ones_like(tcurr)
    # Newton-Raphson iterations
    for j in range(max_iterations):
        E, dE = get_E_dE(tcurr[index], orbit, points[index])
        dt[index] = -E / dE
        tcurr[index] += dt[index]
        index = np.abs(dt) >= tol
        if index.sum() == 0:
            break
    closest_positions = orbit.evaluate(tcurr).reshape(-1, 3)
    # apply the cosine rule to get the incidence angle
    op = np.linalg.norm(points, axis=1)
    os = np.linalg.norm(closest_positions, axis=1)
    ps = np.linalg.norm(closest_positions - points, axis=1)
    incidence_angles = np.arccos((os**2 - op**2 - ps**2) / (2 * op * ps))
    return tcurr.squeeze(), ps.squeeze(), incidence_angles.squeeze()


def get_E_dE(t, orbit, M):
    """
    Parameters
    ----------
    t : ndarray (N, )
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
    V = orbit.evaluate(t, order=1).reshape(-1, 3)
    # LOS vector
    D = (M - orbit.evaluate(t)).reshape(-1, 3)
    # scalar product
    E = np.sum(V * D, axis=1).squeeze()
    # acceleration
    Acc = orbit.evaluate(t, order=2).reshape(-1, 3)
    # scalar product
    term1 = np.sum(D * Acc, axis=1).squeeze()
    # squared speed norm
    term2 = (np.linalg.norm(V, axis=1).squeeze()) ** 2
    # combine
    dE = term1 - term2
    return E, dE

# localization functions


def iterative_localization(orbit, t, r, alt, xyz_init, max_iterations=10000, tol=0.01):
    """Solves the Range-Doppler equations for a set of points using the Newton-Raphson method

    Parameters
    ----------
    orbit: fitted Orbit instance
    t: ndarray (N,) or float 
        Time of closest approach 
    r: ndarray (N,) or float 
        Distance to sensor
    alt: ndarray (N,) or float
        Height at which we localize the point 
    xyz_init: ndarray (N,3)
        Initial xyz point from which to begin iterations 
    max_iterations: int
        Maximum number of iterations of Newton-Raphson
    tol: float
        Tolerance on the step in x, y, z (in meters)
        iterations stop when all steps dx, dy and dz are below tol 
    Returns: 
        xyz: ndarray (N, 3)
           Localized 3D point in geocentric coordinates

    Notes
    -----
    F(xyz) = (f(xyz), g(xyz), h(xyz))
    where 
        f(xyz) = satV.(xyz - satPos) 
            denotes the dot product between speed and the LOS
            f(xyz) = 0 means that the point is in the plane orthogonal to speed
        g(xyz) = (xyz - satPos)**2 - rdist**2 
            denotes LOS distance squared minus range squared
            g(xyz) = 0 means LOS distance equals the range
        h(xyz) = (x^2 +y^2)/(a+h)^2 + (z^2)/(b+h)^2 - 1 
            denotes the ellipsoid above the earth with height h  
            h(xyz) = 0 means that the point is on the ellipsoid (at height h)
    To find the 3D position of the point
    We need to find the root where F(xyz) = 0 
    Linearization with Taylor expansion:  
         Find the xyz that solve -F(xyz0) = delta*(xyz-xyz0)
         where delta is the derivative matrix

    References
    ----------
    [0] Delft Object-oriented Radar Interferometric Software User manual.
        Available at http://doris.tudelft.nl/usermanual/node195.html
    """
    # deal with scalar case
    t = np.atleast_1d(t)
    r = np.atleast_1d(r)
    N = len(t)
    satPos = orbit.evaluate(t).reshape(N, 3)  # (N, 3)
    satV = orbit.evaluate(t, order=1).reshape(N, 3)  # (N, 3)
    # xyz is variable that will change throughout the iterations
    xyz = xyz_init.copy().reshape(N, 3)
    # init
    ell_axis = np.outer(const.EARTH_WGS84_AXIS_A_M +
                        alt, np.ones(3)).reshape(N, 3)
    ell_axis[:, 2] = const.EARTH_WGS84_AXIS_B_M + alt  # (N, 3)
    # mask on points on which to iterate
    index = np.ones((N,), dtype=bool)
    step = np.ones((N, 3))
    # iterate
    for i in range(max_iterations):
        step[index] = get_step(xyz[index], satPos[index],
                               satV[index], r[index], ell_axis[index])
        xyz[index] += step[index]
        index = np.any(np.abs(step) > tol, axis=1)
        if index.sum() == 0:
            break
    return xyz.squeeze()


def get_step(xyz, satPos, satV, r, ell_axis):
    """Computes the Newton-Raphson step of the Range-Doppler localization
    algorithm on an array of points 

    Parameters
    ----------
    xyz : ndarray (N, 3)
        Geocentric coordinates of points. Current solution of the Range-Doppler equations 
    satPos : ndarray (N, 3)
        Satellite position for each point.
    satV : ndarray (N, 3)
        Satellite velocity for each point.
    r : ndarray (N, )
        the range of each point.
    ell_axis : ndarray (N, 3)
        Earth ellipsoid axis in the x, y, z direction, incremented by the altitude of each point

    Returns
    -------
    step : ndarray (N, 3)
        Step to take in geocentric coordinates to move towards the optimal solution.

    """
    N = len(xyz)
    F = np.zeros((N, 3))
    delta = np.zeros((N, 3, 3))
    # vector between satellite and point
    LOS = xyz - satPos  # (N, 3)
    # compute F(xyz)
    F[:, 0] = np.sum(satV * LOS, axis=1)
    F[:, 1] = np.linalg.norm(LOS, axis=1) ** 2 - r ** 2
    F[:, 2] = np.sum((xyz / ell_axis)**2, axis=1) - 1
    # compute the jacobian matrix
    delta[:, 0, :] = satV
    delta[:, 1, :] = 2 * LOS
    delta[:, 2, :] = 2 * xyz / (ell_axis ** 2)
    # find the step in xyz
    step = np.linalg.solve(delta, -F)
    return step
