import numpy as np
from eos.sar import cheb 
from eos.sar import const 

class Orbit:
    """
    Orbit object encapsulating the position variation with time,
    as well the possibility to get the nth derivative ( for speed and acceleration for ex)
    """
    def __init__(self, state_vectors, degree = 11):
        """
        Constructor
        Parameters
        ----------
        state_vectors: list of dict
                        List of state vectors (time, position, velocity)
        degree: int 
            the degree of the polynomial
        """
        self.sv = state_vectors
        self.degree = degree
        self.fit() 
   
  
    def fit(self):
        """
        Fit the orbit representation on the samples
        """
        self.coeffs, self.cheb_domain = cheb.build_cheb_interp(self.sv, self.degree)
        # Also store the speed/acc coefficients 
        self.speed_coeffs = cheb.get_diff_coeffs(self.coeffs, self.cheb_domain, der = 1 )
        self.acc_coeffs = cheb.get_diff_coeffs(self.speed_coeffs, self.cheb_domain, der = 1)
    
    def evaluate(self, t):
        """Evaluate the position of satellite along the orbit at time t
        Parameters
        ----------
        t: numpy.1d array (n, )
           The time on which to evaluate the position of satellite along orbit
        Returns:
        -------
        (n, 3) numpy.ndarray 
            position of satellite for each azimuth time provided
        """
        return cheb.evaluate_cheb_interp(t, self.coeffs, self.cheb_domain)
        
    def evaluate_der(self, t, der = 1):
        
        """Evaluate the derivative of the orbit 
        Parameters: 
        ----------
        t: numpy.1d array (n, )
           The time on which to evaluate the orbit derivative 
        der: the derivative order
        Returns
        -------
        (n, 3) numpy.ndarray 
                 speed/acceleration/higher order derivative
        """
        if der == 1:
            der_coeffs = self.speed_coeffs
        elif der == 2:
            der_coeffs = self.acc_coeffs
        else: 
            # estimate coeffs
            der_coeffs = cheb.get_diff_coeffs(self.coeffs, self.cheb_domain, der = der )
        return cheb.evaluate_cheb_interp(t, der_coeffs, self.cheb_domain)    

# projection functions 
def distance_linear_interp(t0, t1, P0, P1, M):
    """Compute the time that minimizes the distance between M and the line
    defined by (P0, P1). The function is vectorized.

    Parameters
    ----------
    t0: ndarray (n,)
    t1: ndarray (n,)
    P0: ndarray (n, 3)
    P1: ndarray (n, 3)
    M: ndarray (n, 3)

    Returns
    -------
    t: ndarray (n,)
    P: ndarray (n, 3)
        Sensor position when distance is minimized
    d: ndarray (n,)
    """
    P0P1 = P1 - P0
    MP0 = P0 - M

    l = - (MP0[:, 0] * P0P1[:, 0] +
           MP0[:, 1] * P0P1[:, 1] +
           MP0[:, 2] * P0P1[:, 2]) / (
                P0P1[:, 0] * P0P1[:, 0] +
                P0P1[:, 1] * P0P1[:, 1] +
                P0P1[:, 2] * P0P1[:, 2])

    t = t0 + l * (t1 - t0)
    P = P0 + l[..., np.newaxis] * P0P1
    d = np.linalg.norm(P - M, axis=1)

    return t, P, d

def closest_approach(orbit, xs, ys, zs, start, end):
    """Find when and at what distance the sensor flew over a list of points

    Parameters
    ----------
    orbit: Orbit instance
        Already fitted orbit instance containing state-vectors interpolation
    xs, ys, zs: ndarray
        Cartesian coordinates the fixed geocentric frame
    start, end: float
        Time domain of interest (with margin)

    Returns
    -------
    t: ndarray
        Time of closest approach
    r: ndarray
        Distance to sensor
    i: ndarray
        Incidence angle
    """
    points = np.column_stack((xs, ys, zs))

    # interpolate orbit
    # one step every 0.0025s which is about a motion of 20m for the satellite
    # this avoids using complicated formulas for minimum distances and prevents
    # the apparition of oscillations
    # takes one step margin after end to include it in times
    times = np.arange(start, end + 0.0025, 0.0025)
    positions = orbit.evaluate(times)
    # find position before/after in orbit that minimizes the distance with
    # each point using ternary search; we assume array is unimodal
    n_pts = points.shape[0]
    n_t = times.shape[0]

    ii = np.zeros((n_pts,), dtype=int)
    ll = np.full((n_pts,), n_t - 1, dtype=int)

    n_iter = np.ceil(np.log(n_t) / np.log(3/2))
    iter = 0

    while iter < n_iter:
        delta = np.round((ll - ii) / 3).astype(int)

        jj = ii + delta
        kk = ll - delta

        distances_jj = np.linalg.norm(positions[jj, :] - points, axis=1)
        distances_ll = np.linalg.norm(positions[kk, :] - points, axis=1)

        mask = distances_jj < distances_ll

        ii[~mask] = jj[~mask]
        ll[mask] = kk[mask]

        iter += 1

    # compute now the minimum distance using linear interpolation
    t0 = times[ii]
    t1 = times[ll]

    P0 = positions[ii]
    P1 = positions[ll]

    t, closest_positions, distances = distance_linear_interp(
        t0, t1, P0, P1, points)

    # apply the cosine rule to get the incidence angle
    op = np.linalg.norm(points, axis=1)
    os = np.linalg.norm(closest_positions, axis=1)
    ps = np.linalg.norm(closest_positions - points, axis=1)
    incidence_angles = np.arccos((os**2 - op**2 - ps**2) / (2 * op * ps))

    return t.squeeze(), distances.squeeze(), incidence_angles.squeeze()

def iterative_projection(orbit, x, y, z, tinit = None, max_iterations = 20, tol = 1.2*1e-7 ): 
    """Solves the point of closest approach using the Newton-Raphson algorithm
    Parameters
    ----------
    orbit: fitted Orbit instance 
    x, y, z: Iterable  
        geocentric coordinates
    tinit: Iterable
           Initial guess for the azimuth time, same len as x
    max_iterations: int
            maximum number of iterations for reaching the solution
    tol: float 
            the tolerance in seconds of azimuth time precision on the orbit 
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
        sv_times = np.asarray([s['time'] for s in orbit.sv])
        start = sv_times.min()
        end = sv_times.max()
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
        dt[index] = -E/ dE
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
        The scalar product of the speed with the LOS vector
    dE : ndarray (N, )
        The derivative of E w.r.t. time
    
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
    V = orbit.evaluate_der(t, der = 1).reshape(-1, 3)
    # LOS vector
    D = (M - orbit.evaluate(t)).reshape(-1, 3)
    # scalar product 
    E = np.sum(V * D, axis = 1).squeeze()
    # acceleration 
    Acc = orbit.evaluate_der(t, der = 2).reshape(-1, 3)
    # scalar product 
    term1 = np.sum(D * Acc, axis = 1 ).squeeze()
    # squared speed norm
    term2 = (np.linalg.norm(V, axis = 1 ).squeeze() ) **2
    # combine 
    dE = term1 - term2 
    return E, dE   

# localization functions 
def solve_range_doppler(satPos, satV, rdist, h_point, initial_xyz
            , max_iterations = 10000,
            tol = 0.01 ): 
    """Solves the Range-Doppler equations for ONE point using the Newton-Raphson method
    
    Parameters
    ----------
    satPos: 1darray (3,)
           The geocentric position of the satellite [x, y, z]  
    satV: 1darray (3,)
        The speed of the satellite  [Vx, Vy, Vz]
    rdist: float
        The range distance from the point to the satellite in meters
    h_point: float
        The height at which we localize the point 
    initial_xyz: 1darray (3, )
        the initial xyz point from which to begin iterations 
    max_iterations: int
        The maximum number of iterations of Newton-Raphson
    tol: float
        the tolerance on the step in x, y, z (in meters)
        iterations stop when all steps dx, dy and dz are below tol 
    Returns: 
        xyz: the localized 3D point in geocentric coordinates
        
    Notes
    -----
    F(xyz) = (f(xyz), g(xyz), h(xyz))
    where 
        f(xyz) = satV.(xyz - satPos) 
            denotes the dot product between speed and the LOS
            f(xyz) = 0 means that the point is in the plane ortogonal to speed
        g(xyz) = (xyz - satPos)**2 - rdist**2 
            denotes LOS distance squared minus range squared
            g(xyz) = 0 means LOS distance equals the range
    h(xyz) = (x^2 +y^2)/(a+h)^2 + (z^2)/(b+h)^2 - 1 
            denotes the ellipsoid above the earth with height h  
            h(xyz) = 0 means that the point is on the ellipsoid (at height h)
    To find the 3D position of the point
    We need to find the root where F(xyz) = 0 
    Linearization with Taylor expansion:  
         Find the xyz that solve -F(xyz0) = A*(xyz-xyz0)
         where A is the derivative matrix
    
    References
    ----------
    [0] Delft Object-oriented Radar Interferometric Software User manual.
        Available at http://doris.tudelft.nl/usermanual/node195.html
    """
    AXE_A = const.EARTH_WGS84_AXIS_A_M
    AXE_B = const.EARTH_WGS84_AXIS_B_M
    # xyz is variable that will change throughout the iterations
    xyz = initial_xyz.copy()
    # iterate 
    for i in range(max_iterations):
        # compute -F(xyz)
        distance_sat_xyz = np.array([xyz[0]-satPos[0], xyz[1]-satPos[1], xyz[2]-satPos[2]])
        F = np.zeros(3)
        F[0] = - satV.dot(distance_sat_xyz)
        F[1] = - (np.linalg.norm(distance_sat_xyz) ** 2 - rdist** 2)
        F[2] = - (((xyz[0] * xyz[0] + xyz[1] * xyz[1]) / ((h_point + AXE_A) ** 2)) + (
                    (xyz[2] * xyz[2]) / ((h_point + AXE_B) ** 2)) - 1)
        # compute A the jacobian matrix 
        delta = np.zeros((3, 3))
        delta[0, :] = satV
        delta[1, :] = 2 * distance_sat_xyz
        delta[2, :2] = 2 * xyz[:2] / ((AXE_A + h_point) ** 2)
        delta[2, 2] = 2 * xyz[2] / ((AXE_B + h_point) ** 2)
        # find the step in xyz
        step = np.linalg.solve(delta, F)
        xyz += step
        if np.abs(step[0]) <= tol and np.abs(step[1]) <= tol and np.abs(step[2]) <= tol:
            break
    return xyz

