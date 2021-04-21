# Sentinel-1 specific burst functions 
import numpy as np 
import pyproj 
from eos.sar import backproj 
from eos.sar import const

def fill_meta(model , bid): 
    """
    Parameters
    ----------
    model : Sentinel1Model instance
        Instance created using s1m module on a subswath of a product
    bid : integer
        Burst index in the swath that corresponds to model 
    
    Returns
    -------
    burst_metadata : dict
        Metadata necessary for further burst processing

    """
    burst_metadata = {}
    burst_metadata['state_vectors'] = model.state_vectors
    burst_metadata['burst_times'] = model.burst_times[bid]
    burst_metadata['slant_range_time'] = model.slant_range_time
    burst_metadata['azimuth_frequency'] = model.azimuth_frequency
    burst_metadata['range_frequency'] = model.range_frequency
    burst_metadata['burst_roi'] = model.burst_rois[bid]
    burst_metadata['lines_per_burst'] = model.lines_per_burst
    burst_metadata['samples_per_burst'] = model.samples_per_burst
    # temporary burst bounds taken as the swath bound
    # TODO should be replaced by actual burst bounds from gcps (in EACOP_new_attrs)
    lons, lats = model.lon_lat_bbox.boundary.xy
    burst_metadata['lon_lat_bbox'] = [(lon, lat) for lon,lat in zip(lons[:4], lats[:4])]
    return burst_metadata

def x2r(x, burst_meta): 
    """
    Parameters
    ----------
    x : np.ndarray or scalar
        x coordinate in burst referenced to the first valid column 
    burst_meta : dict 
        Metadata of burst

    Returns
    -------
    np.ndarray or scalar
        One way range expressed in meters
    """
    x0 = burst_meta['burst_roi'][0]
    Fr = burst_meta['range_frequency'] 
    tau0 = burst_meta['slant_range_time']
    return ( (x + x0)/Fr  + tau0) * const.LIGHT_SPEED_M_PER_SEC/2 

def r2x(r, burst_meta): 
    """
    Parameters
    ----------
    r : np.ndarray or scalar
        One way range expressed in meters
    burst_meta : dict
        Metadata of burst

    Returns
    -------
    np.ndarray or scalar
        x coordinate in burst referenced to the first valid column 

    """
    x0 = burst_meta['burst_roi'][0]
    Fr = burst_meta['range_frequency'] 
    tau0 = burst_meta['slant_range_time']
    return (2*r/const.LIGHT_SPEED_M_PER_SEC - tau0)*Fr - x0 

def y2ta(y, burst_meta):
    """

    Parameters
    ----------
    y : np.ndarray or scalar
        y coordinate in burst referenced to the first valid line.
    burst_meta : dict
        Metadata of burst.

    Returns
    -------
    np.ndarray or scalar
        Azimuth timestamp.

    """
    start_valid = burst_meta['burst_times'][1]
    PRF = burst_meta['azimuth_frequency']
    return y /PRF + start_valid

def ta2y(ta, burst_meta): 
    """

    Parameters
    ----------
    ta : np.ndarray or scalar
        Azimuth timestamp.
    burst_meta : dict
        Metadata of burst.

    Returns
    -------
    np.ndarray or scalar
        y coordinate in burst referenced to the first valid line.

    """
    start_valid = burst_meta['burst_times'][1]
    PRF = burst_meta['azimuth_frequency']
    return (ta - start_valid)*PRF


def burst_projection(burst_metadata, lon, lat, alt , orbit ,  apd_correction=True, 
               bistatic_correction=True, crs='epsg:4326',
               max_iterations = 20, tol = 1.2*1e-7, ): 
    """

    Parameters
    ----------
    burst_metadata : dict
        Metadata of burst.
    lon : np.ndarray or scalar
        Longitude in the crs defined by epsg. 
    lat : np.ndarray or scalar
        Latitude in the crs defined by epsg.
    alt : np.ndarray or scalar
        Altitude in the crs defined by epsg.
    orbit: eos.sar.backproj.Orbit
        Used to interpolate the position and velocity along the orbit 
    apd_correction : boolean, optional
           Atmospheric Path Delay (APD) range correction . The default is True.
    bistatic_correction : boolean, optional
        Bistatic azimuth correction. The default is True.
    crs : string, optional
        CRS in which the 3D point is given
                Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
    max_iterations : int, optional
        Maximum iterations until solution is returned.
        The default is 20.
    tol : float, optional
        Tolerance on the azimuth time step size (in seconds)
        used to stop the iterations. The default is 1.2*1e-7.
    

    Returns
    -------
    x : np.ndarray or scalar
        x coordinate in burst referenced to the first valid column.
    y : np.ndarray or scalar
        y coordinate in burst referenced to the first valid line.
    i : np.ndarray or scalar
        Incidence angle.

    """
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    alt = np.atleast_1d(alt)
    # # convert to geocentric cartesian
    transformer = pyproj.Transformer.from_crs(crs, 'epsg:4978')
    x, y, z = transformer.transform(lat, lon, alt)
    # project in the slc image
    tinit = (burst_metadata['burst_times'][1] + burst_metadata['burst_times'][2])/2 * np.ones_like(x)
    t, r, i = backproj.iterative_projection(orbit, x, y, z, tinit, 
                                            max_iterations, tol)
    if apd_correction:
        alt = alt.squeeze()
        r += (alt * alt  / 8.55e7 - alt / 3411.0 + 2.41) / np.cos(i)

    # slant range (x coordinate)
    x = (2 * r / const.LIGHT_SPEED_M_PER_SEC - burst_metadata['slant_range_time']) * burst_metadata['range_frequency']

    # bistatic residual error correction, as described by Schubert et al in
    # Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
    # Results. Remote Sens. 7, 9431-9449 (2015)
    if bistatic_correction:
        t -= (x - 0.5 * burst_metadata['samples_per_burst']) / (2 * burst_metadata['range_frequency'])     

    x = x - burst_metadata['burst_roi'][0]
    y = ta2y(t, burst_metadata)
    return x, y, i 

def burst_localization(burst_metadata, x, y, alt, orbit, apd_correction = True, bistatic_correction = True,
                       max_iterations = 10000, tol = 0.01):
    """
    

    Parameters
    ----------
    burst_metadata : dict
        Metadata of burst.
    x : np.ndarray or scalar
        x coordinate in burst referenced to the first valid column.
    y : np.ndarray or scalar
        y coordinate in burst referenced to the first valid line.
    alt : np.ndarray or scalar
        Altitude above the EARTH_WGS84 ellipsoid.
    orbit: eos.sar.backproj.Orbit
        Used to interpolate the position and velocity along the orbit
    apd_correction : boolean, optional
           Atmospheric Path Delay (APD) range correction . The default is True.
    bistatic_correction : boolean, optional
        Bistatic azimuth correction. The default is True.
    max_iterations : int, optional
        Maximum iterations until solution is returned.
        The default is 10000.
    tol : float, optional
        Tolerance on the 3D point location in the x, y, z direction (in meters)
        used to stop the iterations. The default is 0.01 meters.

    Returns
    -------
    lon : np.ndarray or scalar
        Longitude in the crs defined by epsg 4326. 
    lat : np.ndarray or scalar
        Latitude in the crs defined by epsg 4326.

    """
    AXE_A = const.EARTH_WGS84_AXIS_A_M
    AXE_B = const.EARTH_WGS84_AXIS_B_M
    # make sure we work with numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    alt = np.atleast_1d(alt)
    num_pts = len(x)
    # image coordinates to range and az time 
    r = x2r(x, burst_metadata)
    ta = y2ta(y, burst_metadata)
    if bistatic_correction: 
        # correct azimuth time
        ta += ((x + burst_metadata['burst_roi'][0]) - \
            0.5*burst_metadata['samples_per_burst'])/(2*burst_metadata['range_frequency'])
    # now evaluate satellite position and velocity 
    positions = orbit.evaluate(ta)
    velocities = orbit.evaluate(ta, order = 1)
    if apd_correction: 
        # Rough estimation of geometry 
        Lsat = np.linalg.norm(positions, axis = 1)
        # Earth radius taken at the intersection of the line joining satellite 
        # and earth center with the ellipsoid
        ERadius = np.sqrt( (AXE_A * AXE_B)**2/(AXE_B**2 * \
              (positions[:,0]**2 + positions[:, 1]**2) + (AXE_A * positions[:,2])**2 ) )\
                * Lsat
        # cosine rule 
        incidence = np.arccos((Lsat**2 - (ERadius+alt)**2 - r**2) / (2 * (ERadius + alt ) * r))
        # correct range 
        r -= (alt**2/8.55e7 - alt/3411.0 + 2.41)/np.cos(incidence)
    # initial geocentric point xyz definition
    # from lon, lat, alt to x, y, z 
    toXYZ = pyproj.Transformer.from_crs('epsg:4326', 'epsg:4978')
    # point at swath centroid, 0 altitude as init 
    lonc, latc = np.mean(burst_metadata['lon_lat_bbox'], axis = 0)
    XYZ = np.array(toXYZ.transform(lonc, latc, 0))
    # localize each point
    points3D = np.zeros((num_pts, 3))
    for j in range(num_pts): 
        XYZ = backproj.solve_range_doppler(positions[j], velocities[j], r[j] , alt[j], XYZ, max_iterations, tol)
        points3D[j] = XYZ
    points3D = points3D.squeeze() 
    tolonlat = pyproj.Transformer.from_crs('epsg:4978','epsg:4326',always_xy=True)
    lon, lat , _ = tolonlat.transform(*points3D.T)
    return lon, lat 
