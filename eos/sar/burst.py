# Sentinel-1 specific burst functions 
import numpy as np 
import pyproj 
from . import backproj 

LIGHT_SPEED = 299792458

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
    return ( (x + x0)/Fr  + tau0) * LIGHT_SPEED/2 

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
    return (2*r/LIGHT_SPEED - tau0)*Fr - x0 

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


def burstprojection(burst_metadata, lon, lat, alt ,  apd_correction=True, 
               bistatic_correction=True, epsg=4326, degree = 11, iterative = True, 
               max_iterations = 20, tol = 1.2*1e-7): 
    """

    Parameters
    ----------
    burst_metadata : dict
        Metadata of burst.
    lon : np.ndarray or scalar
        longitude in the crs defined by epsg. 
    lat : np.ndarray or scalar
        latitude in the crs defined by epsg.
    alt : np.ndarray or scalar
        altitude in the crs defined by epsg.
    apd_correction : boolean, optional
           Atmospheric Path Delay (APD) range correction . The default is True.
    bistatic_correction : boolean, optional
        Bistatic azimuth correction. The default is True.
    epsg : int, optional
        EPSG code of the coordinate system used for `lon` and `lat`
                Defaults to 4326 (i.e. WGS 84 - 'lonlat').
    degree : int, optional
        degree of the polynomial fitting the orbit. The default is 11.
    iterative : boolean, optional
        Enables the iterative(Newton) projection algorithm. The default is True.
    max_iterations : int, optional
        Ignored if iterative is False, maximum iterations until solution is returned.
        The default is 20.
    tol : float, optional
        Ignored if iterative is False, tolerance on the azimuth time step size (in seconds)
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
    transformer = pyproj.Transformer.from_crs('epsg:{}'.format(epsg), 'epsg:4978')
    x, y, z = transformer.transform(lat, lon, alt)
    # orbit
    orbit = backproj.Orbit(burst_metadata['state_vectors'], degree = degree )
    # project in the slc image

    if iterative:     
        tinit = (burst_metadata['burst_times'][1] + burst_metadata['burst_times'][2])/2 * np.ones_like(x)
        t, r, i = backproj.iterative_projection(orbit, x, y, z, tinit, 
                                                max_iterations, tol)
    else: 
        # start with invalid lines
        start =  burst_metadata['burst_times'][0]
        # end with invalid lines 
        end = burst_metadata['lines_per_burst']/burst_metadata['azimuth_frequency'] + start
        # take small margin 
        margin = 0.05 * (end - start)
        t, r, i  = backproj.closest_approach(orbit, x, y, z, start - margin,  end + margin)
    if apd_correction:
        alt = alt.squeeze()
        r += (alt * alt  / 8.55e7 - alt / 3411.0 + 2.41) / np.cos(i)

    # slant range (x coordinate)
    x = (2 * r / LIGHT_SPEED - burst_metadata['slant_range_time']) * burst_metadata['range_frequency']

    # bistatic residual error correction, as described by Schubert et al in
    # Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
    # Results. Remote Sens. 7, 9431-9449 (2015)
    if bistatic_correction:
        t -= (x - 0.5 * burst_metadata['samples_per_burst']) / (2 * burst_metadata['range_frequency'])     

    x = x - burst_metadata['burst_roi'][0]
    y = ta2y(t, burst_metadata)
    return x, y, i 
