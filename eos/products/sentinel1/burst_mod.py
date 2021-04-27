import numpy as np
import pyproj
from eos.sar import mod
from eos.sar import range_doppler
from eos.sar import const

def fill_meta(model, bid):
    """
    Parameters
    ----------
    model : Sentinel1Model instance
        Instance created using s1m module on a subswath of a product.
    bid : integer
        Burst index in the swath that corresponds to model. 

    Returns
    -------
    burst_metadata : dict
        Metadata necessary for further burst processing.

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
    burst_metadata['azimuth_anx_time'] = model.burst_azimuth_anx_times[bid]
    burst_metadata['lon_lat_bbox'] = model.burst_lon_lat_bboxes[bid]
    return burst_metadata

class S1BurstModel(mod.SensorModel): 
    
    def __init__(self, burst_meta): 
        # time of first valid row
        first_row_time = burst_meta['burst_times'][1]
        # time of first valid col
        first_col_time = burst_meta['slant_range_time'] + burst_meta['burst_roi'][0] / burst_meta['range_frequency']
        super(S1BurstModel, self).__init__(burst_meta['state_vectors']
        , burst_meta['azimuth_frequency'], burst_meta['range_frequency'],
        first_row_time, first_col_time)
        self.__dict__.update(burst_meta)
    
    def projection(self, x, y, alt, apd_correction=True,
                         bistatic_correction=True, crs='epsg:4326',
                         max_iterations=20, tol=1.2*1e-7, ):
        """
    
        Parameters
        ----------
        x, y : np.ndarray or scalar
            Coordinates in the crs defined by crs parameter. 
        alt: ndarray or scalar 
            Altitude above the EARTH_WGS84 ellipsoid.
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
        col : np.ndarray or scalar
            column coordinate in burst referenced to the first valid column.
        row : np.ndarray or scalar
            row coordinate in burst referenced to the first valid line.
        i : np.ndarray or scalar
            Incidence angle.
    
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        alt = np.atleast_1d(alt)
        # # convert to geocentric cartesian
        transformer = pyproj.Transformer.from_crs(crs, 'epsg:4978', always_xy=True)
        X, Y, Z = transformer.transform(x, y, alt)
       
        # project in the slc image
        tinit = (self.burst_times[1] +
                 self.burst_times[2])/2 * np.ones_like(x)
        t, r, i = range_doppler.iterative_projection(self.orbit, X, Y, Z, tinit,
                                                max_iterations, tol)
        if apd_correction:
            alt = alt.squeeze()
            r += (alt * alt / 8.55e7 - alt / 3411.0 + 2.41) / np.cos(i)
    
        # slant range (col coordinate)
        col = (2 * r / const.LIGHT_SPEED_M_PER_SEC -
             self.slant_range_time) * self.range_frequency
    
        # bistatic residual error correction, as described by Schubert et al in
        # Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
        # Results. Remote Sens. 7, 9431-9449 (2015)
        if bistatic_correction:
            t -= (col - 0.5 * self.samples_per_burst
                  ) / (2 * self.range_frequency)
    
        col = col - self.burst_roi[0]
        row = self.to_row(t)
        return col, row, i


    def localization(self, col, row, alt, apd_correction=True,
                           bistatic_correction=True, max_iterations=10000, tol=0.01):
        """
    
    
        Parameters
        ----------
        burst_metadata : dict
            Metadata of burst.
        col : np.ndarray or scalar
            column coordinate in burst referenced to the first valid column.
        row : np.ndarray or scalar
            row coordinate in burst referenced to the first valid line.
        alt : np.ndarray or scalar
            Altitude above the EARTH_WGS84 ellipsoid.
        orbit: eos.sar.range_doppler.Orbit
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
        # make sure we work with numpy arrays
        col = np.atleast_1d(col)
        row = np.atleast_1d(row)
        alt = np.atleast_1d(alt)
        
        num_pts = len(col)
        
        # image coordinates to range and az time
        r = self.to_rng(col)
        ta = self.to_ta(row)
        
        if bistatic_correction:
            # correct azimuth time
            col_mid = col + self.burst_roi[0] - \
                0.5*self.samples_per_burst
            ta += col_mid/(2*self.range_frequency)
        
        if apd_correction:
            # evaluate satellite position
            positions = self.orbit.evaluate(ta)
            # Rough estimation of geometry
            Lsat = np.linalg.norm(positions, axis=1)
            
            # Earth radius taken at the intersection of the line joining satellite
            # and earth center with the ellipsoid
            ell_axis = const.EARTH_WGS84_AXIS_A_M * np.ones(3)
            ell_axis[2] = const.EARTH_WGS84_AXIS_B_M
            ERadius = Lsat/np.sqrt(np.sum((positions/ell_axis)**2, axis=1))
            
            # cosine rule
            incidence = np.arccos(
                (Lsat**2 - (ERadius+alt)**2 - r**2) / (2 * (ERadius + alt) * r))
            
            # correct range
            r -= (alt**2/8.55e7 - alt/3411.0 + 2.41)/np.cos(incidence)
        
        # initial geocentric point xyz definition
        # from lon, lat, alt to x, y, z
        toXYZ = pyproj.Transformer.from_crs('epsg:4326', 'epsg:4978', always_xy=True)
        # point at swath centroid, 0 altitude as init
        lonc, latc = np.mean(self.lon_lat_bbox, axis=0)
        XYZ = np.array(toXYZ.transform(lonc, latc, 0))
        XYZ = np.repeat(XYZ.reshape(1, 3), repeats=num_pts, axis=0)
        
        # localize each point
        points3D = range_doppler.solve_range_doppler(
            self.orbit, ta, r, alt, XYZ, max_iterations, tol)
        
        tolonlat = pyproj.Transformer.from_crs(
            'epsg:4978', 'epsg:4326', always_xy=True)
        lon, lat, _ = tolonlat.transform(*points3D.T)
        
        return lon, lat