"""Sentinel1 models for projection/localization."""
import numpy as np
import pyproj
from eos.sar import model, range_doppler, const, coordinates, orbit


def burst_model_from_burst_meta(burst_meta, **kwargs):
    """Create a Sentinel1BurstModel from a burst meta dict.

    Parameters
    ----------
    burst_meta : dict
        Dict containing all metadata of the burst and sentinel1 product needed
        for processing
    **kwargs : keyword arguments for the constructor of Sentinel1BurstModel.

    Returns
    -------
    Sentinel1BurstModel instance.

    """
    return Sentinel1BurstModel(burst_meta['range_frequency'],
                               burst_meta['azimuth_frequency'],
                               burst_meta['slant_range_time'],
                               burst_meta['samples_per_burst'],
                               burst_meta['burst_times'],
                               burst_meta['burst_roi'],
                               burst_meta['approx_geom'],
                               burst_meta['state_vectors'],
                               **kwargs)

class Sentinel1BaseModel(coordinates.CoordinateMixin, model.SensorModel):
    """Enables operations like projection and localization."""

    def __init__(self,
                 first_row_time, 
                 first_col_time, 
                 approx_geom,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 state_vectors,
                 degree=11,
                 bistatic_correction=True,
                 apd_correction=True,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1BaseModel used to perform projection and localization\
        in a Sentinel1 image. 

        Parameters
        ----------
        first_row_time: float
            Azimuth time of the first line in the image
        first_col_time: float
            Two way slant range time of the first column in the image
        approx_geom: list of tuples (lon, lat)
            Approximate geometry of the image (represented by 4 corners)
        range_frequency : float
            Two way range time sampling frequency .
        azimuth_frequency : float
            Azimuth time sampling frequency.
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
        samples_per_burst : int
            Number of columns per burst in the sentinel1 raster.
        state_vectors : Iterable of dict
            List of state vectors (time, position, velocity).
        degree : int, optional
            Degree of the orbit polynomial. The default is 11.
        bistatic_correction : Boolean, optional
            Apply bistatic correction on the azimuth time. The default is True.
        apd_correction : Boolean, optional
            Apply atmospheric correction on the range. The default is True.
        max_iterations : int, optional
            Maximum iterations of the iterative projection and localization
            algorithms. The default is 20.
        tolerance : float, optional
            Tolerance on the geocentric position used as a stopping criterion.
            For localization, tolerance is taken on 3D point position,
            iterations stop when the step in x, y, z is less than tolerance.
            For projection, the tolerance is considered on the satellite
            position of closest approach. Converted to azimuth time tolerance
            using the speed. The default is 0.001.

        Returns
        -------
        None.

        """
        self.range_frequency = range_frequency
        self.azimuth_frequency = azimuth_frequency
        self.slant_range_time = slant_range_time
        self.samples_per_burst = samples_per_burst
        self.orbit = orbit.Orbit(state_vectors, degree)
        # processing params
        self.bistatic_correction = bistatic_correction
        self.apd_correction = apd_correction
        # stopping criteria
        self.max_iterations = max_iterations
        # setting the tolerance
        self.localization_tolerance = tolerance
        self.projection_tolerance = tolerance \
            / np.linalg.norm(state_vectors[0]['velocity'])


        # set these for the CoordinateMixin
        self.first_row_time = first_row_time 
        self.first_col_time = first_col_time
        
        # set some params necessary for processing
        self.azt_init = first_row_time
        self.approx_geom = approx_geom
        
    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None):
        """Projects a 3D point into the image coordinates.

        Parameters
        ----------
        x, y : ndarray or scalar
            Coordinates in the crs defined by crs parameter.
        alt: ndarray or scalar
            Altitude defined by vert_crs if provided or EARTH_WGS84 ellipsoid.
        crs : string, optional
            CRS in which the point is given
                    Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
        vert_crs: string, optional
            Vertical crs

        Returns
        -------
        rows : ndarray or scalar
            Row coordinate in image referenced to the first line.
        cols : ndarray or scalar
            Column coordinate in image referenced to the first column.
        i : ndarray or scalar
            Incidence angle.

        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        alt = np.atleast_1d(alt)

        if vert_crs is None:
            src_crs = crs
        else:
            src_crs = pyproj.crs.CompoundCRS(
                name='ukn_reference', components=[crs, vert_crs])

        transformer = pyproj.Transformer.from_crs(
            src_crs, 'epsg:4978', always_xy=True)

        # convert to geocentric cartesian
        gx, gy, gz = transformer.transform(x, y, alt)

        azt_init = self.azt_init * np.ones_like(x)
        
        azt, rng, i = range_doppler.iterative_projection(
                                    self.orbit, gx, gy,
                                    gz, azt_init=azt_init,
                                    max_iterations=self.max_iterations,
                                    tol=self.projection_tolerance)
        # Apply corrections on rng and azt if needed
        if self.apd_correction:
            alt = alt.squeeze()
            rng += (alt * alt / 8.55e7 - alt / 3411.0 + 2.41) / np.cos(i)

        # bistatic residual error correction, as described by Schubert et al in
        # Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
        # Results. Remote Sens. 7, 9431-9449 (2015)
        if self.bistatic_correction:
            # slant range (col coordinate)
            azt -=  0.5 * (2 * rng / const.LIGHT_SPEED_M_PER_SEC -
                    self.slant_range_time -
                    0.5*self.samples_per_burst/self.range_frequency)

        # convert to row and col
        row, col = self.to_row_col(azt, rng)

        return row, col, i

    def localization(self, row, col, alt, crs='epsg:4326', vert_crs=None):
        """Localize a point in the image at a certain altitude.

        Parameters
        ----------
        row : ndarray or scalar
            row coordinate in image referenced to the first line.
        col : ndarray or scalar
            column coordinate in image referenced to the first column.
        alt : ndarray or scalar
            Altitude above the EARTH_WGS84 ellipsoid.
        crs : string, optional
            CRS in which the point is returned
                    Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
        vert_crs: string, optional
            Vertical crs in which the point is returned

        Returns
        -------
        x, y, z : ndarray or scalar
            Coordinates of the point in the crs
        """
        # make sure we work with numpy arrays
        row = np.atleast_1d(row)
        col = np.atleast_1d(col)
        alt = np.atleast_1d(alt)

        # image coordinates to range and az time
        azt, rng = self.to_azt_rng(row, col)

        # Make corrections on azt and rng if needed
        if self.bistatic_correction:
            # correct azimuth time
            azt += 0.5 * (2 * rng / const.LIGHT_SPEED_M_PER_SEC -
                    self.slant_range_time -
                    0.5*self.samples_per_burst/self.range_frequency)

        if self.apd_correction:

            # evaluate satellite position
            positions = self.orbit.evaluate(azt)
            # Rough estimation of geometry
            os = np.linalg.norm(positions, axis=1)

            # Earth radius taken at the intersection of the line joining
            # satellite and earth center with the ellipsoid
            ell_axis = np.array([const.EARTH_WGS84_AXIS_A_M,
                                 const.EARTH_WGS84_AXIS_A_M,
                                 const.EARTH_WGS84_AXIS_B_M])

            earth_radius = os/np.sqrt(np.sum((positions/ell_axis)**2, axis=1))
            op = earth_radius + alt

            # cosine rule
            cos_incidence = (os**2 - op**2 - rng**2) / (2 * op * rng)

            # correct range
            rng -= (alt**2/8.55e7 - alt/3411.0 + 2.41)/cos_incidence

        # initial geocentric point xyz definition
        # from lon, lat, alt to x, y, z
        to_gxyz = pyproj.Transformer.from_crs(
            'epsg:4326', 'epsg:4978', always_xy=True)

        # point at swath centroid, 0 altitude as init
        lon_c, lat_c = np.mean(self.approx_geom, axis=0)

        gx_init, gy_init, gz_init = to_gxyz.transform(
                                        lon_c * np.ones_like(alt),
                                        lat_c * np.ones_like(alt),
                                        alt)

        # localize each point
        gx, gy, gz = range_doppler.iterative_localization(
            self.orbit, azt, rng, alt, (gx_init, gy_init, gz_init),
            max_iterations=self.max_iterations,
            tol=self.localization_tolerance)

        if vert_crs is None:
            dst_crs = crs
        else:
            dst_crs = pyproj.crs.CompoundCRS(
                name='ukn_reference', components=[crs, vert_crs])
        todst = pyproj.Transformer.from_crs(
            'epsg:4978', dst_crs, always_xy=True)
        x, y, z = todst.transform(gx, gy, gz)

        return x, y, z

class Sentinel1BurstModel(Sentinel1BaseModel):
    """Enables operations like projection and localization at the burst."""

    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 burst_times,
                 burst_roi,
                 approx_geom,
                 state_vectors,
                 degree=11,
                 bistatic_correction=True,
                 apd_correction=True,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1BurstModel used to perform projection and localization\
        in a Sentinel1 burst.

        Parameters
        ----------
        range_frequency : float
            Two way range time sampling frequency .
        azimuth_frequency : float
            Azimuth time sampling frequency.
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
        samples_per_burst : int
            Number of columns per burst in the sentinel1 raster.
        burst_times : (3,) ndarray/tuple (start_time, start_valid, end_valid)
            start_time is the azimuth time of the first line in the burst
            start/end_valid denote the azimuth time of the
            first/last valid line in the burst.
        burst_roi : (4,) ndarray/tuple (x, y, w, h)
            Coordinates of the burst in the sentinel-1 raster file.
        approx_geom : List of tuples
            Each tuple element is a (lon, lat) corner of the approx geom
            of the burst
        state_vectors : Iterable of dict
            List of state vectors (time, position, velocity).
        degree : int, optional
            Degree of the orbit polynomial. The default is 11.
        bistatic_correction : Boolean, optional
            Apply bistatic correction on the azimuth time. The default is True.
        apd_correction : Boolean, optional
            Apply atmospheric correction on the range. The default is True.
        max_iterations : int, optional
            Maximum iterations of the iterative projection and localization
            algorithms. The default is 20.
        tolerance : float, optional
            Tolerance on the geocentric position used as a stopping criterion.
            For localization, tolerance is taken on 3D point position,
            iterations stop when the step in x, y, z is less than tolerance.
            For projection, the tolerance is considered on the satellite
            position of closest approach. Converted to azimuth time tolerance
            using the speed. The default is 0.001.

        Returns
        -------
        None.

        """

        # set these for the CoordinateMixin
        first_row_time = burst_times[1]  # start valid
        first_col_time = slant_range_time + burst_roi[0] / range_frequency
        approx_geom = approx_geom
        super().__init__(first_row_time, 
         first_col_time, 
         approx_geom,
         range_frequency,
         azimuth_frequency,
         slant_range_time,
         samples_per_burst,
         state_vectors,
         degree,
         bistatic_correction,
         apd_correction,
         max_iterations,
         tolerance)
        
        # specific to current burst
        self.burst_times = burst_times
        self.burst_roi = burst_roi
        
        # reset the initial azimuth guess at the center of burst
        self.azt_init = (self.burst_times[1] + self.burst_times[2])/2

class Sentinel1SwathModel(Sentinel1BaseModel):
    """Enables operations like projection and localization at a swath."""

    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 bursts_times,
                 bursts_rois,
                 bursts_approx_geom,
                 state_vectors,
                 degree=11,
                 bistatic_correction=True,
                 apd_correction=True,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1SwathModel used to perform projection and localization\
        in a Sentinel1 swath.

        Parameters
        ----------
        range_frequency : float
            Two way range time sampling frequency .
        azimuth_frequency : float
            Azimuth time sampling frequency.
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
        samples_per_burst : int
            Number of columns per burst in the sentinel1 raster.
        bursts_times : list of (3,) tuple (start_time, start_valid, end_valid)
            start_time is the azimuth time of the first line in the burst
            start/end_valid denote the azimuth time of the
            first/last valid line in the burst.
        bursts_rois : list of (4,) tuple (x, y, w, h)
            Coordinates of the burst in the sentinel-1 raster file.
        bursts_approx_geom : List of list of tuples
            Each element is a list of tuples (lon, lat) corners of the approx geom
            of the burst
        state_vectors : Iterable of dict
            List of state vectors (time, position, velocity).
        degree : int, optional
            Degree of the orbit polynomial. The default is 11.
        bistatic_correction : Boolean, optional
            Apply bistatic correction on the azimuth time. The default is True.
        apd_correction : Boolean, optional
            Apply atmospheric correction on the range. The default is True.
        max_iterations : int, optional
            Maximum iterations of the iterative projection and localization
            algorithms. The default is 20.
        tolerance : float, optional
            Tolerance on the geocentric position used as a stopping criterion.
            For localization, tolerance is taken on 3D point position,
            iterations stop when the step in x, y, z is less than tolerance.
            For projection, the tolerance is considered on the satellite
            position of closest approach. Converted to azimuth time tolerance
            using the speed. The default is 0.001.

        Returns
        -------
        None.

        """
        # set these for the CoordinateMixin
        first_row_time = bursts_times[0][1]  # start valid
       
        self.x_min = min(roi[0] for roi in bursts_rois) 
        first_col_time = slant_range_time + self.x_min / range_frequency
        # swath polygon
        approx_geom = bursts_approx_geom[0][:2] + bursts_approx_geom[-1][2:]
        
        # call the base class constructor
        super().__init__(first_row_time, 
                 first_col_time, 
                 approx_geom,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 state_vectors,
                 degree,
                 bistatic_correction,
                 apd_correction,
                 max_iterations,
                 tolerance)
        
        # setting image size 
        self.y_min = bursts_rois[0][1]
        x_max = max(roi[0] + roi[2] - 1 for roi in bursts_rois)
        self.w = (x_max - self.x_min + 1) 
        self.h = (bursts_times[-1][2] - first_row_time) \
                * self.azimuth_frequency
                
        # additional burst params, will surely be needed for coord conversion
        self.bursts_times = bursts_times 
        self.bursts_rois = bursts_rois
        
        # reset the initial azimuth guess at the center of swath
        self.azt_init = (self.bursts_times[0][1] + self.bursts_times[-1][2])/2
        

def swath_model_from_bursts_meta(bursts_metadata, **kwargs):
    """
    Generate Sentinel1SwathModel instance from list of bursts metadata.

    Parameters
    ----------
    bursts_metadata : list of dicts 
        each dict contains attribute metadata relative to the bursts of the 
        swath.
    **kwargs : keyword arguments
        one of degree, bistatic_correction, apd_correction, max_iterations,
        tolerance. Processing parameters for the projection and localization. 

    Returns
    -------
    Sentinel1SwathModel
        Model for projection and localization inside the swath.

    """
    bursts_times = [b['burst_times'] for b in bursts_metadata] 
    bursts_rois = [b['burst_roi'] for b in bursts_metadata]
    bursts_approx_geom = [b['approx_geom'] for b in bursts_metadata]  
    return Sentinel1SwathModel(bursts_metadata[0]['range_frequency'], 
                                bursts_metadata[0]['azimuth_frequency'], 
                                bursts_metadata[0]['slant_range_time'], 
                                bursts_metadata[0]['samples_per_burst'], 
                                bursts_times, 
                                bursts_rois, 
                                bursts_approx_geom, 
                                bursts_metadata[0]['state_vectors'], 
                                **kwargs)