"""Sentinel1 models for projection/localization."""
import numpy as np
import pyproj
from eos.sar import model, range_doppler, const, coordinates, orbit, roi, utils
from . import doppler_info


def burst_model_from_burst_meta(burst_meta, doppler=None, doppler_kwargs={}, **kwargs):
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
    if not doppler:
        doppler = doppler_info.doppler_from_meta(burst_meta, **doppler_kwargs)
    return Sentinel1BurstModel(doppler,
                               burst_meta['range_frequency'],
                               burst_meta['azimuth_frequency'],
                               burst_meta['slant_range_time'],
                               burst_meta['samples_per_burst'],
                               burst_meta['wave_length'],
                               burst_meta['burst_times'],
                               burst_meta['burst_roi'],
                               burst_meta['approx_geom'],
                               burst_meta['state_vectors'],
                               chirp_rate=burst_meta.get('chirp_rate'),
                               pri=burst_meta.get('pri'),
                               rank=burst_meta.get('rank'),
                               **kwargs)


class Sentinel1BaseModel(coordinates.CoordinateMixin, model.SensorModel):
    """Enables operations like projection and localization."""

    def __init__(self,
                 first_row_time,
                 first_col_time,
                 approx_geom,
                 range_frequency,
                 azimuth_frequency,
                 width,
                 height,
                 wavelength, 
                 slant_range_time,
                 samples_per_burst,
                 state_vectors,
                 degree=11,
                 pri=None,
                 rank=None,
                 bistatic_correction=True,
                 full_bistatic_correction_reference=None,
                 apd_correction=True,
                 intra_pulse_correction=False,
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
        width: int
            width of the image
        height: int
            height of the image
        wavelength: float
            wavelength in m 
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
        samples_per_burst : int
            Number of columns per burst in the sentinel1 raster.
        state_vectors : Iterable of dict
            List of state vectors (time, position, velocity).
        degree : int, optional
            Degree of the orbit polynomial. The default is 11.
        pri:
        rank:
        bistatic_correction : Boolean, optional
            Apply bistatic correction on the azimuth time. The default is True.
        full_bistatic_correction_reference: Dict, optional
            Metadata of one of the bursts of IW2.
        apd_correction : Boolean, optional
            Apply atmospheric correction on the range. The default is True.
        intra_pulse_correction:
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

        self.w = width 
        self.h = height
        self.wavelength = wavelength  # for TopoCorrection

        self.slant_range_time = slant_range_time
        self.samples_per_burst = samples_per_burst
        self.orbit = orbit.Orbit(state_vectors, degree)
        # processing params
        self.pri = pri
        self.rank = rank
        self.bistatic_correction = bistatic_correction
        self.full_bistatic_correction_reference = full_bistatic_correction_reference
        self.apd_correction = apd_correction
        self.intra_pulse_correction = intra_pulse_correction
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
            
    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None, azt_init=None):
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
        azt_init: ndarray or scalar, optional
            Initial azimuth time guess of the points. If not given, the first
            row time will be used. The default is None. 

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
        
        if azt_init is not None:
            err_msg = "Init azimuth time should be scalar or have the\
                 same length of the points"
            azt_init = utils.check_input_len(azt_init, len(x), err_msg)
        else: 
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

        if self.intra_pulse_correction:
            rng -= self._intra_pulse(azt, rng)

        if self.bistatic_correction:
            ref = self.full_bistatic_correction_reference
            if ref is None:
                # bistatic residual error correction, as described by Schubert et al in
                # Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
                # Results. Remote Sens. 7, 9431-9449 (2015)
                # slant range (col coordinate)
                azt -= 0.5 * (2 * rng / const.LIGHT_SPEED_M_PER_SEC -
                              self.slant_range_time -
                              0.5*self.samples_per_burst/self.range_frequency)
            else:
                assert self.pri is not None
                assert self.rank is not None
                # full bistatic error correction, as described by Gisinger et al., in
                # "Recent Findings on the Sentinel-1 Geolocation Accuracy Using the
                # Australian Corner Reflector Array." IGARSS 2018-2018 IEEE International
                # Geoscience and Remote Sensing Symposium. IEEE, 2018.

                # this correction requires the IW2 values of slant_range_time,
                # samples_per_burst and range_frequency from the ref metadata

                azt -= ((ref['slant_range_time'] + 0.5 * ref['samples_per_burst'] / ref['range_frequency']) / 2
                        - self.rank * self.pri + (2 * rng / const.LIGHT_SPEED_M_PER_SEC) / 2)

        # convert to row and col
        row, col = self.to_row_col(azt, rng)

        return row, col, i

    def localization(self, row, col, alt, crs='epsg:4326', vert_crs=None,
                     x_init=None, y_init=None, z_init=None):
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
        x_init: ndarray or scalar, optional
            Initial guess of the x component. The default is None. 
        y_init: ndarray or scalar, optional
            Initial guess of the y component. The default is None.
        z_init: ndarray or scalar, optional
            Initial guess of the z component. The default is None.

        Returns
        -------
        x, y, z : ndarray or scalar
            Coordinates of the point in the crs
        
        Notes
        -----
        If no initial guess for the 3D point is given, the initial point for 
        the iterative localization is taken at the centroid of the approx 
        geometry of the model, with altitudes given by the alt array. 
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

            # TODO: full_bistatic_correction_reference

        # TODO: intra pulse correction

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

        
        if vert_crs is None:
            dst_crs = crs
        else:
            dst_crs = pyproj.crs.CompoundCRS(
                name='ukn_reference', components=[crs, vert_crs])
            
        if (x_init is not None) and (y_init is not None) and (z_init is not None): 
            to_gxyz = pyproj.Transformer.from_crs(
                dst_crs, 'epsg:4978', always_xy=True)
            out_len = len(alt)
            err_msg = "{} length should be the same as row/col/alt len"
            x_init = utils.check_input_len(x_init, out_len, err_msg.format("x_init"))
            y_init = utils.check_input_len(y_init, out_len, err_msg.format("y_init"))
            z_init = utils.check_input_len(z_init, out_len, err_msg.format("z_init"))
        else: 
            # initial geocentric point xyz definition
            # from lon, lat, alt to x, y, z
            to_gxyz = pyproj.Transformer.from_crs(
                'epsg:4326', 'epsg:4978', always_xy=True)

            # point at swath centroid, 0 altitude as init
            lon_c, lat_c = np.mean(self.approx_geom, axis=0)
            
            x_init = lon_c * np.ones_like(alt)
            y_init = lat_c * np.ones_like(alt)
            z_init = alt
        
        gx_init, gy_init, gz_init = to_gxyz.transform(
            x_init,
            y_init,
            z_init)

        # localize each point
        gx, gy, gz = range_doppler.iterative_localization(
            self.orbit, azt, rng, alt, (gx_init, gy_init, gz_init),
            max_iterations=self.max_iterations,
            tol=self.localization_tolerance)


        todst = pyproj.Transformer.from_crs(
            'epsg:4978', dst_crs, always_xy=True)
        x, y, z = todst.transform(gx, gy, gz)

        return x, y, z

class Sentinel1BurstModel(Sentinel1BaseModel):
    """Enables operations like projection and localization at the burst."""

    def __init__(self,
                 doppler: doppler_info.Sentinel1Doppler,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 wavelength, 
                 burst_times,
                 burst_roi,
                 approx_geom,
                 state_vectors,
                 degree=11,
                 chirp_rate=None,
                 pri=None,
                 rank=None,
                 bistatic_correction=True,
                 full_bistatic_correction_reference=None,
                 apd_correction=True,
                 intra_pulse_correction=False,
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
        wavelength: float
            wavelength in m 
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
        chirp_rate:
        pri:
        rank:
        bistatic_correction : Boolean, optional
            Apply bistatic correction on the azimuth time. The default is True.
        full_bistatic_correction_reference:
        apd_correction : Boolean, optional
            Apply atmospheric correction on the range. The default is True.
        intra_pulse_correction:
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
        super().__init__(first_row_time,
                         first_col_time,
                         approx_geom,
                         range_frequency,
                         azimuth_frequency,
                         burst_roi[2], 
                         burst_roi[3], 
                         wavelength, 
                         slant_range_time,
                         samples_per_burst,
                         state_vectors,
                         degree,
                         pri,
                         rank,
                         bistatic_correction,
                         full_bistatic_correction_reference,
                         apd_correction,
                         intra_pulse_correction,
                         max_iterations,
                         tolerance)

        # specific to current burst
        self.burst_times = burst_times
        self.burst_roi = roi.Roi.from_roi_tuple(burst_roi)

        # for _intra_pulse
        self.chirp_rate = chirp_rate

        # reset the initial azimuth guess at the center of burst
        self.azt_init = (self.burst_times[1] + self.burst_times[2])/2
        self.doppler = doppler

    def _intra_pulse(self, t, r):
        """
        Compute intra-pulse motion range correction (azimuth dependent range shift
        depending on the Doppler frequency under which the target has been observed).

        The correction is described in Piantanida, R., et al. "Accurate Geometric
        Calibration of Sentinel-1 Data." EUSAR 2018; 12th European Conference on
        Synthetic Aperture Radar. VDE, 2018. and Scheiber, R, et al. "Speckle
        tracking and interferometric processing of TerraSAR-X TOPS data for mapping
        nonstationary scenarios." IEEE Journal of Selected Topics in Applied Earth
        Observations and Remote Sensing 8.4 (2015): 1709-1720.

        Args:
            t (array): azimuth time
            r (array): range distance to sensor

        Retruns:
            dr (float or array): intra-pulse motion range correction
        """
        if not isinstance(t, (list, np.ndarray)):
            d = self._intra_pulse(np.asarray([t]), np.asarray([r]))
            assert len(d) == 1
            return d[0]

        LIGHT_SPEED = const.LIGHT_SPEED_M_PER_SEC

        # Doppler FM rate
        range_dependent_doppler_rate = self.doppler.get_rg_dpt_dop_rate(2 * r / LIGHT_SPEED)
        assert len(range_dependent_doppler_rate) == len(r)

        # Doppler Centroid Rate
        doppler_rate = self.doppler.get_dop_rate(range_dependent_doppler_rate)

        # Doppler Centroid Frequency
        f_geom = self.doppler.get_dop_centroid(2 * r / LIGHT_SPEED)

        # azimuth dependent Doppler Centroid Frequency
        mid_time = self.doppler.get_burst_mid_time()
        ref_time = self.doppler.get_ref_time(f_geom, range_dependent_doppler_rate)
        f = doppler_rate * (t - mid_time - ref_time)

        dr = (f + f_geom) / self.chirp_rate * LIGHT_SPEED / 2
        return dr


class Sentinel1SwathModel(Sentinel1BaseModel):
    """Enables operations like projection and localization at a swath."""

    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 lines_per_burst,
                 wavelength,
                 doppler: doppler_info.Sentinel1Doppler,
                 bursts_times,
                 bursts_rois,
                 bursts_approx_geom,
                 state_vectors,
                 degree=11,
                 pri=None,
                 rank=None,
                 bistatic_correction=True,
                 full_bistatic_correction_reference=None,
                 apd_correction=True,
                 intra_pulse_correction=False,
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
        lines_per_burst : int
        wavelength: float
            wavelength in m 
        doppler
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
        pri:
        rank:
        bistatic_correction : Boolean, optional
            Apply bistatic correction on the azimuth time. The default is True.
        full_bistatic_correction_reference:
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

        self.col_min = min(roi_[0] for roi_ in bursts_rois)
        first_col_time = slant_range_time + self.col_min / range_frequency
        # swath polygon
        approx_geom = bursts_approx_geom[0][:2] + bursts_approx_geom[-1][2:]

        # setting image size
        self.row_min = bursts_rois[0][1
                                      ]
        col_max = max(roi_[0] + roi_[2] - 1 for roi_ in bursts_rois)
        w = (col_max - self.col_min + 1)
        h = int(np.round((bursts_times[-1][2] - first_row_time)
                              * azimuth_frequency))
        # call the base class constructor
        super().__init__(first_row_time,
                         first_col_time,
                         approx_geom,
                         range_frequency,
                         azimuth_frequency,
                         w, 
                         h, 
                         wavelength, 
                         slant_range_time,
                         samples_per_burst,
                         state_vectors,
                         degree,
                         pri,
                         rank,
                         bistatic_correction,
                         full_bistatic_correction_reference,
                         apd_correction,
                         False,  # intra_pulse_correction
                         max_iterations,
                         tolerance)

        self.doppler = doppler
        self.lines_per_burst = lines_per_burst

        # additional burst params, will surely be needed for coord conversion
        self.bursts_times = bursts_times
        self.bursts_rois = [roi.Roi.from_roi_tuple(_roi) for _roi in bursts_rois]
        # reset the initial azimuth guess at the center of swath
        self.azt_init = (self.bursts_times[0][1] + self.bursts_times[-1][2])/2

    def burst_orig_in_swath(self, burst_id):
        """
        Computes the coordinates of the origin of the burst in the swath frame.

        Parameters
        ----------
        burst_id : int
            Id of the burst.

        Returns
        -------
        orig : tuple 
            (col, row) coordinates of the burst origin in the swath.

        """
        col = self.bursts_rois[burst_id].col - self.col_min
        azt = self.bursts_times[burst_id][1]
        row, _ = self.to_row_col(azt, 0)
        orig = (col,  int(np.round(row)))
        return orig

    def compute_overlaps(self):
        """
        Compute the number of lines that overlap (cover the same ground feature)
        between the end of a burst and the start of another, based on the azimuth time.
        Do this for all bursts and store results in self.overlaps.      

        Returns
        -------
        None.

        """
        n_bursts = len(self.bursts_times)
        # n_bursts - 1 overlaps
        self.overlaps = np.zeros(n_bursts - 1, dtype=int)
        for i in range(n_bursts - 1):
            # ith overlap between i and i+1 burst
            current_burst_end = self.bursts_times[i][2]
            next_burst_start = self.bursts_times[i + 1][1]
            self.overlaps[i] = int(np.round((current_burst_end -
                                             next_burst_start)*self.azimuth_frequency))

    def overlap_roi(self, overlap_id):
        """
        Computes the overlap region of a given id between two bursts. The overlap
        is given as a roi w.r.t. the burst "origin".

        Parameters
        ----------
        overlap_id : int
            id of the overlap [0, n_bursts -1].

        Returns
        -------
        overlap_prev_roi : eos.sar.roi.Roi
            roi inside the previous burst.
        overlap_next_roi : eos.sar.roi.Roi
            roi inside the next burst.

        """
        if not hasattr(self, 'overlaps'):
            self.compute_overlaps()
        assert overlap_id >= 0 and overlap_id < len(self.overlaps),\
            "overlap id out of bound"
        ovl = self.overlaps[overlap_id]
        # previous burst
        h, w = self.bursts_rois[overlap_id].get_shape()
        overlap_prev_roi = roi.Roi(0, h - ovl, w, ovl)
        # next burst
        h, w = self.bursts_rois[overlap_id + 1].get_shape()
        overlap_next_roi = roi.Roi(0, 0, w, ovl)
        return overlap_prev_roi, overlap_next_roi

    def burst_roi_without_ovl(self, burst_id):
        """
        Compute the burst roi without overlap w.r.t. the burst "origin".

        Parameters
        ----------
        burst_id : int
            id of the burst.

        Returns
        -------
        burst_roi_without_ovl : eos.sar.roi.Roi
            roi of the burst adjusted for debursting. the roi
            is computed w.r.t. the burst origin. 

        """
        if not hasattr(self, 'overlaps'):
            self.compute_overlaps()

        assert burst_id >= 0 and burst_id < len(self.bursts_rois),\
            "burst id out of bound"
        h, w = self.bursts_rois[burst_id].get_shape()
        ovl_prev = self.overlaps[burst_id - 1] if burst_id else 0
        ovl_next = self.overlaps[burst_id] if burst_id < len(
            self.overlaps) else 0
        remove_lines_at_top = ovl_prev//2
        remove_lines_at_bottom = ovl_next - ovl_next//2
        burst_roi_without_ovl = roi.Roi(0, remove_lines_at_top, w,
            h - remove_lines_at_top - remove_lines_at_bottom)
        return burst_roi_without_ovl

    def adjust_roi_to_swath(self, request_roi):
        """
        Adjust an roi to that it does not go beyond the swath's boundaries.

        Parameters
        ----------
        request_roi: eos.sar.roi.Roi
            Roi in swath coordinates. Region defined inside the swath.

        Returns
        -------
        roi_in_swath: eos.sar.roi.Roi
            Region of interest adjusted to the swath's boundaries
        """
        return request_roi.make_valid((self.h, self.w))

    def get_read_write_rois(self, roi_in_swath=None):
        """
        Compute the region to read from each burst if given a roi contained in
        a swath. The writing roi is also returned, with the corresponding burst
        ids.
        The output size might be smaller than the `roi_in_swath` if it goes beyond
        the swath's boundaries.

        Parameters
        ----------
        roi_in_swath : eos.sar.roi.Roi, optional
            roi in swath. Region defined inside the swath.
            If not given, the whole swath is taken. The default is None.

        Returns
        -------
        burst_ids : list of int
            Ids of the bursts intersected by the roi.
        rois_read : list of eos.sar.roi.Roi
            Each roi corresponds to the region to be read from 
            the tiff file.
        rois_write : list of eos.sar.roi.Roi
            Each roi corresponds to the region where the output
            data should be written in the output image.
        out_shape: tuple
            (h, w) Output image shape

        """
        if roi_in_swath is None:
            roi_in_swath = roi.Roi(0, 0, self.w, self.h)

        roi_in_swath = self.adjust_roi_to_swath(roi_in_swath)

        col, row, w, h = roi_in_swath.to_roi()
        out_shape = (h, w)

        col += self.col_min  # x is now relative to the tiff img
        previous_bursts_h = 0  # current y in the input image fully debursted
        previous_roi_h = 0  # current y in the output crop

        burst_ids = []
        rois_read = []
        rois_write = []

        for burst_id in range(len(self.bursts_rois)):
            # burst roi without overlap relative to tiff img
            bcol, brow, bw, bh = self.burst_roi_without_ovl(burst_id).translate_roi(
                                                   self.bursts_rois[burst_id].col,
                                                   self.bursts_rois[burst_id].row).to_roi() 
            # loop until we find first burst intersecting roi
            if previous_bursts_h + bh > row + previous_roi_h:
                col_min = max(col, bcol)
                col_max = min(col + w, bcol + bw)
                debursted_to_tif = brow - previous_bursts_h
                row_min = row + previous_roi_h + debursted_to_tif
                row_max = min(row + h, previous_bursts_h + bh) + \
                    debursted_to_tif
                col_size = col_max - col_min
                row_size = row_max - row_min
                burst_ids.append(burst_id)
                rois_read.append(roi.Roi(col_min, row_min, col_size, row_size))
                rois_write.append(roi.Roi(col_min - col, previous_roi_h,
                                          col_size, row_size))
                previous_roi_h += row_size
            previous_bursts_h += bh
            if previous_bursts_h >= row + h:
                break

        return burst_ids, rois_read, rois_write, out_shape


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

    def alleq(prop):
        burst = bursts_metadata[0]
        return all(b[prop] == burst[prop] for b in bursts_metadata)
    assert alleq('range_frequency')
    assert alleq('azimuth_frequency')
    assert alleq('slant_range_time')
    assert alleq('lines_per_burst')
    assert alleq('wave_length')
    assert alleq('pri')
    assert alleq('rank')

    doppler = doppler_info.doppler_from_meta(bursts_metadata[0])
    return Sentinel1SwathModel(bursts_metadata[0]['range_frequency'],
                               bursts_metadata[0]['azimuth_frequency'],
                               bursts_metadata[0]['slant_range_time'],
                               bursts_metadata[0]['samples_per_burst'],
                               bursts_metadata[0]['lines_per_burst'],
                               bursts_metadata[0]['wave_length'],
                               doppler,
                               bursts_times,
                               bursts_rois,
                               bursts_approx_geom,
                               bursts_metadata[0]['state_vectors'],
                               pri=bursts_metadata[0].get('pri'),
                               rank=bursts_metadata[0].get('rank'),
                               **kwargs)
