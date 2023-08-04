"""Sentinel1 models for projection/localization."""
from __future__ import annotations

import numpy as np
import pyproj
from eos.products.sentinel1.metadata import Sentinel1BurstMetadata, Sentinel1GRDMetadata
from eos.sar import model, range_doppler, coordinates, roi, utils
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector, GeoImagePoints
from eos.products import sentinel1


def grd_model_from_meta(meta: Sentinel1GRDMetadata,
                        orbit: Orbit,
                        coord_corrector: Corrector = Corrector()):
    """ Create a Sentinel1GRDModel from a GRD meta dict.

    Parameters
    ----------
    meta : Sentinel1GRDMetadata
    orbit: Orbit
    coord_corrector: eos.sar.projection_correction.Corrector
        Corrector object containing a list of ImageCorrection in this case

    Returns
    -------
    Sentinel1GRDModel instance.

    """
    srgr = sentinel1.srgr.Sentinel1SRGRConverter(meta.srgr)
    proj_model = Sentinel1GRDModel(
        meta.image_start,
        meta.approx_geom,
        meta.range_pixel_spacing,
        meta.azimuth_time_interval,
        meta.width,
        meta.height,
        meta.wave_length,
        orbit,
        srgr,
        coord_corrector,
    )
    return proj_model


def burst_model_from_burst_meta(burst_meta: Sentinel1BurstMetadata,
                                orbit: Orbit,
                                coord_corrector=Corrector(),
                                **kwargs) -> Sentinel1BurstModel:
    """Create a Sentinel1BurstModel from a burst meta dict.

    Parameters
    ----------
    burst_meta : Sentinel1BurstMetadata
    orbit: Orbit
    coord_corrector: eos.sar.projection_correction.Corrector
        Corrector object containing a list of ImageCorrection in this case
    **kwargs : keyword arguments for the constructor of Sentinel1BurstModel.

    Returns
    -------
    Sentinel1BurstModel instance.

    """
    return Sentinel1BurstModel(burst_meta.range_frequency,
                               burst_meta.azimuth_frequency,
                               burst_meta.slant_range_time,
                               burst_meta.wave_length,
                               burst_meta.burst_times,
                               burst_meta.burst_roi,
                               burst_meta.approx_geom,
                               orbit,
                               coord_corrector=coord_corrector,
                               **kwargs)


class Sentinel1BaseModel(model.SensorModel):
    """Enables operations like projection and localization."""

    def __init__(self,
                 first_row_time,
                 approx_geom,
                 width,
                 height,
                 wavelength,
                 orbit: Orbit,
                 coord_corrector: Corrector,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1BaseModel used to perform projection and localization\
        in a Sentinel1 image.

        Parameters
        ----------
        first_row_time: float
            Azimuth time of the first line in the image
        approx_geom: list of tuples (lon, lat)
            Approximate geometry of the image (represented by 4 corners)
        width: int
            width of the image
        height: int
            height of the image
        wavelength: float
            wavelength in m
        orbit: Orbit
            Orbit instance
        coord_corrector: eos.sar.projection_correction.Corrector
            Corrector object containing a list of ImageCorrection in this case
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
        self.w = width
        self.h = height
        self.wavelength = wavelength  # for TopoCorrection

        self.orbit = orbit
        # stopping criteria
        self.max_iterations = max_iterations
        # setting the tolerance
        self.localization_tolerance = tolerance
        self.projection_tolerance = tolerance \
            / np.linalg.norm(orbit.sv[0].velocity)

        # set for the CoordinateMixin
        self.first_row_time = first_row_time

        # set some params necessary for processing
        self.azt_init = first_row_time
        self.approx_geom = approx_geom

        self.coord_corrector = coord_corrector

    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None, azt_init=None, as_azt_rng=False):
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
        as_azt_rng: bool, optional
            Returns azimuth/range instead of rows/cols. The incidence angle is unchanged.
            Defaults to False.

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

        if not self.coord_corrector.empty():
            # create a geo_im_pt
            geo_im_pt = GeoImagePoints(gx, gy, gz, azt, rng)

            # apply corrections
            geo_im_pt = self.coord_corrector.estimate_and_apply(geo_im_pt)

            azt, rng = geo_im_pt.get_azt_rng(squeeze=True)

        if as_azt_rng:
            return azt, rng, i

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

        # First localization, no correction is enabled
        # localize each point
        gx, gy, gz = range_doppler.iterative_localization(
            self.orbit, azt, rng, alt, (gx_init, gy_init, gz_init),
            max_iterations=self.max_iterations,
            tol=self.localization_tolerance)

        if not self.coord_corrector.empty():
            # create a geo_im_pt
            geo_im_pt = GeoImagePoints(gx, gy, gz, azt, rng)

            # apply corrections
            geo_im_pt = self.coord_corrector.estimate_and_apply(geo_im_pt, inverse=True)

            azt, rng = geo_im_pt.get_azt_rng()

            # Perform localization again with corrected coords
            # Should converge quickly (probably one iteration)
            gx, gy, gz = range_doppler.iterative_localization(
                self.orbit, azt, rng, alt, (gx, gy, gz),
                max_iterations=self.max_iterations,
                tol=self.localization_tolerance)

        todst = pyproj.Transformer.from_crs(
            'epsg:4978', dst_crs, always_xy=True)
        x, y, z = todst.transform(gx, gy, gz)

        return x, y, z


class Sentinel1SLCBaseModel(coordinates.SLCCoordinateMixin, Sentinel1BaseModel):
    def __init__(self,
                 first_row_time,
                 first_col_time,
                 approx_geom,
                 range_frequency,
                 azimuth_frequency,
                 width,
                 height,
                 wavelength,
                 orbit,
                 coord_corrector,
                 max_iterations=20,
                 tolerance=0.001):
        """
        Parameters
        ----------
        first_row_time: float
            Azimuth time of the first line in the image
        first_col_time: float
            Two way slant range time of the first column in the image
        approx_geom: list of tuples (lon, lat)
            Approximate geometry of the image (represented by 4 corners)
        range_frequency : float
            Two way range time sampling frequenc
        azimuth_frequency : float
            Azimuth time sampling frequency
        width: int
            width of the image
        height: int
            height of the image
        wavelength: float
            wavelength in m
        orbit: Orbit
            Orbit instance
        coord_corrector: eos.sar.projection_correction.Corrector
            Corrector object containing a list of ImageCorrection in this case
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
        ...
        """
        super().__init__(first_row_time,
                         approx_geom,
                         width,
                         height,
                         wavelength,
                         orbit,
                         coord_corrector,
                         max_iterations,
                         tolerance)

        self.first_col_time = first_col_time
        self.range_frequency = range_frequency
        self.azimuth_frequency = azimuth_frequency


class Sentinel1BurstModel(Sentinel1SLCBaseModel):
    """Enables operations like projection and localization at the burst."""

    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 wavelength,
                 burst_times,
                 burst_roi,
                 approx_geom,
                 orbit: Orbit,
                 coord_corrector=Corrector(),
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1BurstModel used to perform projection and localization\
        in a Sentinel1 burst.

        Parameters
        ----------
        range_frequency : float
            Two way range time sampling frequency
        azimuth_frequency : float
            Azimuth time sampling frequency
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
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
        orbit: Orbit
            Orbit instance
        coord_corrector: eos.sar.projection_correction.Corrector
            Corrector object containing a list of ImageCorrection in this case
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

        # set these for the SLCCoordinateMixin
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
                         orbit,
                         coord_corrector,
                         max_iterations,
                         tolerance)

        # reset the initial azimuth guess at the center of burst
        self.azt_init = (burst_times[1] + burst_times[2]) / 2


class Sentinel1SwathModel(Sentinel1SLCBaseModel):
    """Enables operations like projection and localization at a swath."""

    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 wavelength,
                 bursts_times,
                 bursts_rois,
                 bursts_approx_geom,
                 bsids,
                 orbit: Orbit,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1SwathModel used to perform projection and localization\
        in a Sentinel1 swath.

        Parameters
        ----------
        range_frequency : float
            Two way range time sampling frequency
        azimuth_frequency : float
            Azimuth time sampling frequency
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
        wavelength: float
            wavelength in m
        bursts_times : list of (3,) tuple (start_time, start_valid, end_valid)
            start_time is the azimuth time of the first line in the burst
            start/end_valid denote the azimuth time of the
            first/last valid line in the burst.
        bursts_rois : list of (4,) tuple (x, y, w, h)
            Coordinates of the burst in the sentinel-1 raster file.
        bursts_approx_geom : List of list of tuples
            Each element is a list of tuples (lon, lat) corners of the approx geom
            of the burst
        bsids: list of str
            BSID of each burst of the model
        orbit: Orbit
            Orbit instance
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
        # set these for the SLCCoordinateMixin
        first_row_time = bursts_times[0][1]  # start valid

        self.col_min = min(roi_[0] for roi_ in bursts_rois)
        first_col_time = slant_range_time + self.col_min / range_frequency
        # swath polygon
        approx_geom = bursts_approx_geom[0][:2] + bursts_approx_geom[-1][2:]

        # setting image size
        self.row_min = bursts_rois[0][1]
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
                         orbit,
                         Corrector(),
                         max_iterations,
                         tolerance)

        # additional burst params, will surely be needed for coord conversion
        self.bursts_times = bursts_times
        self.bursts_rois = [roi.Roi.from_roi_tuple(_roi) for _roi in bursts_rois]
        self.bsids = bsids

        # reset the initial azimuth guess at the center of swath
        self.azt_init = (self.bursts_times[0][1] + self.bursts_times[-1][2]) / 2

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
        orig = (col, int(np.round(row)))
        return orig

    def compute_overlaps(self):
        """
        Compute the number of lines that overlap (cover the same ground feature)
        between the end of a burst and the start of another, based on the azimuth time.
        Do this for all bursts and store results in self.overlaps.
        Also store all osids in self.osids (set).

        Returns
        -------
        None.

        """
        n_bursts = len(self.bursts_times)

        self.overlaps = []
        self.osids = set()
        for i in range(n_bursts - 1):
            # ith overlap between i and i+1 burst
            current_burst_end = self.bursts_times[i][2]
            next_burst_start = self.bursts_times[i + 1][1]
            self.overlaps.append(int(np.round((current_burst_end -
                                               next_burst_start) * self.azimuth_frequency)))

            bsint = sentinel1.overlap.Bsint(self.bsids[i:i + 2])
            self.osids.update(bsint.osids())

    def get_overlaps_roi(self):
        """
        Computes the overlap rois within a swath.

        Returns
        -------
        osids: set[sentinel1.overlap.Osid]
            osids for the corresponding swath
        within_burst_rois: dict osid -> roi.Roi
            Rois of overlap region within burst.
        write_rois : dict osid -> roi.Roi
            Write roi in the final overlap array
        out_shapes : dict osid -> tuple
            (overalp_height, swath_width) tuple.
        within_swath_rois : dict osid -> roi.Roi
            Rois within the swath of the output arrays
        """
        if not hasattr(self, 'overlaps'):
            self.compute_overlaps()

        n_bursts = len(self.bursts_times)

        within_burst_rois = {}
        out_shapes = {}
        write_rois = {}
        within_swath_rois = {}

        for i in range(n_bursts - 1):
            bsint = sentinel1.overlap.Bsint(self.bsids[i:i + 2])
            end_of_burst, start_of_next_burst = bsint.osids()

            ovl_h = self.overlaps[i]
            # previous burst
            h, w = self.bursts_rois[i].get_shape()
            within_burst_rois[end_of_burst] = roi.Roi(0, h - ovl_h, w, ovl_h)
            # next burst
            _, w = self.bursts_rois[i + 1].get_shape()
            within_burst_rois[start_of_next_burst] = roi.Roi(0, 0, w, ovl_h)

            for osid, burst_id in zip([end_of_burst, start_of_next_burst], range(i, i + 2)):

                _, ovl_row, ovl_w, ovl_h = within_burst_rois[osid].to_roi()

                out_shapes[osid] = (ovl_h, self.w)

                col_shift, row_shift = self.burst_orig_in_swath(burst_id)

                write_rois[osid] = roi.Roi(col_shift, 0, ovl_w, ovl_h)

                within_swath_rois[osid] = roi.Roi(0, ovl_row + row_shift, self.w, ovl_h)

        return self.osids, within_burst_rois, write_rois, out_shapes, within_swath_rois

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
        remove_lines_at_top = ovl_prev // 2
        remove_lines_at_bottom = ovl_next - ovl_next // 2
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

    def get_debursting_rois(self, roi_in_swath=None, adjust_roi_to_swath=True):
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
        bsids : list of bsids
            BSID of each burst intersecting the ROI.
        rois_read : dict bsid -> eos.sar.roi.Roi
            Each roi corresponds to the region to be read from
            the tiff file.
        rois_write : dict bsid -> eos.sar.roi.Roi
            Each roi corresponds to the region where the output
            data should be written in the output image.
        out_shape: tuple
            (h, w) Output image shape

        """
        if roi_in_swath is None:
            roi_in_swath = roi.Roi(0, 0, self.w, self.h)

        if adjust_roi_to_swath:
            roi_in_swath = self.adjust_roi_to_swath(roi_in_swath)

        col, row, w, h = roi_in_swath.to_roi()
        out_shape = (h, w)

        col += self.col_min  # x is now relative to the tiff img
        previous_bursts_h = 0  # current y in the input image fully debursted
        previous_roi_h = 0  # current y in the output crop

        bsids = set()
        within_burst_rois = {}
        rois_write = {}

        for bid, bsid in enumerate(self.bsids):
            # burst roi without overlap relative to tiff img
            bcol, brow, bw, bh = self.burst_roi_without_ovl(bid).translate_roi(
                self.bursts_rois[bid].col,
                self.bursts_rois[bid].row).to_roi()
            # loop until we find first burst vertically intersecting roi
            if previous_bursts_h + bh > row + previous_roi_h:
                col_min = max(col, bcol)
                col_max = min(col + w, bcol + bw)
                debursted_to_tif = brow - previous_bursts_h
                row_min = row + previous_roi_h + debursted_to_tif
                row_max = min(row + h, previous_bursts_h + bh) + \
                    debursted_to_tif
                col_size = col_max - col_min
                row_size = row_max - row_min

                write_roi = roi.Roi(col_min - col, previous_roi_h, col_size, row_size)
                if write_roi.w > 0 and write_roi.h > 0:
                    bsids.add(bsid)

                    within_burst_rois[bsid] = roi.Roi(
                        col_min - self.bursts_rois[bid].col,
                        row_min - self.bursts_rois[bid].row,
                        col_size, row_size)
                    rois_write[bsid] = write_roi

                previous_roi_h += row_size
            previous_bursts_h += bh
            if previous_bursts_h >= row + h:
                break

        return bsids, within_burst_rois, rois_write, out_shape


class Sentinel1MosaicModel(Sentinel1SLCBaseModel):
    """Enables operations like projection and localization at a mosaic."""

    def __init__(self,
                 first_row_time,
                 first_col_time,
                 approx_geom,
                 range_frequency,
                 azimuth_frequency,
                 width,
                 height,
                 wavelength,
                 orbit: Orbit,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1MosaicModel used to perform projection and localization\
        in a Sentinel1 mosaic.

        Parameters
        ----------
        first_row_time: float
            Azimuth time of the first line in the image
        first_col_time: float
            Two way slant range time of the first column in the image
        range_frequency : float
            Two way range time sampling frequency .
        azimuth_frequency : float
            Azimuth time sampling frequency.
        wavelength: float
            wavelength in m
        width: int
            width of the image
        height: int
            height of the image
        orbit: Orbit
                Orbit instance
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

        """
        super().__init__(first_row_time,
                         first_col_time,
                         approx_geom,
                         range_frequency,
                         azimuth_frequency,
                         width,
                         height,
                         wavelength,
                         orbit,
                         Corrector(),
                         max_iterations,
                         tolerance)

    def to_dict(self) -> dict:
        metadata = dict(
            range_frequency=self.range_frequency,
            azimuth_frequency=self.azimuth_frequency,
            wavelength=self.wavelength,
            first_col_time=self.first_col_time,
            first_row_time=self.first_row_time,
            width=self.w,
            height=self.h,
            approx_geom=self.approx_geom,
            orbit=self.orbit.to_dict(),
            max_iterations=self.max_iterations,
            tolerance=self.localization_tolerance,
        )
        return metadata

    @staticmethod
    def from_dict(dict):
        # do a copy since it gets modified
        dict = dict.copy()
        dict['orbit'] = Orbit.from_dict(dict['orbit'])
        return Sentinel1MosaicModel(**dict)

    def to_cropped_mosaic(self, roi: roi.Roi):
        first_col_time = self.first_col_time + roi.col / self.range_frequency
        first_row_time = self.first_row_time + roi.row / self.azimuth_frequency

        approx_geom, _, _ = self.get_approx_geom(roi, elev=None) # TODO

        model = Sentinel1MosaicModel(
            first_row_time,
            first_col_time,
            approx_geom,
            self.range_frequency,
            self.azimuth_frequency,
            roi.w,
            roi.h,
            self.wavelength,
            self.orbit,
            max_iterations=self.max_iterations,
            tolerance=self.localization_tolerance,
        )
        return model


def swath_model_from_bursts_meta(bursts_metadata: list[Sentinel1BurstMetadata],
                                 orbit: Orbit,
                                 **kwargs) -> Sentinel1SwathModel:
    """
    Generate Sentinel1SwathModel instance from list of bursts metadata.

    Parameters
    ----------
    bursts_metadata : list of Sentinel1BurstMetadata
        each object contains attribute metadata relative to the bursts of the
        swath.
    orbit: Orbit
        Orbit instance
    **kwargs : keyword arguments
        one of degree, bistatic_correction, apd_correction, max_iterations,
        tolerance. Processing parameters for the projection and localization.

    Returns
    -------
    Sentinel1SwathModel
        Model for projection and localization inside the swath.
    """
    # TODO: aggregate state_vectors as well
    bursts_times = [b.burst_times for b in bursts_metadata]
    bursts_rois = [b.burst_roi for b in bursts_metadata]
    bursts_approx_geom = [b.approx_geom for b in bursts_metadata]
    bsids = [b.bsid for b in bursts_metadata]

    def alleq(prop):
        burst = bursts_metadata[0]
        return all(getattr(b, prop) == getattr(burst, prop) for b in bursts_metadata)
    assert alleq('range_frequency')
    assert alleq('azimuth_frequency')
    assert alleq('slant_range_time')
    assert alleq('wave_length')
    assert alleq('pri')
    assert alleq('rank')

    return Sentinel1SwathModel(bursts_metadata[0].range_frequency,
                               bursts_metadata[0].azimuth_frequency,
                               bursts_metadata[0].slant_range_time,
                               bursts_metadata[0].wave_length,
                               bursts_times,
                               bursts_rois,
                               bursts_approx_geom,
                               bsids,
                               orbit,
                               **kwargs)


class Sentinel1GRDModel(coordinates.GRDCoordinateMixin, Sentinel1BaseModel):
    """Enables operations like projection and localization at a mosaic."""

    def __init__(self,
                 first_row_time,
                 approx_geom,
                 range_pixel_spacing,
                 azimuth_time_interval,
                 width,
                 height,
                 wavelength,
                 orbit,
                 srgr,
                 corrector,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1MosaicModel used to perform projection and localization\
        in a Sentinel1 mosaic.

        Parameters
        ----------
        first_row_time: float
            Azimuth time of the first line in the image
        approx_geom: list of tuples (lon, lat)
            Approximate geometry of the image (represented by 4 corners)
        range_pixel_spacing: float
        azimuth_time_interval: float
        width: int
            width of the image
        height: int
            height of the image
        wavelength: float
            wavelength in m
        orbit: Orbit
            Orbit instance
        srgr: eos.sar.srgr.SRGRConverter
            SRGR converter for the GRDCoordinateMixin
        corrector: eos.sar.projection_correction.Corrector
            Corrector object containing a list of ImageCorrection in this case
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

        """
        super().__init__(first_row_time,
                         approx_geom,
                         width,
                         height,
                         wavelength,
                         orbit,
                         corrector,
                         max_iterations,
                         tolerance)

        self.srgr = srgr
        self.azimuth_time_interval = azimuth_time_interval
        self.range_pixel_spacing = range_pixel_spacing


def secondary_project_and_correct(proj_model, x, y, alt, crs, bsids,
                                  corrector_per_bsid, pts_in_burst_mask):
    """
    Project points and correct them in the primary swath.

    Parameters
    ----------
    proj_model : Sentinel1BaseModel
        Model containing the bursts.
    x : array
        x coordinate of points.
    y : array
        y coordinate of points.
    alt : array
        Altitude of points.
    crs : any crs type accepted by pyproj
        CRS of the points.
    bsids : Iterable
        BSIDs of the specific bursts where we wish to have corrected coordinates.
    corrector_per_bsid : dict bsid -> eos.sar.projection_correction.Corrector
        Associated correctors (for coords in this case) to the previous ids.
    pts_in_burst_mask : dict bsid -> ndarray
        Each element is a mask defining which points from the initial x, y, alt arrays
        should be projected in the different bursts.
    Returns
    -------
    azt_no_correc : dict bsid -> array
       Each element is an array of azimuth times without corrections.
    rng_no_correc : dict bsid -> array
       Each element is an array of ranges without corrections.
    azt_correc : dict bsid -> array
       Each element is an array of azimuth times with corrections.
    rng_correc : dict bsid -> array
        Each element is an array of ranges with corrections.

    """

    transformer = pyproj.Transformer.from_crs(
        crs, 'epsg:4978', always_xy=True)
    # convert to geocentric cartesian
    gx, gy, gz = transformer.transform(x, y, alt)

    azt_correc = {}
    rng_correc = {}
    azt_no_correc = {}
    rng_no_correc = {}
    for bsid in bsids:
        corrector = corrector_per_bsid[bsid]
        burst_mask = pts_in_burst_mask[bsid]

        # project points that should fall in secondary burst
        # (according to previous primary projection)
        azt_no_correc[bsid], rng_no_correc[bsid], _ = proj_model.projection(
            gx[burst_mask], gy[burst_mask], gz[burst_mask], crs='epsg:4978', as_azt_rng=True)

        # create geo_im_pt
        geo_im_pt = GeoImagePoints(gx[burst_mask], gy[burst_mask], gz[burst_mask],
                                   azt_no_correc[bsid], rng_no_correc[bsid])

        # estimate and apply corrections
        geo_im_pt = corrector.estimate_and_apply(geo_im_pt)

        # store corrected coords
        azt_correc[bsid], rng_correc[bsid] = geo_im_pt.get_azt_rng()

    return azt_no_correc, rng_no_correc, azt_correc, rng_correc
