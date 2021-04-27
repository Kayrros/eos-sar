import numpy as np
import pyproj
from eos.sar import model, range_doppler, const, coordinates, orbit


def burst_model_from_s1m(s1model, burst, **kwargs):
    """Create a Sentinel1BurstModel from a s1m.Sentinel1Model


    Parameters
    ----------
    s1model : s1m.Sentinel1Model
        Object encapsulating the metadata and processing for one Sentinel1 
        subswath.
    burst : int
        burst id in the s1model subswath, 0 based.
    **kwargs : keyword arguments for the constructor of Sentinel1BurstModel.

    Returns
    -------
    bmodel: Sentinel1BurstModel instance.

    """
    assert (burst >= 0) and burst < len(
        s1model.burst_times), "burst id out of range"
    return Sentinel1BurstModel(s1model.range_frequency,
                               s1model.azimuth_frequency,
                               s1model.slant_range_time,
                               s1model.samples_per_burst,
                               s1model.burst_times[burst],
                               s1model.burst_rois[burst],
                               s1model.burst_lon_lat_bboxes[burst],
                               s1model.state_vectors,
                               **kwargs)


class Sentinel1BurstModel(coordinates.CoordinateMixin, model.SensorModel):
    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 samples_per_burst,
                 burst_times,
                 burst_roi,
                 lon_lat_bbox,
                 state_vectors,
                 degree=11,
                 bistatic_correction=True,
                 apd_correction=True,
                 max_iterations=20,
                 tolerance=0.001):
        """Sentinel1BurstModel used to perform projection and localization
        in a Sentinel1 burst
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
        lon_lat_bbox : (4,2) ndarray
            The lon, lat coordinates of the four corners.
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
            For projection, the tolerance is considered on the satellite position 
            of closest approach. Converted to azimuth time tolerance using the speed. 
            The default is 0.001.

        Returns
        -------
        None.

        """

        # set these for the CoordinateMixin
        self.first_row_time = burst_times[1]  # start valid
        self.first_col_time = slant_range_time + burst_roi[0] / range_frequency
        self.range_frequency = range_frequency
        self.azimuth_frequency = azimuth_frequency

        # generic to all bursts in a product
        self.slant_range_time = slant_range_time
        self.samples_per_burst = samples_per_burst
        self.orbit = orbit.Orbit(state_vectors, degree)

        # specific to current burst
        self.burst_times = burst_times
        self.burst_roi = burst_roi
        self.lon_lat_bbox = lon_lat_bbox

        # processing params
        self.bistatic_correction = bistatic_correction
        self.apd_correction = apd_correction

        self.max_iterations = max_iterations
        self.localization_tolerance = tolerance
        self.projection_tolerance = tolerance \
            / np.linalg.norm(state_vectors[0]['velocity'])

    def projection(self, xs, ys, alts, crs='epsg:4326', vert_crs=None):
        """Projects a 3D point into the burst coordinates

        Parameters
        ----------
        xs, ys : ndarray or scalar
            Coordinates in the crs defined by crs parameter. 
        alts: ndarray or scalar 
            Altitude defined by vert_crs if provided or EARTH_WGS84 ellipsoid.
        crs : string, optional
            CRS in which the point is given
                    Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
        vert_crs: string, optional 
            Vertical crs 

        Returns
        -------
        cols : ndarray or scalar
            Column coordinate in burst referenced to the first valid column.
        rows : ndarray or scalar
            Row coordinate in burst referenced to the first valid line.
        i : ndarray or scalar
            Incidence angle.

        """
        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)
        alts = np.atleast_1d(alts)

        if vert_crs is None:
            src_crs = crs
        else:
            src_crs = pyproj.crs.CompoundCRS(
                name='ukn_reference', components=[crs, vert_crs])

        transformer = pyproj.Transformer.from_crs(
            src_crs, 'epsg:4978', always_xy=True)

        # convert to geocentric cartesian
        X, Y, Z = transformer.transform(xs, ys, alts)

        # project in the slc image
        tinit = (self.burst_times[1] +
                 self.burst_times[2])/2 * np.ones_like(xs)
        azt, rng, i = range_doppler.iterative_projection(self.orbit, X, Y, Z,
                                                         tinit, self.max_iterations,
                                                         self.projection_tolerance)
        # Apply corrections on rng and azt if needed
        if self.apd_correction:
            alts = alts.squeeze()
            rng += (alts * alts / 8.55e7 - alts / 3411.0 + 2.41) / np.cos(i)

        # bistatic residual error correction, as described by Schubert et al in
        # Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
        # Results. Remote Sens. 7, 9431-9449 (2015)
        if self.bistatic_correction:
            # slant range (col coordinate)
            col = (2 * rng / const.LIGHT_SPEED_M_PER_SEC -
                   self.slant_range_time) * self.range_frequency

            azt -= (col - 0.5 * self.samples_per_burst
                    ) / (2 * self.range_frequency)

        # convert to row and col
        rows, cols = self.to_row_col(azt, rng)

        return cols, rows, i

    def localization(self, cols, rows, alts, crs='epsg:4326', vert_crs=None):
        """

        Parameters
        ----------

        cols : ndarray or scalar
            column coordinate in burst referenced to the first valid column.
        rows : ndarray or scalar
            row coordinate in burst referenced to the first valid line.
        alts : ndarray or scalar
            Altitude above the EARTH_WGS84 ellipsoid.
        crs : string, optional
            CRS in which the point is returned
                    Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
        vert_crs: string, optional 
            Vertical crs in which the point is returned 

        Returns
        -------
        x, y, z: ndarray or scalar
            Coordinates of the point in the crs

        """
        # make sure we work with numpy arrays
        cols = np.atleast_1d(cols)
        rows = np.atleast_1d(rows)
        alts = np.atleast_1d(alts)

        num_pts = len(cols)

        # image coordinates to range and az time
        azt, rng = self.to_azt_rng(rows, cols)

        # Make corrections on azt and rng if needed
        if self.bistatic_correction:
            # correct azimuth time
            azt += (cols + self.burst_roi[0] -
                    0.5*self.samples_per_burst)/(2*self.range_frequency)

        if self.apd_correction:
            # evaluate satellite position
            positions = self.orbit.evaluate(azt)
            # Rough estimation of geometry
            Lsat = np.linalg.norm(positions, axis=1)

            # Earth radius taken at the intersection of the line joining satellite
            # and earth center with the ellipsoid
            ell_axis = const.EARTH_WGS84_AXIS_A_M * np.ones(3)
            ell_axis[2] = const.EARTH_WGS84_AXIS_B_M
            ERadius = Lsat/np.sqrt(np.sum((positions/ell_axis)**2, axis=1))

            # cosine rule
            incidence = np.arccos(
                (Lsat**2 - (ERadius+alts)**2 - rng**2) / (2 * (ERadius + alts) * rng))

            # correct range
            rng -= (alts**2/8.55e7 - alts/3411.0 + 2.41)/np.cos(incidence)

        # initial geocentric point xyz definition
        # from lon, lat, alt to x, y, z
        toXYZ = pyproj.Transformer.from_crs(
            'epsg:4326', 'epsg:4978', always_xy=True)

        # point at swath centroid, 0 altitude as init
        XYZ = np.array(toXYZ.transform(*np.mean(self.lon_lat_bbox, axis=0), 0))
        XYZ = np.repeat(XYZ.reshape(1, 3), repeats=num_pts, axis=0)

        # localize each point
        points3D = range_doppler.iterative_localization(
            self.orbit, azt, rng, alts, XYZ, self.max_iterations, self.localization_tolerance)

        if vert_crs is None:
            dst_crs = crs
        else:
            dst_crs = pyproj.crs.CompoundCRS(
                name='ukn_reference', components=[crs, vert_crs])
        todst = pyproj.Transformer.from_crs(
            'epsg:4978', dst_crs, always_xy=True)
        x, y, z = todst.transform(*points3D.T)

        return x, y, z