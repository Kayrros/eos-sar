#cython: binding=False
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: embedsignature=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: always_allow_keywords=False
#cython: profile=False
#cython: infer_types=False

from typing import Union
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport M_PI, sin, cos, sqrt, ceil, floor, acos

from eos.sar import const, model
from eos.sar.roi import Roi
from eos.sar.coordinates import GRDCoordinate, SLCCoordinate
import eos.dem

from affine import Affine
import rasterio
import rasterio.crs
import rasterio.transform
from rasterio.warp import reproject, Resampling
from rasterio.control import GroundControlPoint


# from GeoUtils (snap-engine)
cdef inline (double, double, double) geo2xyzWGS84(double latitude, double longitude, double altitude) noexcept nogil:
    """"
    Convert geodetic coordinate into cartesian XYZ coordinate with specified geodetic system (WGS84)

    Equivalent to pyproj.Transformer.from_crs('epsg:4979', 'epsg:4978') but faster.

    Inputs:
        latitude  The latitude of a given pixel (in degree).
        longitude The longitude of a given pixel (in degree).
        altitude  The altitude of the given pixel (in m)
    Outputs:
        x/y/z     cartesian coordinates in the geodetic system
    """
    cdef double WGS84_a = 6378137.0
    cdef double WGS84_b = 6356752.3142451794975639665996337
    cdef double WGS84_earthFlatCoef = 1.0 / ((WGS84_a - WGS84_b) / WGS84_a)
    cdef double WGS84_e2 = 2.0 / WGS84_earthFlatCoef - 1.0 / (WGS84_earthFlatCoef * WGS84_earthFlatCoef)

    cdef double lat = latitude * (M_PI / 180.)
    cdef double lon = longitude * (M_PI / 180.)

    cdef double sinLat = sin(lat)
    cdef double N = WGS84_a / sqrt(1.0 - WGS84_e2 * sinLat * sinLat)
    cdef double NcosLat = (N + altitude) * cos(lat)

    cdef double x, y, z

    x = NcosLat * cos(lon) # in m
    y = NcosLat * sin(lon) # in m
    z = (N + altitude - WGS84_e2 * N) * sinLat

    return x, y, z

cdef inline double computeElevationAngle(double earthPoint_x, double earthPoint_y, double earthPoint_z,
                                         double sensorPos_x, double sensorPos_y, double sensorPos_z) noexcept nogil:
    """
    Compute the elevation angle (in degree)
    Inputs:
        earthPoint: WGS84 position of a point on earth
        sensorPos: Position of the satellite
    Outputs:
        Elevation angle in degree
    """
    cdef double xDiff = sensorPos_x - earthPoint_x
    cdef double yDiff = sensorPos_y - earthPoint_y
    cdef double zDiff = sensorPos_z - earthPoint_z
    cdef double slantRange2 = xDiff * xDiff + yDiff * yDiff + zDiff * zDiff
    cdef double H2 = sensorPos_x * sensorPos_x + sensorPos_y * sensorPos_y + sensorPos_z * sensorPos_z
    cdef double R2 = earthPoint_x * earthPoint_x + earthPoint_y * earthPoint_y + earthPoint_z * earthPoint_z
    cdef double angle = acos((slantRange2 + H2 - R2) / (2 * sqrt(slantRange2 * H2))) * (180.0 / M_PI)
    return angle

cdef inline double computeIlluminatedArea(double t00_x, double t00_y, double t00_z,
                                          double t01_x, double t01_y, double t01_z,
                                          double t10_x, double t10_y, double t10_z,
                                          double t11_x, double t11_y, double t11_z,
                                          double sld_x, double sld_y, double sld_z) noexcept nogil:
    """
    Inputs:
         x/y/z (WGS84) of t00, t01, t10 and t11
         sld (slantRangeDir) the normalized slant range direction vector
    Outputs:
         corresponding gamma0
    """
    # project points t00, t01, t10 and t11 to the plane perpendicular to the slant range direction vector
    cdef double t00s = t00_x * sld_x + t00_y * sld_y + t00_z * sld_z
    cdef double t01s = t01_x * sld_x + t01_y * sld_y + t01_z * sld_z
    cdef double t10s = t10_x * sld_x + t10_y * sld_y + t10_z * sld_z
    cdef double t11s = t11_x * sld_x + t11_y * sld_y + t11_z * sld_z
    cdef double p00_x = t00_x - t00s * sld_x
    cdef double p00_y = t00_y - t00s * sld_y
    cdef double p00_z = t00_z - t00s * sld_z
    cdef double p01_x = t01_x - t01s * sld_x
    cdef double p01_y = t01_y - t01s * sld_y
    cdef double p01_z = t01_z - t01s * sld_z
    cdef double p10_x = t10_x - t10s * sld_x
    cdef double p10_y = t10_y - t10s * sld_y
    cdef double p10_z = t10_z - t10s * sld_z
    cdef double p11_x = t11_x - t11s * sld_x
    cdef double p11_y = t11_y - t11s * sld_y
    cdef double p11_z = t11_z - t11s * sld_z

    # compute distances between projected points
    cdef double p00p01 = sqrt((p00_x-p01_x)**2 + (p00_y-p01_y)**2 + (p00_z-p01_z)**2)
    cdef double p00p10 = sqrt((p00_x-p10_x)**2 + (p00_y-p10_y)**2 + (p00_z-p10_z)**2)
    cdef double p11p01 = sqrt((p11_x-p01_x)**2 + (p11_y-p01_y)**2 + (p11_z-p01_z)**2)
    cdef double p11p10 = sqrt((p11_x-p10_x)**2 + (p11_y-p10_y)**2 + (p11_z-p10_z)**2)
    cdef double p10p01 = sqrt((p10_x-p01_x)**2 + (p10_y-p01_y)**2 + (p10_z-p01_z)**2)

    # compute semi-perimeters of two triangles: p00-p01-p10 and p11-p01-p10
    cdef double h1 = 0.5 * (p00p01 + p00p10 + p10p01)
    cdef double h2 = 0.5 * (p11p01 + p11p10 + p10p01)

    cdef double gamma0 = \
        sqrt(max(0., h1 * (h1 - p00p01) * (h1 - p00p10) * (h1 - p10p01))) + \
        sqrt(max(0., h2 * (h2 - p11p01) * (h2 - p11p10) * (h2 - p10p01)))
    return gamma0

cdef inline void saveIlluminationArea(int x0, int y0, int w, int h, double azimuthIndex,
                         double rangeIndex, double gamma0Area,
                         double[:,:] gamma0ReferenceArea_view) noexcept nogil:
    """
    Distribute the local illumination area to the 4 adjacent pixels using bi-linear distribution.
    Inputs:
         x0                  The x coordinate of the pixel at the upper left corner of current tile.
         y0                  The y coordinate of the pixel at the upper left corner of current tile.
         w                   The tile width (= gamma0ReferenceArea_view.shape[1])
         h                   The tile height (= gamma0ReferenceArea_view.shape[0])
         azimuthIndex        Azimuth pixel index for the illuminated area.
         rangeIndex          Range pixel index for the illuminated area.
         gamma0Area          The illuminated area.
         gamma0ReferenceArea Buffer for the simulated image.
    """
    cdef int ia0 = int(floor(azimuthIndex))
    cdef int ia1 = ia0 + 1
    cdef int ir0 = int(floor(rangeIndex))
    cdef int ir1 = ir0 + 1

    cdef double wr = rangeIndex - ir0
    cdef double wa = azimuthIndex - ia0
    cdef double wac = 1 - wa
    cdef double wrc

    if ir0 >= x0 and ir0 < x0 + w:
        wrc = 1 - wr
        if ia0 >= y0 and ia0 < y0 + h:
            gamma0ReferenceArea_view[ia0 - y0][ir0 - x0] += wrc * wac * gamma0Area
        if ia1 >= y0 and ia1 < y0 + h:
            gamma0ReferenceArea_view[ia1 - y0][ir0 - x0] += wrc * wa * gamma0Area
    if ir1 >= x0 and ir1 < x0 + w:
        if ia0 >= y0 and ia0 < y0 + h:
            gamma0ReferenceArea_view[ia0 - y0][ir1 - x0] += wr * wac * gamma0Area
        if ia1 >= y0 and ia1 < y0 + h:
            gamma0ReferenceArea_view[ia1 - y0][ir1 - x0] += wr * wa * gamma0Area
            
            
class SARSimulator:

    coordinate: Union[SLCCoordinate, GRDCoordinate]

    def __init__(self,
                 proj_model: model.SensorModel,
                 dem: eos.dem.DEM,
                 oversampling=(4, 4)):
        self.proj_model = proj_model
        self.dem = dem
        self.coordinate = proj_model.coordinate

        self.oversampling_x, self.oversampling_y = oversampling

    def _get_sar_resolutions(self, int col, int row):
        # compute the azimuth and range 'resolution' by localizing two points in each direction
        # and measure the distance in xyz 4978 Coordinates
        # it's not really a resolution, but more a ground sampling distance
        rows = [row, row,   row+1]
        cols = [col, col+1, col]
        p = np.asarray(self.proj_model.localization(rows, cols, np.zeros_like(rows), crs='epsg:4978'))
        p = (p - p[:,0:1])[:,1:]
        range_resolution, azimuth_resolution = np.linalg.norm(p, axis=0)
        return range_resolution, azimuth_resolution

    def _get_dem_transform(self, proj_model, roi):
        x0, y0, w, h = roi.to_roi()
        row = np.asarray([y0, y0, y0+h, y0+h])
        col = np.asarray([x0, x0+w, x0+w, x0])
        lon, lat, _ = proj_model.localization(row, col, alt=np.zeros_like(col))

        gcps = [GroundControlPoint(row=r-y0, col=c-x0, x=ln, y=lt) for r, c, ln, lt in zip(row, col, lon, lat)]
        dst_transform = rasterio.transform.from_gcps(gcps)

        return dst_transform

    def _resample_dem(self, roi: Roi):
        h, w = roi.get_shape()
        geometry, _, _ = self.proj_model.get_approx_geom(roi, dem=self.dem)
        lons, lats = zip(*geometry)
        bounds = (min(lons), min(lats), max(lons), max(lats))
        src_raster, src_transform, crs = self.dem.crop(bounds)

        # NOTE: an affine transform is a poor approximation?
        # we could warp with a higher order model, and use it to get the lon,lat as well
        transform = self._get_dem_transform(self.proj_model, roi)

        fx = self.oversampling_x
        fy = self.oversampling_y
        ### MODIF Arthur Hauck 19/01/2024
        shift = 0
        if self.dem.dst_area_or_point == "Point":
            shift = 0.5
        transform = transform * Affine.scale(1 / fx, 1 / fy) * Affine.translation(shift, shift)
        ### FIN MODIF
        nw = int(ceil(w * fx))
        nh = int(ceil(h * fy))
        # TODO: add 0.5 pixel to the transform for pixel is point? ### YES! (Arthur Hauck 19/01/2024)

        dem = np.zeros((nh, nw), dtype=np.float32)
        reproject(
            src_raster,
            dem,
            src_transform=src_transform,
            src_crs=crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.cubic)

        return dem, transform

    def simulate(self, roi: Roi, *, extends_roi=True, detect_shadows=True):
        """
        Inputs:
            roi: region of interest, defined according to the proj_model
            extends_roi (optional): extends the roi internally due to SAR geometric distortion
            detect_shadows (optional): sets to 0 values that are not visible by the sensor due to elevation
        Outputs:
            array: simulated image of the shape of the roi
        """
        cdef int _detect_shadows = detect_shadows
        assert roi.col == int(roi.col)
        assert roi.row == int(roi.row)
        assert roi.w == int(roi.w)
        assert roi.h == int(roi.h)

        cdef int x0, y0, w, h
        x0, y0, w, h = roi.to_roi()

        # NOTE: rangeSpacing and azimuthSpacing are defined by the IPF
        # here, we use a different definition for the azimuthSpacing simply because it is more convenient
        cdef double azimuthSpacing = self._get_sar_resolutions(x0 + w // 2, y0 + h // 2)[1]
        cdef double rangeSpacing
        if isinstance(self.coordinate, GRDCoordinate):
            rangeSpacing = self.coordinate.range_pixel_spacing
        elif isinstance(self.coordinate, SLCCoordinate):
            rangeSpacing = const.LIGHT_SPEED_M_PER_SEC / (2 * self.coordinate.range_frequency)
        else:
            assert False

        # TODO: make sure this definition of aBeta is ok
        cdef double aBeta = azimuthSpacing * rangeSpacing

        outer_roi = roi.copy()
        if extends_roi:
            self._adjust_outer_roi(outer_roi)
        # add a bit of y margins for the bilinear splatting
        outer_roi.row -= 2
        outer_roi.h += 3
        dem_, dem_transform = self._resample_dem(outer_roi)
        cdef np.ndarray[np.float32_t, ndim=2] dem = dem_

        cdef int Nrows = dem.shape[0] - 1
        cdef int Ncols = dem.shape[1] - 1
        cdef np.ndarray[np.int64_t, ndim=1] rrow = np.arange(0, Nrows+1)
        cdef np.ndarray[np.int64_t, ndim=1] rcol = np.arange(0, Ncols+1)

        cdef double gamma0
        cdef double elevation

        cdef np.ndarray[np.float64_t, ndim=2] gamma0ReferenceArea = np.zeros([h, w], dtype=np.float64)
        cdef double [:,:] gamma0ReferenceArea_view = gamma0ReferenceArea
        # posData
        cdef double sensorPos_x, sensorPos_y, sensorPos_z
        cdef double earthPoint_x, earthPoint_y, earthPoint_z
        cdef double t00_x, t00_y, t00_z
        cdef double t01_x, t01_y, t01_z
        cdef double t10_x, t10_y, t10_z
        cdef double t11_x, t11_y, t11_z
        cdef double sld_x, sld_y, sld_z, norm
        # altitude
        cdef double maxElevAngle

        cdef np.ndarray[np.float64_t, ndim=1] row0_lons = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_lats = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_alts = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] row0_gx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_gy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_gz = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gz = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] row0_sx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_sy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_sz = np.empty((Ncols+1,))

        cdef np.ndarray[np.int8_t, ndim=1] row0_successes = np.empty((Ncols+1,), dtype=np.int8)
        cdef np.ndarray[np.float64_t, ndim=1] row0_rangeIndices = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_azimuthIndices = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] col0_lons = np.empty((Nrows+1,))
        cdef np.ndarray[np.float64_t, ndim=1] col0_lats = np.empty((Nrows+1,))
        cdef np.ndarray[np.float64_t, ndim=1] col0_alts = np.empty((Nrows+1,))
        cdef int mid = dem.shape[1] // 2
        col0_alts = dem[rrow, mid].astype(np.float64)
        col0_lons, col0_lats = dem_transform * (mid, rrow)
        row0, col0, _ = self.proj_model.projection(col0_lons, col0_lats, col0_alts)
        azt0 = self.coordinate.to_azt(row0)

        cdef np.ndarray[np.float64_t, ndim=2] sats = self.proj_model.orbit.evaluate(azt0)

        cdef int i, j
        for i in range(0, Nrows):
            maxElevAngle = 0.0

            # move previous row to row1_* and fill row0 with new data
            row1_gx, row0_gx = row0_gx, row1_gx
            row1_gy, row0_gy = row0_gy, row1_gy
            row1_gz, row0_gz = row0_gz, row1_gz

            # NOTE: if the DEM is nan, computations will be nan and `successes[j]` will be false
            row0_alts = dem[i].astype(np.float64)
            row0_lons, row0_lats = dem_transform * (rcol, i)
            for j in range(Ncols+1):  # geo2xyzWGS84 is much faster than pyproj
                row0_gx[j], row0_gy[j], row0_gz[j] = geo2xyzWGS84(row0_lats[j], row0_lons[j], row0_alts[j])

            ok = self._compute_ground_coordinates(x0, y0, w, h, row0_gx, row0_gy, row0_gz, azt0[i], sats[i],
                            row0_successes, row0_rangeIndices, row0_azimuthIndices)
            if not ok:  # fast path in case the azimuth is outside [y0, y0+h]
                continue

            sensorPos_x = sats[i,0]
            sensorPos_y = sats[i,1]
            sensorPos_z = sats[i,2]

            for j in range(Ncols-1):
                if not row0_successes[j]:
                    continue

                earthPoint_x = row0_gx[j]
                earthPoint_y = row0_gy[j]
                earthPoint_z = row0_gz[j]

                # Prepare computeIlluminatedArea: fetch positions of t00, t01, t10 and t11
                # TODO: is row1 OK? or do we want row-1 (next row)?
                # probably not a big deal for low res DEM, but could have an impact for high res DEM
                t00_x, t00_y, t00_z = row0_gx[j],   row0_gy[j],   row0_gz[j]
                t01_x, t01_y, t01_z = row1_gx[j],   row1_gy[j],   row1_gz[j]
                t10_x, t10_y, t10_z = row0_gx[j+1], row0_gy[j+1], row0_gz[j+1]
                t11_x, t11_y, t11_z = row1_gx[j+1], row1_gy[j+1], row1_gz[j+1]

                # Prepare computeIlluminatedArea: compute slant range direction
                sld_x = sensorPos_x - earthPoint_x
                sld_y = sensorPos_y - earthPoint_y
                sld_z = sensorPos_z - earthPoint_z
                norm = sqrt(sld_x*sld_x + sld_y*sld_y + sld_z*sld_z)
                sld_x /= norm
                sld_y /= norm
                sld_z /= norm

                gamma0 = computeIlluminatedArea(t00_x, t00_y, t00_z,
                                                t01_x, t01_y, t01_z,
                                                t10_x, t10_y, t10_z,
                                                t11_x, t11_y, t11_z,
                                                sld_x, sld_y, sld_z)

                if _detect_shadows:
                    elevation = computeElevationAngle(earthPoint_x, earthPoint_y, earthPoint_z,
                                                      sensorPos_x, sensorPos_y, sensorPos_z)
                else:
                    elevation = 0

                if elevation >= maxElevAngle:
                    maxElevAngle = elevation
                    saveIlluminationArea(x0, y0, w, h, row0_azimuthIndices[j], row0_rangeIndices[j],
                                         gamma0, gamma0ReferenceArea_view)

        gamma0ReferenceArea /= aBeta # Normalize (note: original code did it in outputSimulatedArea)
        return gamma0ReferenceArea

    def _compute_ground_coordinates(self, int x0, int y0, int w, int h,
                                   double[:] x,
                                   double[:] y,
                                   double[:] z,
                                   double azt0,
                                   np.ndarray[np.float64_t, ndim=1] sat,
                                   signed char[:] out_success,
                                   double[:] out_col,
                                   double[:] out_row
                                   ):
        """
        Project a set of ground points to col/row coordinates.
        Each point also has a 'success' flag that is False if the point falls outside the ROI.
        The azimuth is assumed constant in this function to speed up projections.
        The function returns False if projected azimuth time falls outside the ROI.

        Inputs:
            x0/y0/w/h    Region of interest
            x/y/z        Coordinates of the ground in 4978
            azt0         Azimuth time of the current line
            sat          Position of the satellite in 4978
            out_col      x coordinate on the ROI
            out_row      y coordinate on the ROI
        Outputs:
            out_success      Whether the operation succeeed (projection is in the ROI)
        """
        cdef double satx = sat[0]
        cdef double saty = sat[1]
        cdef double satz = sat[2]

        # if the row is not correct, stop early
        cdef double row0 = self.coordinate.to_row(azt0)
        # one pixel margin for the bilinear splatting
        if row0 < y0 - 1 or row0 >= y0 + h:
            return False

        out_row[:] = row0

        cdef long i
        cdef np.ndarray[np.float64_t, ndim=1] rng = np.empty_like(x)
        for i in range(x.size):
            rng[i] = sqrt((satx - x[i]) ** 2 + (saty - y[i]) ** 2 + (satz - z[i]) ** 2)

        cdef np.ndarray[np.float64_t, ndim=1] col = self.coordinate.to_col(rng, azt=azt0)
        for i in range(x.size):
            out_col[i] = col[i]

        for i in range(x.size):
            out_success[i] = (out_col[i] >= x0 - 1) & (out_col[i] <= x0 + w)

        return True

    def _adjust_outer_roi(self, roi: Roi, int N=10):
        """
            Compute the outer ROI. Each 3D point from the outer roi should be projected inside the inner ROI.
        """
        x0, y0, w, h = roi.to_roi()
        xx = np.linspace(x0, x0 + w, N)
        yy = np.linspace(y0, y0 + h, N)
        cols, rows = np.meshgrid(xx, yy)
        cols = cols.ravel()
        rows = rows.ravel()

        lon, lat, alt = self.proj_model.localization(rows, cols, alt=np.zeros_like(rows))
        alt = self.dem.elevation(lon, lat)
        _, cols2, _ = self.proj_model.projection(lon, lat, alt)

        # over-estimate a bit
        max_offset = int(ceil(np.nanmax(np.abs(cols - cols2)))) + 10

        roi.col -= max_offset
        roi.w += max_offset * 2



# Simulator Thibaud Ehret 2024-09-20
class SARSimulator_small_roi:
    """
    Warnings:
        1. Because of assumptions for optimizations (warping the DEM),
        the simulation performs poorly on large ROIs.
        You should split the processing in sub-ROI for such cases.

        2. The last line of the simulation tends to not be accurate. This is a bug.
    """

    coordinate: Union[SLCCoordinate, GRDCoordinate]

    def __init__(self,
                 proj_model: model.SensorModel,
                 dem: eos.dem.DEM,
                 oversampling=(4, 4)):
        self.proj_model = proj_model
        self.dem = dem
        self.coordinate = proj_model.coordinate

        self.oversampling_x, self.oversampling_y = oversampling

    def _get_sar_resolutions(self, int col, int row):
        # compute the azimuth and range 'resolution' by localizing two points in each direction
        # and measure the distance in xyz 4978 Coordinates
        # it's not really a resolution, but more a ground sampling distance
        rows = [row, row,   row+1]
        cols = [col, col+1, col]
        p = np.asarray(self.proj_model.localization(rows, cols, np.zeros_like(rows), crs='epsg:4978'))
        p = (p - p[:,0:1])[:,1:]
        range_resolution, azimuth_resolution = np.linalg.norm(p, axis=0)
        return range_resolution, azimuth_resolution

    def _get_dem_transform(self, proj_model, roi):
        x0, y0, w, h = roi.to_roi()
        row = np.asarray([y0, y0, y0+h, y0+h])
        col = np.asarray([x0, x0+w, x0+w, x0])
        lon, lat, _, masks = proj_model.localize_without_alt(row, col, dem=self.dem)
        assert np.all(masks), "Error during ROI projection"

        gcps = [GroundControlPoint(row=r-y0, col=c-x0, x=ln, y=lt) for r, c, ln, lt in zip(row, col, lon, lat)]
        dst_transform = rasterio.transform.from_gcps(gcps)

        return dst_transform

    def _resample_dem(self, roi: Roi):
        h, w = roi.get_shape()
        geometry, _, _ = self.proj_model.get_approx_geom(roi, dem=self.dem)
        lons, lats = zip(*geometry)
        bounds = (min(lons), min(lats), max(lons), max(lats))
        src_raster, src_transform, crs = self.dem.crop(bounds)

        # NOTE: an affine transform is a poor approximation?
        # we could warp with a higher order model, and use it to get the lon,lat as well
        transform = self._get_dem_transform(self.proj_model, roi)

        fx = self.oversampling_x
        fy = self.oversampling_y
        ### MODIF Arthur Hauck 19/01/2024
        shift = 0
        if self.dem.dst_area_or_point == "Point":
            shift = 0.5
        transform = transform * Affine.scale(1 / fx, 1 / fy) * Affine.translation(shift, shift)
        ### FIN MODIF
        nw = int(ceil(w * fx))
        nh = int(ceil(h * fy))

        dem = np.zeros((nh, nw), dtype=np.float32)
        reproject(
            src_raster,
            dem,
            src_transform=src_transform,
            src_crs=crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.cubic)

        return dem, transform

    def simulate(self, roi: Roi, *, extends_roi: bool=True, detect_shadows: bool=True, extends_roi_n_grid: int=10):
        """
        Inputs:
            roi: region of interest, defined according to the proj_model
            extends_roi (optional): extends the roi internally due to SAR geometric distortion
            detect_shadows (optional): sets to 0 values that are not visible by the sensor due to elevation
            extends_roi_n_grid (optional, int): number of points to define the grid used to extend the ROI
        Outputs:
            array: simulated image of the shape of the roi
        """
        cdef int _detect_shadows = detect_shadows
        assert roi.col == int(roi.col)
        assert roi.row == int(roi.row)
        assert roi.w == int(roi.w)
        assert roi.h == int(roi.h)

        cdef int x0, y0, w, h
        x0, y0, w, h = roi.to_roi()

        # NOTE: rangeSpacing and azimuthSpacing are defined by the IPF
        # here, we use a different definition for the azimuthSpacing simply because it is more convenient
        cdef double azimuthSpacing = self._get_sar_resolutions(x0 + w // 2, y0 + h // 2)[1]
        cdef double rangeSpacing
        if isinstance(self.coordinate, GRDCoordinate):
            rangeSpacing = self.coordinate.range_pixel_spacing
        elif isinstance(self.coordinate, SLCCoordinate):
            rangeSpacing = const.LIGHT_SPEED_M_PER_SEC / (2 * self.coordinate.range_frequency)
        else:
            assert False

        # TODO: make sure this definition of aBeta is ok
        cdef double aBeta = azimuthSpacing * rangeSpacing

        outer_roi = roi.copy()
        if extends_roi:
            self._adjust_outer_roi(outer_roi, extends_roi_n_grid)
        # add a bit of y margins for the bilinear splatting
        outer_roi.row -= 2
        outer_roi.h += 3
        dem_, dem_transform = self._resample_dem(outer_roi)
        cdef np.ndarray[np.float32_t, ndim=2] dem = dem_

        cdef int Nrows = dem.shape[0] - 1
        cdef int Ncols = dem.shape[1] - 1
        cdef np.ndarray[np.int64_t, ndim=1] rrow = np.arange(0, Nrows+1)
        cdef np.ndarray[np.int64_t, ndim=1] rcol = np.arange(0, Ncols+1)

        cdef double gamma0
        cdef double elevation

        cdef np.ndarray[np.float64_t, ndim=2] gamma0ReferenceArea = np.zeros([h, w], dtype=np.float64)
        cdef double [:,:] gamma0ReferenceArea_view = gamma0ReferenceArea
        # posData
        cdef double sensorPos_x, sensorPos_y, sensorPos_z
        cdef double earthPoint_x, earthPoint_y, earthPoint_z
        cdef double t00_x, t00_y, t00_z
        cdef double t01_x, t01_y, t01_z
        cdef double t10_x, t10_y, t10_z
        cdef double t11_x, t11_y, t11_z
        cdef double sld_x, sld_y, sld_z, norm
        # altitude
        cdef double maxElevAngle

        cdef np.ndarray[np.float64_t, ndim=1] row0_lons = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_lats = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_alts = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] row0_gx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_gy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_gz = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gz = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] row0_sx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_sy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_sz = np.empty((Ncols+1,))

        cdef np.ndarray[np.int8_t, ndim=1] row0_successes = np.empty((Ncols+1,), dtype=np.int8)
        cdef np.ndarray[np.float64_t, ndim=1] row0_rangeIndices = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_azimuthIndices = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] col0_lons = np.empty((Nrows+1,))
        cdef np.ndarray[np.float64_t, ndim=1] col0_lats = np.empty((Nrows+1,))
        cdef np.ndarray[np.float64_t, ndim=1] col0_alts = np.empty((Nrows+1,))
        cdef int mid = dem.shape[1] // 2
        col0_alts = dem[rrow, mid].astype(np.float64)
        col0_lons, col0_lats = dem_transform * (mid, rrow)
        row0, col0, _ = self.proj_model.projection(col0_lons, col0_lats, col0_alts)
        cdef np.ndarray[np.float64_t, ndim=1] azt0 = self.coordinate.to_azt(row0)

        cdef np.ndarray[np.float64_t, ndim=2] sats = self.proj_model.orbit.evaluate(azt0)

        cdef int i, j
        cdef bint ok
        with nogil:
            for i in range(0, Nrows):
                maxElevAngle = 0.0

                # move previous row to row1_* and fill row0 with new data
                with gil:
                    row1_gx, row0_gx = row0_gx, row1_gx
                    row1_gy, row0_gy = row0_gy, row1_gy
                    row1_gz, row0_gz = row0_gz, row1_gz

                    # NOTE: if the DEM is nan, computations will be nan and `successes[j]` will be false
                    row0_alts = dem[i].astype(np.float64)
                    row0_lons, row0_lats = dem_transform * (rcol, i)

                for j in range(Ncols+1):  # geo2xyzWGS84 is much faster than pyproj
                    row0_gx[j], row0_gy[j], row0_gz[j] = geo2xyzWGS84(row0_lats[j], row0_lons[j], row0_alts[j])

                with gil:
                    ok = self._compute_ground_coordinates(x0, y0, w, h, row0_gx, row0_gy, row0_gz, azt0[i], sats[i],
                                    row0_successes, row0_rangeIndices, row0_azimuthIndices)
                if not ok:  # fast path in case the azimuth is outside [y0, y0+h]
                    continue

                sensorPos_x = sats[i,0]
                sensorPos_y = sats[i,1]
                sensorPos_z = sats[i,2]

                for j in range(Ncols-1):
                    if not row0_successes[j]:
                        continue

                    earthPoint_x = row0_gx[j]
                    earthPoint_y = row0_gy[j]
                    earthPoint_z = row0_gz[j]

                    # Prepare computeIlluminatedArea: fetch positions of t00, t01, t10 and t11
                    # TODO: is row1 OK? or do we want row-1 (next row)?
                    # probably not a big deal for low res DEM, but could have an impact for high res DEM
                    t00_x, t00_y, t00_z = row0_gx[j],   row0_gy[j],   row0_gz[j]
                    t01_x, t01_y, t01_z = row1_gx[j],   row1_gy[j],   row1_gz[j]
                    t10_x, t10_y, t10_z = row0_gx[j+1], row0_gy[j+1], row0_gz[j+1]
                    t11_x, t11_y, t11_z = row1_gx[j+1], row1_gy[j+1], row1_gz[j+1]

                    # Prepare computeIlluminatedArea: compute slant range direction
                    sld_x = sensorPos_x - earthPoint_x
                    sld_y = sensorPos_y - earthPoint_y
                    sld_z = sensorPos_z - earthPoint_z
                    norm = sqrt(sld_x*sld_x + sld_y*sld_y + sld_z*sld_z)
                    sld_x /= norm
                    sld_y /= norm
                    sld_z /= norm

                    gamma0 = computeIlluminatedArea(t00_x, t00_y, t00_z,
                                                    t01_x, t01_y, t01_z,
                                                    t10_x, t10_y, t10_z,
                                                    t11_x, t11_y, t11_z,
                                                    sld_x, sld_y, sld_z)

                    if _detect_shadows:
                        elevation = computeElevationAngle(earthPoint_x, earthPoint_y, earthPoint_z,
                                                          sensorPos_x, sensorPos_y, sensorPos_z)
                    else:
                        elevation = 0

                    if elevation >= maxElevAngle:
                        maxElevAngle = elevation
                        saveIlluminationArea(x0, y0, w, h, row0_azimuthIndices[j], row0_rangeIndices[j],
                                             gamma0, gamma0ReferenceArea_view)

        gamma0ReferenceArea /= aBeta # Normalize (note: original code did it in outputSimulatedArea)
        return gamma0ReferenceArea

    def _compute_ground_coordinates(self, int x0, int y0, int w, int h,
                                   double[:] x,
                                   double[:] y,
                                   double[:] z,
                                   double azt0,
                                   np.ndarray[np.float64_t, ndim=1] sat,
                                   signed char[:] out_success,
                                   double[:] out_col,
                                   double[:] out_row
                                   ):
        """
        Project a set of ground points to col/row coordinates.
        Each point also has a 'success' flag that is False if the point falls outside the ROI.
        The azimuth is assumed constant in this function to speed up projections.
        The function returns False if projected azimuth time falls outside the ROI.

        Inputs:
            x0/y0/w/h    Region of interest
            x/y/z        Coordinates of the ground in 4978
            azt0         Azimuth time of the current line
            sat          Position of the satellite in 4978
            out_col      x coordinate on the ROI
            out_row      y coordinate on the ROI
        Outputs:
            out_success      Whether the operation succeeed (projection is in the ROI)
        """
        cdef double satx = sat[0]
        cdef double saty = sat[1]
        cdef double satz = sat[2]

        # if the row is not correct, stop early
        cdef double row0 = self.coordinate.to_row(azt0)
        # one pixel margin for the bilinear splatting
        if row0 < y0 - 1 or row0 >= y0 + h:
            return False

        out_row[:] = row0

        cdef long i
        cdef np.ndarray[np.float64_t, ndim=1] rng = np.empty_like(x)
        for i in range(x.size):
            rng[i] = sqrt((satx - x[i]) ** 2 + (saty - y[i]) ** 2 + (satz - z[i]) ** 2)

        cdef np.ndarray[np.float64_t, ndim=1] col = self.coordinate.to_col(rng, azt=azt0)
        for i in range(x.size):
            out_col[i] = col[i]

        for i in range(x.size):
            out_success[i] = (out_col[i] >= x0 - 1) & (out_col[i] <= x0 + w)

        return True

    def _adjust_outer_roi(self, roi: Roi, int N):
        """
            Compute the outer ROI. Each 3D point from the outer roi should be projected inside the inner ROI.
        """
        x0, y0, w, h = roi.to_roi()
        xx = np.linspace(x0, x0 + w, N)
        yy = np.linspace(y0, y0 + h, N)
        cols, rows = np.meshgrid(xx, yy)
        cols = cols.ravel()
        rows = rows.ravel()

        lon, lat, alt, masks = self.proj_model.localize_without_alt(rows, cols, dem=self.dem)
        assert np.all(masks), "Error during ROI projection"
        alt = self.dem.elevation(lon, lat)
        _, cols2, _ = self.proj_model.projection(lon, lat, alt)

        # over-estimate a bit
        max_offset = int(ceil(np.nanmax(np.abs(cols - cols2)))) + 10

        roi.col -= max_offset
        roi.w += max_offset * 2
        
   
        
### 
# MODIF: Simulator Arthur Hauck 2025-03-13 
###       
class MySARSimulator_small_roi(SARSimulator_small_roi):
    """
    Warnings:
        1. Because of assumptions for optimizations (warping the DEM),
        the simulation performs poorly on large ROIs.
        You should split the processing in sub-ROI for such cases.

        2. The last line of the simulation tends to not be accurate. This is a bug.
    """

    coordinate: Union[SLCCoordinate, GRDCoordinate]

    def simulate_with_resampled_dem(self, roi: Roi, resampled_dem: eos.dem.DEM, *, detect_shadows: bool=True):
        """
        Inputs:
            roi: region of interest, defined according to the proj_model
            resampled_dem: resampled DEM as eos.dem.DEM object
            detect_shadows (optional): sets to 0 values that are not visible by the sensor due to elevation
        Outputs:
            array: simulated image of the shape of the roi
        """
        cdef int _detect_shadows = detect_shadows
        assert roi.col == int(roi.col)
        assert roi.row == int(roi.row)
        assert roi.w == int(roi.w)
        assert roi.h == int(roi.h)

        cdef int x0, y0, w, h
        x0, y0, w, h = roi.to_roi()

        # NOTE: rangeSpacing and azimuthSpacing are defined by the IPF
        # here, we use a different definition for the azimuthSpacing simply because it is more convenient
        cdef double azimuthSpacing = self._get_sar_resolutions(x0 + w // 2, y0 + h // 2)[1]
        cdef double rangeSpacing
        if isinstance(self.coordinate, GRDCoordinate):
            rangeSpacing = self.coordinate.range_pixel_spacing
        elif isinstance(self.coordinate, SLCCoordinate):
            rangeSpacing = const.LIGHT_SPEED_M_PER_SEC / (2 * self.coordinate.range_frequency)
        else:
            assert False

        # TODO: make sure this definition of aBeta is ok
        cdef double aBeta = azimuthSpacing * rangeSpacing

        dem_, dem_transform = resampled_dem.array, resampled_dem.transform
        cdef np.ndarray[np.float32_t, ndim=2] dem = dem_

        cdef int Nrows = dem.shape[0] - 1
        cdef int Ncols = dem.shape[1] - 1
        cdef np.ndarray[np.int64_t, ndim=1] rrow = np.arange(0, Nrows+1)
        cdef np.ndarray[np.int64_t, ndim=1] rcol = np.arange(0, Ncols+1)

        cdef double gamma0
        cdef double elevation

        cdef np.ndarray[np.float64_t, ndim=2] gamma0ReferenceArea = np.zeros([h, w], dtype=np.float64)
        cdef double [:,:] gamma0ReferenceArea_view = gamma0ReferenceArea
        # posData
        cdef double sensorPos_x, sensorPos_y, sensorPos_z
        cdef double earthPoint_x, earthPoint_y, earthPoint_z
        cdef double t00_x, t00_y, t00_z
        cdef double t01_x, t01_y, t01_z
        cdef double t10_x, t10_y, t10_z
        cdef double t11_x, t11_y, t11_z
        cdef double sld_x, sld_y, sld_z, norm
        # altitude
        cdef double maxElevAngle

        cdef np.ndarray[np.float64_t, ndim=1] row0_lons = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_lats = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_alts = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] row0_gx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_gy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_gz = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row1_gz = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] row0_sx = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_sy = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_sz = np.empty((Ncols+1,))

        cdef np.ndarray[np.int8_t, ndim=1] row0_successes = np.empty((Ncols+1,), dtype=np.int8)
        cdef np.ndarray[np.float64_t, ndim=1] row0_rangeIndices = np.empty((Ncols+1,))
        cdef np.ndarray[np.float64_t, ndim=1] row0_azimuthIndices = np.empty((Ncols+1,))

        cdef np.ndarray[np.float64_t, ndim=1] col0_lons = np.empty((Nrows+1,))
        cdef np.ndarray[np.float64_t, ndim=1] col0_lats = np.empty((Nrows+1,))
        cdef np.ndarray[np.float64_t, ndim=1] col0_alts = np.empty((Nrows+1,))
        cdef int mid = dem.shape[1] // 2
        col0_alts = dem[rrow, mid].astype(np.float64)
        col0_lons, col0_lats = dem_transform * (mid, rrow)
        row0, col0, _ = self.proj_model.projection(col0_lons, col0_lats, col0_alts)
        cdef np.ndarray[np.float64_t, ndim=1] azt0 = self.coordinate.to_azt(row0)

        cdef np.ndarray[np.float64_t, ndim=2] sats = self.proj_model.orbit.evaluate(azt0)

        cdef int i, j
        cdef bint ok
        with nogil:
            for i in range(0, Nrows):
                maxElevAngle = 0.0

                # move previous row to row1_* and fill row0 with new data
                with gil:
                    row1_gx, row0_gx = row0_gx, row1_gx
                    row1_gy, row0_gy = row0_gy, row1_gy
                    row1_gz, row0_gz = row0_gz, row1_gz

                    # NOTE: if the DEM is nan, computations will be nan and `successes[j]` will be false
                    row0_alts = dem[i].astype(np.float64)
                    row0_lons, row0_lats = dem_transform * (rcol, i)

                for j in range(Ncols+1):  # geo2xyzWGS84 is much faster than pyproj
                    row0_gx[j], row0_gy[j], row0_gz[j] = geo2xyzWGS84(row0_lats[j], row0_lons[j], row0_alts[j])

                with gil:
                    ok = self._compute_ground_coordinates(x0, y0, w, h, row0_gx, row0_gy, row0_gz, azt0[i], sats[i],
                                    row0_successes, row0_rangeIndices, row0_azimuthIndices)
                if not ok:  # fast path in case the azimuth is outside [y0, y0+h]
                    continue

                sensorPos_x = sats[i,0]
                sensorPos_y = sats[i,1]
                sensorPos_z = sats[i,2]

                for j in range(Ncols-1):
                    if not row0_successes[j]:
                        continue

                    earthPoint_x = row0_gx[j]
                    earthPoint_y = row0_gy[j]
                    earthPoint_z = row0_gz[j]

                    # Prepare computeIlluminatedArea: fetch positions of t00, t01, t10 and t11
                    # TODO: is row1 OK? or do we want row-1 (next row)?
                    # probably not a big deal for low res DEM, but could have an impact for high res DEM
                    t00_x, t00_y, t00_z = row0_gx[j],   row0_gy[j],   row0_gz[j]
                    t01_x, t01_y, t01_z = row1_gx[j],   row1_gy[j],   row1_gz[j]
                    t10_x, t10_y, t10_z = row0_gx[j+1], row0_gy[j+1], row0_gz[j+1]
                    t11_x, t11_y, t11_z = row1_gx[j+1], row1_gy[j+1], row1_gz[j+1]

                    # Prepare computeIlluminatedArea: compute slant range direction
                    sld_x = sensorPos_x - earthPoint_x
                    sld_y = sensorPos_y - earthPoint_y
                    sld_z = sensorPos_z - earthPoint_z
                    norm = sqrt(sld_x*sld_x + sld_y*sld_y + sld_z*sld_z)
                    sld_x /= norm
                    sld_y /= norm
                    sld_z /= norm

                    gamma0 = computeIlluminatedArea(t00_x, t00_y, t00_z,
                                                    t01_x, t01_y, t01_z,
                                                    t10_x, t10_y, t10_z,
                                                    t11_x, t11_y, t11_z,
                                                    sld_x, sld_y, sld_z)

                    if _detect_shadows:
                        elevation = computeElevationAngle(earthPoint_x, earthPoint_y, earthPoint_z,
                                                          sensorPos_x, sensorPos_y, sensorPos_z)
                    else:
                        elevation = 0

                    if elevation >= maxElevAngle:
                        maxElevAngle = elevation
                        saveIlluminationArea(x0, y0, w, h, row0_azimuthIndices[j], row0_rangeIndices[j],
                                             gamma0, gamma0ReferenceArea_view)

        gamma0ReferenceArea /= aBeta # Normalize (note: original code did it in outputSimulatedArea)
        return gamma0ReferenceArea
