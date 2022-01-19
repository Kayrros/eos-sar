#!python
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

import time
import numpy as np
cimport numpy as np
from libc.math cimport M_PI, sin, cos, tan, sqrt, isnan, fmin, ceil, round, floor, atan, acos
import rasterio
import pyproj

import eos.sar
from eos.sar import io
from eos.sar import const
from eos.sar.roi import Roi
from eos.products import sentinel1
import eos.dem

transformer_from_4326_to_4978 = pyproj.Transformer.from_crs('epsg:4979', 'epsg:4978', always_xy=True)
transformer_from_4978_to_4326 = pyproj.Transformer.from_crs('epsg:4978', 'epsg:4979', always_xy=True)

from affine import Affine
import rasterio.crs
import rasterio.transform
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.control import GroundControlPoint

"""
This code is a port of TerrainFlatteningOp.java from the sentinel1 toolbox.
The code is thus following the corresponding license:
 * Copyright (C) 2014 by Array Systems Computing Inc. http://www.array.ca
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/

Assumptions compared to original code:
isGDR=false
isPolSAR=false
srgrConvParams=null
no need for sigma0
"""

# from GeoUtils (snap-engine)
"""
public interface WGS84 {
        double a = 6378137.0; // m
        double b = 6356752.3142451794975639665996337; //6356752.31424518; // m
        double earthFlatCoef = 1.0 / ((a - b) / a); //298.257223563;
        double e2 = 2.0 / earthFlatCoef - 1.0 / (earthFlatCoef * earthFlatCoef);
        double e2inv = 1 - WGS84.e2;
        double ep2 = e2 / (1 - e2);
    }

    public interface GRS80 {
        double a = 6378137; // m
        double b = 6356752.314140; // m
        double earthFlatCoef = 1.0 / ((a - b) / a); //298.257222101;
        double e2 = 2.0 / earthFlatCoef - 1.0 / (earthFlatCoef * earthFlatCoef);
        double ep2 = e2 / (1 - e2);
    }
"""

cdef inline (double, double, double) geo2xyzWGS84(double latitude, double longitude, double altitude) nogil:
    """"
    Convert geodetic coordinate into cartesian XYZ coordinate with specified geodetic system (WGS84)
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

assert geo2xyzWGS84(39.361230359352405, 9.514164698791625, 1843.6623015093382) \
        == transformer_from_4326_to_4978.transform(9.514164698791625, 39.361230359352405, 1843.6623015093382)

cdef inline (double, double, double) xyz2geoWGS84(double x, double y, double z) nogil:
    cdef double WGS84_a = 6378137.0
    cdef double WGS84_b = 6356752.3142451794975639665996337
    cdef double WGS84_earthFlatCoef = 1.0 / ((WGS84_a - WGS84_b) / WGS84_a)
    cdef double WGS84_e2 = 2.0 / WGS84_earthFlatCoef - 1.0 / (WGS84_earthFlatCoef * WGS84_earthFlatCoef)
    cdef double WGS84_ep2 = WGS84_e2 / (1 - WGS84_e2)
    cdef double s = sqrt(x * x + y * y)
    cdef double theta = atan(z * WGS84_a / (s * WGS84_b))
    cdef double lat = atan((z + WGS84_ep2 * WGS84_b * sin(theta)**3) /
                           (s - WGS84_e2 * WGS84_a * cos(theta)**3)) * (180. / M_PI)
    cdef double lon = atan(y / x) * (180. / M_PI)
    cdef double sinLat = sin(lat * (M_PI / 180.))
    cdef double N = WGS84_a / sqrt(1.0 - WGS84_e2 * sinLat * sinLat)
    cdef double alt = WGS84_e2 * N - N + z/sinLat

    if lon < 0.0 and y >= 0.0:
        lon += 180.0
    elif lon > 0.0 and y < 0.0:
        lon -= 180.0

    return lat, lon, alt

cdef inline double computeElevationAngle(double earthPoint_x, double earthPoint_y, double earthPoint_z,
                                         double sensorPos_x, double sensorPos_y, double sensorPos_z):
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
    cdef double slantRange = sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff)
    cdef double H2 = sensorPos_x * sensorPos_x + sensorPos_y * sensorPos_y + sensorPos_z * sensorPos_z
    cdef double R2 = earthPoint_x * earthPoint_x + earthPoint_y * earthPoint_y + earthPoint_z * earthPoint_z
    cdef double angle = acos((slantRange * slantRange + H2 - R2) / (2 * slantRange * sqrt(H2))) * (180.0 / M_PI)
    return angle

cdef inline double computeIlluminatedArea(double t00_x, double t00_y, double t00_z,
                                          double t01_x, double t01_y, double t01_z,
                                          double t10_x, double t10_y, double t10_z,
                                          double t11_x, double t11_y, double t11_z,
                                          double sld_x, double sld_y, double sld_z):
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
                         double[:,:] gamma0ReferenceArea_view):
    """
    Distribute the local illumination area to the 4 adjacent pixels using bi-linear distribution.
    Inputs:
         x0                  The x coordinate of the pixel at the upper left corner of current tile.
         y0                  The y coordinate of the pixel at the upper left corner of current tile.
         w                   The tile width (= gamma0ReferenceArea_view.shape[1])
         h                   The tile height (= gamma0ReferenceArea_view.shape[0])
         gamma0Area          The illuminated area.
         azimuthIndex        Azimuth pixel index for the illuminated area.
         rangeIndex          Range pixel index for the illuminated area.
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


class TerrainFlatteningOp:

    def __init__(self, proj_model: eos.sar.model.SensorModel,
                 dem_source: eos.dem.DEMSource=None,
                 oversamplingMultiple=.5,
                 detectShadow=True):
        self.proj_model = proj_model

        if not dem_source:
            dem_source = eos.dem.get_any_source()
        self.dem_source = dem_source

        self.overSamplingFactor = oversamplingMultiple
        self.detectShadow = detectShadow

    def get_transform(self, proj_model, roi):
        x0, y0, w, h = roi.to_roi()
        row = np.asarray([y0, y0, y0+h, y0+h])
        col = np.asarray([x0, x0+w, x0+w, x0])
        lon, lat, _ = proj_model.localization(row, col, alt=np.zeros_like(col))

        gcps = [GroundControlPoint(row=r-y0, col=c-x0, x=ln, y=lt) for r, c, ln, lt in zip(row, col, lon, lat)]
        dst_transform = rasterio.transform.from_gcps(gcps)

        return dst_transform

    def get_sar_resolutions(self, int col, int row):
        # compute the azimuth and range 'resolution' by localizing two points in each direction
        # and measure the distance in xyz 4978 Coordinates
        # it's not really a resolution, but more a sampling rate
        rows = [row, row,   row+1]
        cols = [col, col+1, col]
        p = np.asarray(self.proj_model.localization(rows, cols, np.zeros_like(rows), crs='epsg:4978'))
        p = (p - p[:,0:1])[:,1:]
        range_resolution, azimuth_resolution = np.linalg.norm(p, axis=0)
        return range_resolution, azimuth_resolution

    def resample_dem(self, int x0, int y0, int w, int h):
        # NOTE: an affine transform is a poor approximation?
        # we could warp with a higher order model, and use it to get the lon,lat as well

        roi = Roi(x0, y0, w, h)
        tr = self.get_transform(self.proj_model, roi)

        # fetch the resolution of the farthest pixel (finest resolution)
        midrow = y0 + h // 2
        midcol = x0 + w
        range_resolution, azimuth_resolution = self.get_sar_resolutions(midcol, midrow)

        # TODO: 30 is from SRTM30, use the resolution from the DEM and convert it
        # NOTE: on a typical S1 SLC IW2 burst and SRTM30, Fx,Fy = 4,2
        Fx = int(ceil((30 / range_resolution) * self.overSamplingFactor))
        Fy = int(ceil((30 / azimuth_resolution) * self.overSamplingFactor))
        print('oversampling:', Fx, Fy)
        tr = tr * Affine.scale(1 / Fx, 1 / Fy)
        nw = w * Fx
        nh = h * Fy
        print(nw, nh, w, h)
        # TODO: add 0.5 pixel to the transform for pixel is point?

        # TODO: directly reproject to 4978?
        # then the image generation would be without crs conversion

        bounds = []

        print('download dem'); t = time.time()
        geometry, _, _ = self.proj_model.get_approx_geom(roi)
        lons, lats = zip(*geometry)
        bounds = (min(lons), min(lats), max(lons), max(lats))
        src_raster, src_transform, crs = self.dem_source.crop(bounds)
        print('download dem', time.time() - t)

        dem = np.zeros((nh, nw))
        reproject(
            src_raster,
            dem,
            src_transform=src_transform,
            src_crs=crs,
            dst_transform=tr,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.cubic)

        return dem, tr

    def generateSimulatedImage(self, int x0, int y0, int w, int h, *, extends_roi=True):
        """
        Inputs:
            x0/y0         top left coordinate of the rectangle region to compute
            w/h           width and height of the rectangle region to compute
            tileOverlap*  overlap percentages for the borders
        Outputs:
            gamma0ReferenceArea  2D double numpy array containing gamma0
        """
        cdef int detectShadow = self.detectShadow
        cdef int i, j

        # NOTE: rangeSpacing and azimuthSpacing are defined by the IPF
        # here, we use a different definition for the azimuthSpacing simply because it is more convenient
        cdef double azimuthSpacing = self.get_sar_resolutions(x0 + w // 2, y0 + h // 2)[1]
        cdef double rangeSpacing = const.LIGHT_SPEED_M_PER_SEC / (2 * self.proj_model.range_frequency)
        cdef double aBeta = azimuthSpacing * rangeSpacing

        print('load the dem')
        cdef int x0_out, y0_out, w_out, h_out
        if extends_roi:
            x0_out, y0_out, w_out, h_out = self.get_outer_roi(x0, y0, w, h)
        else:
            x0_out, y0_out, w_out, h_out = x0, y0, w, h
        # add a bit of y margins for the bilinear splatting
        dem_, dem_transform = self.resample_dem(x0_out, y0_out - 2, w_out, h_out + 3)
        cdef np.ndarray[np.float64_t, ndim=2] dem = dem_
        print('load the dem', 'ok')

        cdef int Nrows = dem.shape[0] - 1
        cdef int Ncols = dem.shape[1] - 1
        cdef np.ndarray[np.int64_t, ndim=1] rlat = np.arange(0, Nrows+1)
        cdef np.ndarray[np.int64_t, ndim=1] rlon = np.arange(0, Ncols+1)
        print('Ncols, Nrows:', Ncols, Nrows)

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
        col0_alts = dem[rlat, mid]
        col0_lons, col0_lats = dem_transform * (mid, rlat)
        row0, col0, _ = self.proj_model.projection(col0_lons, col0_lats, col0_alts)
        azt0, _ = self.proj_model.to_azt_rng(row0, col0)

        cdef np.ndarray[np.float64_t, ndim=2] sats = self.proj_model.orbit.evaluate(azt0)

        cdef int fast = 1

        cdef double t = time.time()
        for i in range(0, Nrows):
            maxElevAngle = 0.0
            if i % 100 == 0:
                print(i // 100, time.time() - t)
                t = time.time()

            # move previous row to row1_* and fill row0 with new data
            row1_gx, row0_gx = row0_gx, row1_gx
            row1_gy, row0_gy = row0_gy, row1_gy
            row1_gz, row0_gz = row0_gz, row1_gz

            # NOTE: if the DEM is nan, computations will be nan and `successes[j]` will be false
            row0_alts = dem[i]
            row0_lons, row0_lats = dem_transform * (rlon, i)
            for j in range(Ncols+1):  # geo2xyzWGS84 is much faster than pyproj
                row0_gx[j], row0_gy[j], row0_gz[j] = geo2xyzWGS84(row0_lats[j], row0_lons[j], row0_alts[j])

            if fast:
                ok = self.getPositionConstantAzimuth(x0, y0, w, h, row0_gx, row0_gy, row0_gz, azt0[i], sats[i],
                                row0_successes, row0_rangeIndices, row0_azimuthIndices)
                if not ok:  # fast path in case the azimuth is outside [y0, y0+h]
                    continue
                sensorPos_x = sats[i,0]
                sensorPos_y = sats[i,1]
                sensorPos_z = sats[i,2]
            else:
                row0_gx, row0_gy, row0_gz = transformer_from_4326_to_4978.transform(row0_lons, row0_lats, row0_alts)

                row0_successes.flat[:], row0_rangeIndices, row0_azimuthIndices, row0_sx, row0_sy, row0_sz = \
                        self.getPosition(x0, y0, w, h, row0_gx, row0_gy, row0_gz)

            for j in range(Ncols-1):
                if not row0_successes[j]:
                    continue

                if not fast:
                    sensorPos_x = row0_sx[j]
                    sensorPos_y = row0_sy[j]
                    sensorPos_z = row0_sz[j]

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
                # assert not isnan(gamma0)

                # because of the dem resampling to az/rg coordinates, this traversal order shouldn't be required
                if detectShadow:
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

    def computeNormalizedImage(self, np.ndarray[np.float64_t, ndim=2] gamma0ReferenceArea,
                               np.ndarray[np.float64_t, ndim=2] image):
        """
        Replacement for outputNormalizedImage
        The difference with outputNormalizedImage is that we already assume gamma0 was normalized by aBeta.
        Inputs:
            gamma0ReferenceArea: Image computed from generateSimulatedImage
            image:               Image to normalize
        Outputs:
            normalized_image, a normalized image
        """
        cdef double threshold = 0.05
        # TODO: we need incidenceAngleTPG for threshold / tan(incidenceAngleTPG(x, y) * dtor)

    def computeImageGeoBoundary(self, int xmin, int xmax, int ymin, int ymax):
        """
        Port of function computeImageGeoBoundary
        Inputs:
            xmin/xmax/ymin/ymax: coordinates of an image rectangular region
        Outputs:
            latMin minimum latitude
            latMax maximum latitude
            lonMin minimum longitude
            lonMax maximum longitude
        """
        cdef double latMin, latMax, lonMin, lonMax

        # Note: original code source gives the names 'FirstNear', 'FirstFar',
        # 'LastNear' and 'LastFar' to the points
        lat1, lon1 = self.getGeoPos(xmin, ymin)
        lat2, lon2 = self.getGeoPos(xmax, ymin)
        lat3, lon3 = self.getGeoPos(xmin, ymax)
        lat4, lon4 = self.getGeoPos(xmax, ymax)
        latMin = min([90.0, lat1, lat2, lat3, lat4])
        latMax = max([-90.0, lat1, lat2, lat3, lat4])
        lonMin = min([180.0, lon1, lon2, lon3, lon4])
        lonMax = max([-180.0, lon1, lon2, lon3, lon4])

        return latMin, latMax, lonMin, lonMax

    def getPosition(self, int x0, int y0, int w, int h, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] z):
        """
        Inputs:
            x0/y0/w/h    Bounds of the image
            x/y/z        Coordinates of the ground in 4978
        Outputs:
            success      Whether the operation succeeed (projection is in the image)
            col          x coordinate on the image (range)
            row          y coordinate on the image (azimuth)
            sensorPos    x/y/z coordinate of the satellite in 4978
        """
        cdef np.ndarray[np.float64_t, ndim=1] row, col, satx, saty, satz

        row, col, _ = self.proj_model.projection(x, y, z, crs='epsg:4978')
        success = (row >= y0 - 1) * (row <= y0 + h) * (col >= x0 - 1) * (col <= x0 + w)

        azt, rng = self.proj_model.to_azt_rng(row, col)
        sat = self.proj_model.orbit.evaluate(azt)

        satx = sat[:, 0]
        saty = sat[:, 1]
        satz = sat[:, 2]
        return success, col, row, satx, saty, satz

    def getPositionConstantAzimuth(self, int x0, int y0, int w, int h,
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
        Assumes that the azimuth is constant to speed up projections.
        Assumes that the conversion azt/rng to col/row is without correction.
        Inputs:
            x0/y0/w/h    Bounds of the image
            x/y/z        Coordinates of the ground in 4978
        Outputs:
            success      Whether the operation succeeed (projection is in the image)
            col          x coordinate on the image (range)
            row          y coordinate on the image (azimuth)
            sensorPos    x/y/z coordinate of the satellite in 4978
        """
        cdef double satx = sat[0]
        cdef double saty = sat[1]
        cdef double satz = sat[2]
        cdef double c = const.LIGHT_SPEED_M_PER_SEC
        cdef double range_frequency = self.proj_model.range_frequency
        cdef double first_col_time = self.proj_model.first_col_time
        cdef double first_row_time = self.proj_model.first_row_time
        cdef double azimuth_frequency = self.proj_model.azimuth_frequency

        # if the row is not correct, stop early
        cdef double row0 = (azt0 - first_row_time) * azimuth_frequency
        # one pixel margin for the bilinear splatting
        if row0 < y0 - 1 or row0 >= y0 + h:
            return False

        out_row[:] = row0

        cdef long i
        cdef np.ndarray[np.float64_t, ndim=1] rng = np.empty_like(x)
        for i in range(x.size):
            rng[i] = sqrt((satx - x[i]) ** 2 + (saty - y[i]) ** 2 + (satz - z[i]) ** 2)

        # TODO: find a better way to copy to out_col
        cdef np.ndarray[np.float64_t, ndim=1] col = self.proj_model.to_col(rng)
        for i in range(x.size):
            out_col[i] = col[i]

        for i in range(x.size):
            out_success[i] = (out_col[i] >= x0 - 1) & (out_col[i] <= x0 + w)

        return True

    def get_outer_roi(self, int x0, int y0, int w, int h, int N=10):
        """
        Compute the outer ROI. Each 3D point from the outer roi should be projected inside the inner ROI.
        """
        xx = np.linspace(x0, x0 + w, N)
        yy = np.linspace(y0, y0 + h, N)
        cols, rows = np.meshgrid(xx, yy)
        cols = cols.ravel()
        rows = rows.ravel()

        lon, lat, alt = self.proj_model.localization(rows, cols, alt=np.zeros_like(rows))
        alt = self.dem_source.elevation(lon, lat)
        rows2, cols2, _ = self.proj_model.projection(lon, lat, alt)

        # offset due to negative altitude
        min_range_offset = max(0, int(ceil(- (cols - cols2).min())))
        # offset due to positive altitude
        max_range_offset = max(0, int(ceil((cols - cols2).max())))
        print('range offset for outer roi:', min_range_offset, max_range_offset)

        x0 -= min_range_offset
        w += min_range_offset + max_range_offset
        return x0, y0, w, h

