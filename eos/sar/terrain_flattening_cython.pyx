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

import numpy as np
cimport numpy as np
from libc.math cimport M_PI, sin, cos, tan, sqrt, isnan, fmin, ceil, round, floor, atan, acos
from eos.products import sentinel1
from eos.sar import io
import rasterio
import xmltodict
import pyproj
from srtm4 import srtm4

cdef extern from "<math.h>" nogil:
    long double sqrtl(long double x)

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

#orbitOnWest: false if antenna pointing right and descending or ascending and pointing left, else true. Sentinel 1 is pointing right.
#nearRangeOnLeft: true

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
# Note the original code also uses the class Geo2xyzWGS84, which enables to precompute
# the parts depending on lat inside a loop on lon (it is exactly the same computation)
# But here as we use an inline function, this optimization will be done automatically
cdef inline geo2xyzWGS84(double latitude, double longitude, double altitude):
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
    cdef double N = (WGS84_a / sqrt(1.0 - WGS84_e2 * sinLat * sinLat))
    cdef double NcosLat = (N + altitude) * cos(lat)

    cdef double x, y, z

    x = NcosLat * cos(lon) # in m
    y = NcosLat * sin(lon) # in m
    z = (N + altitude - WGS84_e2 * N) * sinLat
    return x, y, z


cdef inline xyz2geoWGS84(double x, double y, double z):
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
    cdef double N = (WGS84_a / sqrt(1.0 - WGS84_e2 * sinLat * sinLat))
    cdef double alt = WGS84_e2 * N - N + z/sinLat

    if (lon < 0.0 and y >= 0.0):
        lon += 180.0
    elif (lon > 0.0 and y < 0.0):
        lon -= 180.0

    return lat, lon, alt

class TerrainFlatteningOp:
    def __init__(self, xml_content, dem_path, tiff_path,
               additionalOverlap = 0.1,
               oversamplingMultiple=1.,
               detectShadow=True):
        i = xmltodict.parse(xml_content)['product']
        d = i['imageAnnotation']['imageInformation']
        rangeSpacing = float(d['rangePixelSpacing'])
        azimuthSpacing = float(d['azimuthPixelSpacing'])

        d = i['generalAnnotation']['productInformation']
        self.orbitOnWest = d['pass'] != "Descending" # TODO: it should be != but it gives wrong results
        print(self.orbitOnWest)

        self.overSamplingFactor = oversamplingMultiple
        minSpacing = fmin(rangeSpacing, azimuthSpacing)
        # TODO: implement overSamplingFactor for target DEM. For example:
        if minSpacing < 30.: # SRTM30
            self.overSamplingFactor *= ceil(30.0 / minSpacing)
        self.aBeta = azimuthSpacing * rangeSpacing
        self.detectShadow = detectShadow
        self.additionalOverlap = additionalOverlap

        burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, 5)
        self.bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta)

        self.db = rasterio.open(dem_path, 'r')
        self.db_raster = self.db.read(1)
        self.demResolution = min(self.db.res)
        print(self.demResolution)

        image_reader = io.open_image(tiff_path) 
        self.burst_array = io.read_window(image_reader, self.bmod.burst_roi)
        self.sourceImageWidth = self.burst_array.shape[1]
        self.sourceImageHeight = self.burst_array.shape[0]

    def getGeoPos(self, int x, int y):
        """
        I couldn't find the exact implementation of getGeoPos used in the
        original code.
        Returns the latitude and longitude value for a given pixel co-ordinate.
        Inputs:
            x/y      the pixel's co-ordinates given as x,y
        Outputs:
            lat/lon (WSG84)
        """
        # Note: The uses of this functions can probably be removed by reworking the code
        lon, lat, _, masks = self.bmod.localize_without_alt(y, x, elev=srtm4)
        assert not np.any(masks['invalid']), "Potential failure"
        return lat, lon

    def getDemElevation(self, double lat, double lon):
        """
        Inputs:
            lat/lon (WSG84)
        Outputs:
            Elevation (in m)
        """
        row, col = self.db.index(lon, lat)
        if row < 0 or col < 0 or row >= self.db_raster.shape[0] or col >= self.db_raster.shape[1]:
            return float("NaN")
        return self.db_raster[row, col]


    def interpolate_dem(self, double [:,:] height_view, double x, double y):
        """
        Interpolates the dem at (x, y)
        This function stays close to the original code, but could be optimized.
        """
        cdef int h = height_view.shape[0]
        cdef int w = height_view.shape[1]
        cdef int i0, j0, i1, j1, tmp, iMax, jMax
        cdef double di, dj, ki, kj

        # computeCornerBasedIndex
        x += 0.5
        y += 0.5

        # computeIndex
        # Compute indices for the bilinear interpolation (i0, i1, j0, j1) and the coefficients ki, kj
        i0 = int(floor(x)) # Note: the difference with (int)x is the behaviour on negatives
        j0 = int(floor(y))

        di = x - (i0 + 0.5)
        dj = y - (j0 + 0.5)
        iMax = w - 1
        jMax = h - 1

        if di >= 0:
            i1 = i0 + 1
            i0 = max(0, min(iMax, i0))
            i1 = max(0, min(iMax, i1))
            ki = di
        else:
            i1 = i0 - 1
            i0 = max(0, min(iMax, i0))
            i1 = max(0, min(iMax, i1)) 
            tmp = i1
            i1 = i0
            i0 = tmp
            ki = di + 1

        if dj >= 0:
            j1 = j0 + 1
            j0 = max(0, min(jMax, j0))
            j1 = max(0, min(jMax, j1))
            kj = dj
        else:
            j1 = j0 - 1
            j0 = max(0, min(jMax, j0))
            j1 = max(0, min(jMax, j1))
            tmp = j1
            j1 = j0
            j0 = tmp
            kj = dj + 1

        # resample
        return height_view[j0, i0] * (1. - ki) * (1. - kj) + \
            height_view[j0, i1] * ki * (1. - kj) + \
            height_view[j1, i0] * (1. - ki) * kj + \
            height_view[j1, i1] * ki * kj
        

    def generateSimulatedImage(self, int x0, int y0, int w, int h,
                               double tileOverlapUp, double tileOverlapDown, double tileOverlapLeft, double tileOverlapRight):
        """
        Port of function generateSimulatedImage
        Inputs:
            x0/y0         top left coordinate of the rectangle region to compute
            w/h           width and height of the rectangle region to compute
            tileOverlap*  overlap percentages for the borders
        Outputs:
            gamma0ReferenceArea  2D double numpy array containing gamma0
        """
        cdef int ymin = max(y0 - int(h * tileOverlapUp), 0)
        cdef int ymax = min(y0 + h + int(h * tileOverlapDown), self.sourceImageHeight)
        cdef int xmin = max(x0 - int(w * tileOverlapLeft), 0)
        cdef int xmax = min(x0 + w + int(w * tileOverlapRight), self.sourceImageWidth)
        cdef int i, j, jj

        latMin, latMax, lonMin, lonMax = self.computeImageGeoBoundary(xmin, xmax, ymin, ymax)

        # add extralat/extralon margins
        latMin -= 20 * self.demResolution
        latMax += 20 * self.demResolution
        lonMin -= 20 * self.demResolution
        lonMax += 20 * self.demResolution

        cdef int rows = int(round((latMax - latMin) / self.demResolution))
        cdef int cols = int(round((lonMax - lonMin) / self.demResolution))
        cdef np.ndarray[np.float64_t, ndim=2] height = np.empty([rows, cols], dtype=np.float64)
        cdef double [:, :] height_view = height # Using a cython view for performance
        for i in range(rows):
            for j in range(cols):
                height_view[i, j] = self.getDemElevation(latMax - i * self.demResolution, \
                                                         lonMin + j * self.demResolution)
        """
        Resampler logic of the original code:
            final ResamplingRaster resamplingRaster = new ResamplingRaster(demNoDataValue, height)
            final Resampling.Index resamplingIndex = selectedResampling.createIndex()
            . ResamplingRaster defines a wrapper aroung the data array, which defines a function
            to retrieve the values. We simply implement that by accessing directly the array.
            . selectedResampling (bilinear interpolation) is a class that enables to sample with the chosen interpolation of the array. 'computeCornerBasedIndex' is used, which adds (0.5, 0.5) to the coordinates to 'counter the consideration of pixel center in base function'. 'computeCornerBasedIndex' fills resamplingIndex (a vector of two 2D coordinates) with the indices of the corners of the region to sample. Then 'resample' retrieves the 4 samples and performs the averaging.
            For this code we use a simpler abstraction by just using one function to interpolate
        """

        cdef double delta = self.demResolution / self.overSamplingFactor
        cdef double ratio = delta / self.demResolution  # thus ratio = 1. / self.overSamplingFactor
        cdef int nLat = (self.overSamplingFactor * rows)
        cdef int nLon = (self.overSamplingFactor * cols)

        cdef np.ndarray[np.float64_t, ndim=1] azimuthIndex = np.empty([nLon], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] rangeIndex = np.empty([nLon], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] gamma0Area = np.empty([nLon], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] elevationAngle = np.empty([nLon], dtype=np.float64)
        cdef np.ndarray[np.int32_t, ndim=1] savePixel = np.empty([nLon], dtype=np.int32)
        cdef double [:] azimuthIndex_view = azimuthIndex
        cdef double [:] rangeIndex_view = rangeIndex
        cdef double [:] gamma0Area_view = gamma0Area
        cdef double [:] elevationAngle_view = elevationAngle
        cdef int [:] savePixel_view = savePixel
        cdef np.ndarray[np.float64_t, ndim=1] lats = np.empty([nLon], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] lons = np.empty([nLon], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] alts = np.empty([nLon], dtype=np.float64)

        cdef np.ndarray[np.float64_t, ndim=2] gamma0ReferenceArea = np.zeros([h, w], dtype=np.float64)
        cdef double [:,:] gamma0ReferenceArea_view = gamma0ReferenceArea
        cdef double lat, iRatio, lon, jRatio
        # posData
        cdef double sensorPos_x, sensorPos_y, sensorPos_z
        cdef double earthPoint_x, earthPoint_y, earthPoint_z
        cdef double rangeIndex_, azimuthIndex_
        cdef double t00_x, t00_y, t00_z
        cdef double t01_x, t01_y, t01_z
        cdef double t10_x, t10_y, t10_z
        cdef double t11_x, t11_y, t11_z
        cdef double sld_x, sld_y, sld_z, norm
        # altitude
        cdef double alt00, alt01, alt10, alt11
        cdef double maxElevAngle = 0.0

        for i in range(1, nLat):
            lat = latMax - i * delta
            iRatio = i * ratio

            for j in range(nLon):
                lon = lonMin + j * delta
                jRatio = j * ratio

                # Fetch altitude
                alt00 = self.interpolate_dem(height_view, jRatio, iRatio)
                if isnan(alt00):
                    continue

                #success, rangeIndex_, azimuthIndex_, sensorPos_x, sensorPos_y, sensorPos_z = self.getPosition(x0, y0, w, h, earthPoint_x, earthPoint_y, earthPoint_z)
                #success, rangeIndex_, azimuthIndex_, sensorPos_x, sensorPos_y, sensorPos_z = self.getPosition2(x0, y0, w, h, lat, lon, alt00)
                lats[j] = lat
                lons[j] = lon
                alts[j] = alt00
            successes, rangeIndices, azimuthIndices, sensorPos_xs, sensorPos_ys, sensorPos_zs = self.getPosition3(x0, y0, w, h, lats, lons, alts)
            #print(successes, rangeIndices, azimuthIndices, sensorPos_xs, sensorPos_ys, sensorPos_zs)
            
            for j in range(nLon):
                lon = lonMin + j * delta
                jRatio = j * ratio

                success = successes[j]

                #print(i, j, success)
                if not(success):
                    savePixel_view[j] = 0
                    continue

                rangeIndex_ = rangeIndices[j]
                azimuthIndex_ = azimuthIndices[j]
                sensorPos_x = sensorPos_xs[j]
                sensorPos_y = sensorPos_ys[j]
                sensorPos_z = sensorPos_zs[j]
                alt00 = alts[j]
                earthPoint_x, earthPoint_y, earthPoint_z = geo2xyzWGS84(lat, lon, alt00)

                alt01 = self.interpolate_dem(height_view, jRatio, iRatio - ratio)
                alt10 = self.interpolate_dem(height_view, jRatio + ratio, iRatio)
                alt11 = self.interpolate_dem(height_view, jRatio + ratio, iRatio - ratio)

                # final LocalGeometry localGeometry = new LocalGeometry(lat, delta)
                # -> set latitude of t00 and t10 to lat, of t01 and t11 to lat+delta
                # localGeometry.setLon(lon, alt00, alt01, alt10, alt11)
                # -> sets longitude of t00 and t01 to lon, of t10 and t11 to lon+delta.
                # -> sets altitude of t00, t01, t10, t11 to alt00, alt01, alt10, alt11 respectively

                # Prepare computeIlluminatedArea: check if we have dem values for all 4 positions
                if isnan(alt00) or isnan(alt01) or isnan(alt10) or isnan(alt11):
                    print("Skipping because of NaN in DEM", i, j)
                    savePixel_view[j] = 0
                    continue

                # Prepare computeIlluminatedArea: compute positions of t00, t01, t10 and t11
                t00_x, t00_y, t00_z = geo2xyzWGS84(lat, lon, alt00)
                t01_x, t01_y, t01_z = geo2xyzWGS84(lat+delta, lon, alt01)
                t10_x, t10_y, t10_z = geo2xyzWGS84(lat, lon+delta, alt10)
                t11_x, t11_y, t11_z = geo2xyzWGS84(lat+delta, lon+delta, alt11)
                
                # Prepare computeIlluminatedArea: compute slant range direction 
                sld_x = sensorPos_x - earthPoint_x
                sld_y = sensorPos_y - earthPoint_y
                sld_z = sensorPos_z - earthPoint_z
                norm = sqrt(sld_x*sld_x + sld_y*sld_y + sld_z*sld_z)
                sld_x /= norm
                sld_y /= norm
                sld_z /= norm

                gamma0 = self.computeIlluminatedArea(t00_x, t00_y, t00_z,
                               t01_x, t01_y, t01_z,
                               t10_x, t10_y, t10_z,
                               t11_x, t11_y, t11_z,
                               sld_x, sld_y, sld_z)
                #print(i, j, gamma0)
                #print(t00_x, t00_y, t00_z)
                #print(t01_x, t01_y, t01_z)
                #print(t10_x, t10_y, t10_z)
                #print(t11_x, t11_y, t11_z)
                #print(sld_x, sld_y, sld_z)
                gamma0Area_view[j] = gamma0
                #print (alt00, alt01, alt10, alt11)
                assert(not(isnan(gamma0)))

                if self.detectShadow:
                    elevationAngle_view[j] = self.computeElevationAngle(earthPoint_x, earthPoint_y, earthPoint_z, sensorPos_x, sensorPos_y, sensorPos_z)

                rangeIndex_view[j] = rangeIndex_
                azimuthIndex_view[j] = azimuthIndex_
                savePixel_view[j] = 1 if rangeIndex_ > (x0 - 1) and rangeIndex_ < (x0 + w) and azimuthIndex_ > (y0 - 1) and azimuthIndex_ < (y0 + h) else 0

            maxElevAngle = 0.0
            for j in range(nLon):
                if self.orbitOnWest:
                    jj = j
                else:
                    jj = nLon-1 -j
                if savePixel_view[jj] == 1: # Cython doesn't like booleans :-(
                    if self.detectShadow:
                        if elevationAngle_view[jj] < maxElevAngle:
                            continue
                        maxElevAngle = elevationAngle_view[jj]
                    self.saveIlluminationArea(x0, y0, w, h,
                                              azimuthIndex_view[jj], rangeIndex_view[jj],
                                              gamma0Area_view[jj], gamma0ReferenceArea)
        gamma0ReferenceArea /= self.aBeta # Normalize (note: original code did it in outputSimulatedArea)
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
        #TODO: we need incidenceAngleTPG for threshold / tan(incidenceAngleTPG(x, y) * dtor)

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

    def getPosition(self, int x0, int y0, int w, int h, double earthPoint_x, double earthPoint_y, double earthPoint_z):
        """
        Port of function getPosition.
        Inputs:
            x0/y0/w/h    Coordinates of the tile
            earthPoint   The coordinate for target on earth surface.
        Outputs:
            success      Whether the operation succeeed
            rangeIndex   x coordinate on the image
            azimuthIndex y coordinate on the image
            sensorPos    x/y/z coordinate of the satellite
        """
        cdef double azimuthIndex, rangeIndex
        cdef double sensorPos_x, sensorPos_y, sensorPos_z
        cdef double lat, lon

        lat, lon, alt = xyz2geoWGS84(earthPoint_x, earthPoint_y, earthPoint_z)
        row, col, _ = self.bmod.projection(lon, lat, alt)
        rangeIndex = col
        azimuthIndex = row

        if not((azimuthIndex >= y0 - 1) and (azimuthIndex <= y0 + h) and (rangeIndex >= x0 - 1) and (rangeIndex <= x0 + w)):
            return False, 0, 0, 0, 0, 0

        azt, rng = self.bmod.to_azt_rng(row, col)
        sat = self.bmod.orbit.evaluate(azt) # sat is in geocentric epsg:4978
        transformer = pyproj.Transformer.from_crs(
        'epsg:4978', 'epsg:4326', always_xy=True) # Note: It would be better to write a cython function
        sensorPos_x, sensorPos_y, sensorPos_z = transformer.transform(sat[0], sat[1], sat[2])

        return True, rangeIndex, azimuthIndex, sensorPos_x, sensorPos_y, sensorPos_z

    def getPosition2(self, int x0, int y0, int w, int h, double lat, double lon, double alt):
        """
        Same as getPosition, but we don't need to call xyz2geoWGS84 for nothing
        Inputs:
            x0/y0/w/h    Coordinates of the tile
            lat/lon/alt   The coordinate for target on earth surface.
        Outputs:
            success      Whether the operation succeeed
            rangeIndex   x coordinate on the image
            azimuthIndex y coordinate on the image
            sensorPos    x/y/z coordinate of the satellite
        """
        cdef double azimuthIndex, rangeIndex
        cdef double sensorPos_x, sensorPos_y, sensorPos_z

        row, col, _ = self.bmod.projection(lon, lat, alt)
        rangeIndex = col
        azimuthIndex = row

        if not((azimuthIndex >= y0 - 1) and (azimuthIndex <= y0 + h) and (rangeIndex >= x0 - 1) and (rangeIndex <= x0 + w)):
            return False, 0, 0, 0, 0, 0

        azt, rng = self.bmod.to_azt_rng(row, col)
        sat = self.bmod.orbit.evaluate(azt) # sat is in geocentric epsg:4978
        transformer = pyproj.Transformer.from_crs(
        'epsg:4978', 'epsg:4326', always_xy=True) # Note: It would be better to write a cython function
        sensorPos_x, sensorPos_y, sensorPos_z = transformer.transform(sat[0], sat[1], sat[2])

        return True, rangeIndex, azimuthIndex, sensorPos_x, sensorPos_y, sensorPos_z


    def getPosition3(self, int x0, int y0, int w, int h, np.ndarray[np.float64_t, ndim=1] lat, np.ndarray[np.float64_t, ndim=1] lon, np.ndarray[np.float64_t, ndim=1] alt):
        """
        Same as getPosition, but we don't need to call xyz2geoWGS84 for nothing
        Inputs:
            x0/y0/w/h    Coordinates of the tile
            lat/lon/alt   The coordinate for target on earth surface.
        Outputs:
            success      Whether the operation succeeed
            rangeIndex   x coordinate on the image
            azimuthIndex y coordinate on the image
            sensorPos    x/y/z coordinate of the satellite
        """
        cdef np.ndarray[np.float64_t, ndim=1] azimuthIndex, rangeIndex, sensorPos_x, sensorPos_y, sensorPos_z

        row, col, _ = self.bmod.projection(lon, lat, alt)
        rangeIndex = col
        azimuthIndex = row
        success = (azimuthIndex >= y0 - 1) * (azimuthIndex <= y0 + h) * (rangeIndex >= x0 - 1) * (rangeIndex <= x0 + w)

        azt, rng = self.bmod.to_azt_rng(row, col)
        sat = self.bmod.orbit.evaluate(azt) # sat is in geocentric epsg:4978
        transformer = pyproj.Transformer.from_crs(
        'epsg:4978', 'epsg:4326', always_xy=True) # Note: It would be better to write a cython function
        #print(row)
        #print(col)
        #print(sat)
        sensorPos_x, sensorPos_y, sensorPos_z = transformer.transform(sat[:, 0], sat[:, 1], sat[:, 2])

        return success, rangeIndex, azimuthIndex, sensorPos_x, sensorPos_y, sensorPos_z

    def getPixPos(self, double lat, double lon, double alt):
        """
        Port of function getPixPos
        Inputs:
            lat/lon/alt: WGS84 Position of a point on earth
        Outputs:
            success      Whether the operation succeeed
            rangeIndex   x coordinate on the image
            azimuthIndex y coordinate on the image
        """
        cdef double azimuthIndex, rangeIndex

        row, col, _ = self.bmod.projection(lon, lat, alt)
        rangeIndex = col
        azimuthIndex = row

        return rangeIndex, azimuthIndex


    def computeError(self, double startPixelPos_x, double startPixelPos_y, double x0, double y0):
        """
        Port of function computeError
        Inputs:
             startPixelPos_x  x position of pixel A without taking elevation into account
             startPixelPos_y  y position of pixel A without taking elevation into account
             x0               x position of pixel B with taking elevation into account
             y0               y position of pixel B with taking elevation into account
        Outputs:
             distance         Squared L2 distance between (x, y) and (x0, y0). -1 on failure
        """
        cdef double alt, geoPos_lat, geoPos_lon, endPixelPos_x, endPixelPos_y
        startPixelPos_x = min(max(startPixelPos_x, 0.), float(self.sourceImageWidth - 1))
        startPixelPos_y = min(max(startPixelPos_y, 0.), float(self.sourceImageHeight - 1))
        geoPos_lat, geoPos_lon = self.getGeoPos(startPixelPos_x, startPixelPos_y)
        alt = self.getDemElevation(geoPos_lat, geoPos_lon)
        if isnan(alt):
            return -1, 0, 0

        endPixelPos_x, endPixelPos_y = self.getPixPos(geoPos_lat, geoPos_lon, alt)

        return (x0 - endPixelPos_x) ** 2 + (y0 - endPixelPos_y) ** 2, endPixelPos_x, endPixelPos_y

    def saveIlluminationArea(self, int x0, int y0, int w, int h, double azimuthIndex,
                             double rangeIndex, double gamma0Area,
                             double[:,:] gamma0ReferenceArea_view):
        """
        Port of saveIlluminationArea
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
        cdef int ia0 = int(azimuthIndex)
        cdef int ia1 = ia0 + 1
        cdef int ir0 = int(rangeIndex)
        cdef int ir1 = ir0 + 1

        cdef double wr = rangeIndex - ir0
        cdef double wa = azimuthIndex - ia0
        cdef double wac = 1 - wa
        cdef double wrc

        if (ir0 >= x0 and ir0 < x0 + w):
            wrc = 1 - wr
            if (ia0 >= y0 and ia0 < y0 + h):
                gamma0ReferenceArea_view[ia0 - y0][ir0 - x0] += wrc * wac * gamma0Area
            if (ia1 >= y0 and ia1 < y0 + h):
                gamma0ReferenceArea_view[ia1 - y0][ir0 - x0] += wrc * wa * gamma0Area
        if (ir1 >= x0 and ir1 < x0 + w):
            if (ia0 >= y0 and ia0 < y0 + h):
                gamma0ReferenceArea_view[ia0 - y0][ir1 - x0] += wr * wac * gamma0Area
            if (ia1 >= y0 and ia1 < y0 + h):
                gamma0ReferenceArea_view[ia1 - y0][ir1 - x0] += wr * wa * gamma0Area


    def getTruePixelPos(self, double x0, double y0):
        """
        Port of getTruePixelPos
        Given a pixel position (x0, y0),
        returns new (x1, y1), such that getGeoPos(x1, y1) returns the lat/lon that verify the property that getPixPos(lat, lon, getDemElevation(lat, lon)) points to x0, y0.

        Given how the function is used, something simpler could probably be used

        Computes the pixel position
        Inputs:
             x0      x position
             y0      y position
        Outputs
             success Whether the operation succeeed
             x       corrected x position
             y       corrected y position

        This functions returns new x, y such that computeError(x, y, x0, y0) is minimised
        Note: the minimisation algorithm doesn't seem particularly clever,
        using other algorithms (with scipy for example) might be better.
        """
        cdef int maxIterations = 100
        cdef double errThreshold = 2.0

        cdef double startPixelPos_x = x0
        cdef double startPixelPos_y = y0
        cdef double endPixelPos_x, endPixelPos_y

        cdef int numIter, i
        cdef double err2, errX, errY, alpha, tmpErr2, tmpErrX, tmpErrY
        for numIter in range(maxIterations):
            err2, endPixelPos_x, endPixelPos_y = self.computeError(startPixelPos_x, startPixelPos_y, x0, y0)
            if err2 == -1:
                return False, []
            if err2 < errThreshold:
                break

            errX = x0 - endPixelPos_x
            errY = y0 - endPixelPos_y

            alpha = 1.0
            tmpErr2 = err2
            # The following is a sort of gradient descent with learning rate (alpha) decrease if error increases
            for i in range(4):
                tmpErrX = alpha*errX
                tmpErrY = alpha*errY
                tmpStartPixelPos_x = startPixelPos_x + tmpErrX
                tmpStartPixelPos_y = startPixelPos_y + tmpErrY
                tmpErr2, _, _ = self.computeError(tmpStartPixelPos_x, tmpStartPixelPos_y, x0, y0)
                if tmpErr2 == -1:
                    continue
                if tmpErr2 < err2:
                    errX = tmpErrX
                    errY = tmpErrY
                    break
                else:
                    alpha /= 2.0
            #if the gradient descent managed to find a better minimum, continue
            if (tmpErr2 < err2):
                startPixelPos_x += errX
                startPixelPos_y += errY
            # if the gradient descent failed apply gradient descent anyway but with a random (<1) learning rate different for x and y
            else:
                startPixelPos_x += 0.1*errX # Note didn't check the real fixed random values the original java implementation would have generated. 0.1 seems just better than a random value. The overall algorithm is not particularly clever. For example, I wouldn't reset alpha to 1 every iteration.
                startPixelPos_y += 0.1*errY


        if (numIter == maxIterations):
            return False, 0., 0.
        # Original code updates endPixelPos but doesn't use it

        return True, startPixelPos_x, startPixelPos_y

    def computeTileOverlapPercentage(self, int x0, y0, int w, int h):
        """
        Port of computeTileOverlapPercentage
        Given a rectangle (x0, y0, w, h) on the image, returns the border sizes to add to the region.
        Inputs:
            x0               x coordinate of the top left corner of the region
            y0               y coordinate of the top left corner of the region
            w                width of the region
            h                height of the region
        Outputs:
            bordersizeup     percentage of pixels to use for the top border
            bordersizedown   percentage of pixels to use for the bottom border
            bordersizeleft   percentage of pixels to use for the left border
            bordersizeright  percentage of pixels to use for the right border
        """
        cdef int xMin, xMax, yMin, yMax, x, y
        cdef double pixPos_x, pixPos_y
        cdef double tileOverlapUp = 0.0, tileOverlapDown = 0.0, tileOverlapLeft = 0.0, tileOverlapRight = 0.0

        xMin = max(x0 - w//2, 0)
        xMax = min(x0 + w + w//2, self.sourceImageWidth)
        yMin = max(y0 - h//2, 0)
        yMax = min(y0 + h + h//2, self.sourceImageHeight)

        for y in range(yMin, yMax, 20):
            for x in range(xMin, xMax, 20):
                success, pixPos_x, pixPos_y = self.getTruePixelPos(float(x), float(y))
                if success:
                    tileOverlapUp = max((y - pixPos_y) / h, tileOverlapUp)
                    tileOverlapDown = max((pixPos_y - y) / h, tileOverlapDown)
                    tileOverlapLeft = max((x - pixPos_x) / w, tileOverlapLeft)
                    tileOverlapRight = max((pixPos_x - x) / w, tileOverlapRight)
        tileOverlapUp    += self.additionalOverlap
        tileOverlapDown  += self.additionalOverlap
        tileOverlapLeft  += self.additionalOverlap
        tileOverlapRight += self.additionalOverlap
        return tileOverlapUp, tileOverlapDown, tileOverlapLeft, tileOverlapRight

    def computeElevationAngle(self, double earthPoint_x, double earthPoint_y, double earthPoint_z, double sensorPos_x, double sensorPos_y, double sensorPos_z):
        """
        Port of computeElevationAngle
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

        return acos((slantRange * slantRange + H2 - R2) / (2 * slantRange * sqrt(H2))) * (180.0 / M_PI)

    def computeIlluminatedArea(self,
                               double t00_x, double t00_y, double t00_z,
                               double t01_x, double t01_y, double t01_z,
                               double t10_x, double t10_y, double t10_z,
                               double t11_x, double t11_y, double t11_z,
                               double sld_x, double sld_y, double sld_z):
        """
        Port of computeIlluminatedArea, but assuming that:
        . We already checked the dem values validity
        . We pass the position of t00, t01, t10 and t11 instead of passing
          LocalGeometry lg to compute it
        . We pass the normalized slant range direction vector directly instead of passing what is needed to compute it
        Inputs:
             x/y/z (WGS84) of t00, t01, t10 and t11
             sld (slantRangeDir) the normalized slant range direction vector
        Outputs:
             corresponding gamma0
        """
        # We use long double because the computation is quite sensitive to the precision

        # project points t00, t01, t10 and t11 to the plane perpendicular to the slant range direction vector
        cdef long double t00s = t00_x * sld_x + t00_y * sld_y + t00_z * sld_z
        cdef long double t01s = t01_x * sld_x + t01_y * sld_y + t01_z * sld_z
        cdef long double t10s = t10_x * sld_x + t10_y * sld_y + t10_z * sld_z
        cdef long double t11s = t11_x * sld_x + t11_y * sld_y + t11_z * sld_z
        cdef long double p00_x = t00_x - t00s * sld_x
        cdef long double p00_y = t00_y - t00s * sld_y
        cdef long double p00_z = t00_z - t00s * sld_z
        cdef long double p01_x = t01_x - t01s * sld_x
        cdef long double p01_y = t01_y - t01s * sld_y
        cdef long double p01_z = t01_z - t01s * sld_z
        cdef long double p10_x = t10_x - t10s * sld_x
        cdef long double p10_y = t10_y - t10s * sld_y
        cdef long double p10_z = t10_z - t10s * sld_z
        cdef long double p11_x = t11_x - t11s * sld_x
        cdef long double p11_y = t11_y - t11s * sld_y
        cdef long double p11_z = t11_z - t11s * sld_z

        # compute distances between projected points
        cdef long double p00p01 = sqrtl((p00_x-p01_x)**2 + (p00_y-p01_y)**2 + (p00_z-p01_z)**2)
        cdef long double p00p10 = sqrtl((p00_x-p10_x)**2 + (p00_y-p10_y)**2 + (p00_z-p10_z)**2)
        cdef long double p11p01 = sqrtl((p11_x-p01_x)**2 + (p11_y-p01_y)**2 + (p11_z-p01_z)**2)
        cdef long double p11p10 = sqrtl((p11_x-p10_x)**2 + (p11_y-p10_y)**2 + (p11_z-p10_z)**2)
        cdef long double p10p01 = sqrtl((p10_x-p01_x)**2 + (p10_y-p01_y)**2 + (p10_z-p01_z)**2)

        # compute semi-perimeters of two triangles: p00-p01-p10 and p11-p01-p10
        cdef long double h1 = 0.5 * (p00p01 + p00p10 + p10p01)
        cdef long double h2 = 0.5 * (p11p01 + p11p10 + p10p01)

        cdef double gamma0 = \
            sqrtl(max(0., h1 * (h1 - p00p01) * (h1 - p00p10) * (h1 - p10p01))) + \
            sqrtl(max(0., h2 * (h2 - p11p01) * (h2 - p11p10) * (h2 - p10p01)))
        return gamma0
