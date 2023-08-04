from dataclasses import dataclass
import os
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
import rasterio
import rasterio.session
import rasterio.errors
import rasterio.windows
import pyproj

try:
    import multidem
    has_multidem = True
except ImportError:
    has_multidem = False

try:
    import srtm4
    has_srtm4 = True
except ImportError:
    has_srtm4 = False


# copy-pasted from multidem
def write_crop_to_file(array, transform, crs, path):
    """
    Write a georeferenced raster to a GeoTIFF file.

    Args:
        array (np.ndarray): raster array
        transform (affine.Affine): raster transform
        crs (rasterio.crs.CRS): raster CRS
        path (str): path to output file
    """
    height, width = array.shape
    profile = dict(driver="GTiff",
                   count=1,
                   width=width,
                   height=height,
                   dtype=array.dtype,
                   transform=transform,
                   crs=crs,
                   tiled=True,
                   compress="deflate",
                   predictor=2,
                   blockxsize=256,
                   blockysize=256)

    with rasterio.open(path, "w", **profile) as f:
        f.write(array, 1)


@dataclass(frozen=True)
class DEM:
    array: NDArray[np.float32]  # should be relative to the ellipsoid and with good pixel convention (TODO)
    transform: Any
    crs: str = "EPSG:4326"
    nodata: float = np.nan

    def elevation(self,
                  lons: ArrayLike,
                  lats: ArrayLike,
                  interpolation: str = "bilinear") -> NDArray[np.float32]:
        """
        Gives the altitude of a (list of) point(s).

        Args:
            lons, lats: longitude (or list of longitudes) and latitude (or list of latitudes)
            interpolation (str): if 'bilinear' (default) returns the height bilinearily interpolated,
                else if 'nearest' returns the nearest neighbor value
        Returns:
            alts: height (or list/array of heights) in meters above the ellipsoid
        """
        is_input_iterable = isinstance(lons, Iterable)

        if not isinstance(lats, Iterable):
            lats = [lats]
        if not isinstance(lons, Iterable):
            lons = [lons]

        assert len(lons) == len(lats), "arguments must have same length"

        geo_coords = np.array([lons, lats])
        img_coords = np.around(~self.transform * geo_coords, 6) - 0.5
        print('x:', img_coords[0].min(), img_coords[0].max(), self.array.shape[1])
        print('y:', img_coords[1].min(), img_coords[1].max(), self.array.shape[0])
        # TODO: check that img_coords is contained inside the DEM

        if interpolation == "nearest":
            alts = np.array([self.array[y, x] for x, y in img_coords])
        else:
            dem_subparts = []
            for x, y in zip(img_coords[0], img_coords[1]):
                xx = int(x)
                yy = int(y)
                dem = self.array[yy:yy+2, xx:xx+2]
                dem_subparts.append(dem.flatten())

            dem_subparts = np.stack(dem_subparts, axis=0)
            alts = bilinear_interp(dem_subparts, img_coords[0], img_coords[1])

        if not is_input_iterable:
            return alts[0]
        elif isinstance(lons, np.ndarray):
            return np.asarray(alts)
        else:
            return alts.tolist() if isinstance(alts, np.ndarray) else alts

    def crop(self, bounds):
        """
        Return a crop of the `source` DEM on the given `bounds`.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            np.ndarray: raster DEM crop array
            affine.Affine: transform
            rasterio.crs.CRS: CRS
        """
        # lon_min, lat_min, lon_max, lat_max = bounds
        # (x0, y0) = np.around(~self.transform * (lon_min, lat_min), 6) - 0.5
        # (x1, y1) = np.around(~self.transform * (lon_max, lat_max), 6) - 0.5
        return self.array, self.transform, self.crs


class DEMSource:
    """
    Notes:
        The only supported datum is ellipsoidal.
    """

    def fetch_dem(self, bounds) -> DEM:
        """
        Return a DEM instance on the given `bounds`.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            dem: eos.dem.DEM instance
        """
        raise NotImplementedError



class SRTM4Source(DEMSource):

    def fetch_dem(self, bounds):
        array, transform, crs = srtm4.crop(bounds, datum="ellipsoidal")
        assert crs == 'EPSG:4326'
        return DEM(
            array=array,
            transform=transform,
        )


class MultidemSource(DEMSource):

    def __init__(self, demsource="SRTM30"):
        """
        Args:
            demsource (str): DEM source "SRTM30" (default), "TDM90", "SRTM90" or
                "SRTM90-CGIAR-CSI"
        """
        multidem._source_validation(demsource)
        self.demsource = demsource

    def fetch_dem(self, bounds) -> DEM:
        array, transform, crs = multidem.crop(bounds, source=self.demsource, datum="ellipsoidal")
        assert crs == 'EPSG:4326'
        return DEM(
            array=array,
            transform=transform,
        )


def bilinear_interp(array, x, y):
    """Returns the value for the fractional row/col using bilinear interpolation
        between the cells.

    Args:
        array (numpy.ndarray): 2-D array of floats.
        x (list): horizontal image coordinates.
        y (list): vertical image coordinates.

    Returns:
        numpy.ndarray: 1-D array of floats.
    """
    i = np.array(np.floor(x), dtype=int)
    j = np.array(np.floor(y), dtype=int)

    dx, dy = x - i, y - j

    ones = np.ones(len(x))
    u, v = ones - dx, ones - dy

    h_interp = (
        u * v * array[:, 0] + dx * v * array[:, 1] + u * dy * array[:, 2] + dx * dy * array[:, 3]
    )

    return np.around(h_interp, 5)

class GLO30Source(DEMSource):

    def __init__(self):
        pass

    def fetch_dem(self, bounds) -> DEM:
        raise NotImplementedError


def get_any_source() -> DEMSource:
    if has_multidem:
        demsource = os.environ.get('EOS_SAR_MULTIDEM_SOURCE', 'SRTM30')
        return MultidemSource(demsource)
    if has_srtm4:
        return SRTM4Source()
    return GLO30Source()
