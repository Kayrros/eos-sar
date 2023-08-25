from __future__ import annotations
import abc
from dataclasses import dataclass
import os
from typing import Any, Iterable, Union
from typing_extensions import TypeAlias
import affine
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import rasterio
import rasterio.session
import rasterio.errors
import rasterio.windows

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

try:
    import dem_stitcher
    has_demstitcher = True
except ImportError:
    has_demstitcher = False


Bounds: TypeAlias = tuple[float, float, float, float]
""" (lon_min, lat_min, lon_max, lat_max) """


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


def lonlat_list_to_bounds(lons: ArrayLike, lats: ArrayLike) -> Bounds:
    lon_min = np.min(lons)
    lat_min = np.min(lats)
    lon_max = np.max(lons)
    lat_max = np.max(lats)
    return (lon_min, lat_min, lon_max, lat_max)


def _bilinear_interp(array, x, y):
    """Returns the value for the fractional row/col using bilinear interpolation
        between the cells.

    Args:
        array : numpy.ndarray
            Values to interpolate. It should be a 3D array of shape (N, 2, 2),
            where (2, 2) represent the window around each sample, N samples.
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
        u * v * array[:, 0, 0] + dx * v * array[:, 0, 1] + u * dy * array[:, 1, 0] + dx * dy * array[:, 1, 1]
    )

    return np.around(h_interp, 5)


@dataclass(frozen=True)
class DEM:
    array: NDArray[np.float32]
    """ raster containing heights in meters relative to the ellipsoid """
    transform: affine.Affine
    crs: str = "EPSG:4326"
    """ always "EPSG:4326" """

    def __post_init__(self):
        # make the array read-only, just in case
        self.array.setflags(write=False)

    def elevation(self,
                  lons: ArrayLike,
                  lats: ArrayLike,
                  interpolation: str = "bilinear") -> Union[float, list[float], NDArray[np.float32]]:
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

        lats_arr = np.atleast_1d(np.asarray(lats))
        lons_arr = np.atleast_1d(np.asarray(lons))
        assert len(lons_arr) == len(lats_arr), "arguments must have same length"

        geo_coords = np.array([lons_arr, lats_arr])
        # TODO: not sure about the -0.5
        img_coords = np.around(~self.transform * geo_coords, 6) - 0.5
        assert (img_coords >= 0).all()
        maxx = img_coords[0].max()
        maxy = img_coords[1].max()
        assert (maxx + 1 < self.array.shape[1]).all(), f"x coord max {maxx}, shape: {self.array.shape}"  # +1 because we need data for interpolation
        assert (maxy + 1 < self.array.shape[0]).all(), f"y coord max {maxy}, shape: {self.array.shape}"

        if interpolation == "nearest":
            alts = np.array([self.array[int(round(y)), int(round(x))] for x, y in zip(*img_coords)])
        else:
            dem_subparts = []
            for x, y in zip(img_coords[0], img_coords[1]):
                xx = int(x)
                yy = int(y)
                window = self.array[yy:yy + 2, xx:xx + 2]
                dem_subparts.append(window)

            dem_subparts = np.stack(dem_subparts, axis=0)
            alts = _bilinear_interp(dem_subparts, img_coords[0], img_coords[1])

        if not is_input_iterable:
            return alts[0]
        elif isinstance(lons, np.ndarray):
            return np.asarray(alts)
        else:
            return alts.tolist() if isinstance(alts, np.ndarray) else alts

    def crop(self, bounds: Bounds) -> tuple[NDArray[np.float32], affine.Affine, str]:
        """
        Return a crop of the DEM on the given `bounds`, as an array, transform, crs.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            np.ndarray: raster DEM crop array
            affine.Affine: transform
            rasterio.crs.CRS: CRS
        """

        xmin, ymin = np.array(~self.transform * np.array((bounds[0], bounds[3])))
        xmax, ymax = np.array(~self.transform * np.array((bounds[2], bounds[1])))

        xmin = int(np.floor(xmin))
        ymin = int(np.floor(ymin))
        xmax = int(np.floor(xmax))
        ymax = int(np.floor(ymax))

        if xmin < 0 or ymin < 0 or xmax >= self.array.shape[1] or ymax >= self.array.shape[0]:
            raise ValueError("out of bounds")

        array = self.array[ymin:ymax + 1, xmin:xmax + 1]
        resx = self.transform.a
        resy = self.transform.e
        transform = affine.Affine.translation(xmin * resx, ymin * resy) * self.transform

        return array, transform, self.crs

    def subset(self, bounds: Bounds, copy: bool = False) -> DEM:
        """
        Return a subset of the DEM on the given `bounds` as a new DEM instance.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max
            copy (bool optional): Copy the array so that the parent instance can be freed. Otherwise it uses a view.

        Returns:
            DEM: subset
        """
        array, transform, _ = self.crop(bounds)
        if copy:
            array = array.copy()
        return DEM(array=array, transform=transform)

    @staticmethod
    def from_rasterio_dataset(dataset: rasterio.DatasetReader):
        array = dataset.read(1)
        profile: dict[str, Any] = dataset.profile
        transform = profile["transform"]
        assert profile["crs"] == "EPSG:4326"
        return DEM(array=array, transform=transform)


class DEMSource(abc.ABC):
    """
    Notes:
        The only supported datum is ellipsoidal.
    """

    @abc.abstractmethod
    def fetch_dem(self, bounds: Bounds) -> DEM:
        """
        Return a DEM instance on the given `bounds`.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            dem: eos.dem.DEM instance
        """


@dataclass(frozen=True)
class SRTM4Source(DEMSource):

    def fetch_dem(self, bounds: Bounds) -> DEM:
        array, transform, crs = srtm4.crop(bounds, datum="ellipsoidal")
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert transform is not None
        assert crs == 'EPSG:4326'
        return DEM(array=array, transform=transform)

    def elevation(self, lons, lats, interpolation="bilinear"):
        warnings.warn("DEMSource.elevation is deprecated. Use DEMSource.fetch_dem(bounds).elevation(lons, lats).",
                      DeprecationWarning)
        assert interpolation == "bilinear"
        return srtm4.srtm4(lons, lats)

    def crop(self, bounds):
        warnings.warn("DEMSource.crop is deprecated. Use DEMSource.fetch_dem(bounds).crop(bounds).",
                      DeprecationWarning)
        return srtm4.crop(bounds, datum="ellipsoidal")


class MultidemSource(DEMSource):

    def __init__(self, demsource="SRTM30"):
        """
        Args:
            demsource (str): DEM source "SRTM30" (default), "TDM90", "SRTM90" or
                "SRTM90-CGIAR-CSI"
        """
        multidem._source_validation(demsource)
        self.demsource = demsource

    def fetch_dem(self, bounds: Bounds) -> DEM:
        array, transform, crs = multidem.crop(bounds, source=self.demsource, datum="ellipsoidal")
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert crs == 'EPSG:4326'
        return DEM(array=array, transform=transform)

    def elevation(self, lons, lats, interpolation="bilinear"):
        warnings.warn("DEMSource.elevation is deprecated. Use DEMSource.fetch_dem(bounds).elevation(lons, lats).",
                      DeprecationWarning)
        return multidem.elevation(lons, lats, interpolation=interpolation,
                                  source=self.demsource, datum="ellipsoidal")

    def crop(self, bounds):
        warnings.warn("DEMSource.crop is deprecated. Use DEMSource.fetch_dem(bounds).crop(bounds).",
                      DeprecationWarning)
        return multidem.crop(bounds, source=self.demsource, datum="ellipsoidal")


@dataclass(frozen=True)
class DEMStitcherSource(DEMSource):

    dem_name: str = "glo30"
    """ see https://github.com/ACCESS-Cloud-Based-InSAR/dem-stitcher#dems-supported """

    def fetch_dem(self, bounds: Bounds) -> DEM:
        array, profile = dem_stitcher.stitch_dem(list(bounds), "glo_30")
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert profile["crs"] == "EPSG:4326"
        transform = profile["transform"]
        return DEM(array=array, transform=transform)


def get_any_source() -> DEMSource:
    if has_multidem:
        demsource = os.environ.get('EOS_SAR_MULTIDEM_SOURCE', 'SRTM30')
        return MultidemSource(demsource)
    if has_srtm4:
        return SRTM4Source()
    if has_demstitcher:
        return DEMStitcherSource()
    raise RuntimeError("couldn't find a DEM source; please install multidem, srtm4 or dem-stitcher.")
