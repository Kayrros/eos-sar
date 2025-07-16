from __future__ import annotations

import abc
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import affine
import numpy as np
import rasterio
import rasterio.errors
import rasterio.session
import rasterio.windows
from numpy.typing import ArrayLike, NDArray
from typing_extensions import TypeAlias

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
    profile = dict(
        driver="GTiff",
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
        blockysize=256,
    )

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
        u * v * array[:, 0, 0]
        + dx * v * array[:, 0, 1]
        + u * dy * array[:, 1, 0]
        + dx * dy * array[:, 1, 1]
    )

    return np.around(h_interp, 5)


class OutOfBoundsException(IndexError):
    pass


@dataclass(frozen=True)
class DEM:
    array: NDArray[np.float32]
    """ raster containing heights in meters relative to the ellipsoid """
    transform: affine.Affine
    """ the transform associated with the raster, expressed in the "pixel is area" convention (GDAL default) """
    crs: str = "EPSG:4326"
    """ always "EPSG:4326" """

    def __post_init__(self):
        # make the array read-only, just in case
        self.array.setflags(write=False)

    def _assert_in_raster(self, xmin: float, xmax: float, ymin: float, ymax: float):
        if xmin < 0:
            raise OutOfBoundsException(
                f"x coord min {xmin} negative, out of raster bounds"
            )
        if xmax > self.array.shape[1] - 1:
            raise OutOfBoundsException(
                f"x coord max {xmax}, out of raster bounds, shape: {self.array.shape}"
            )
        if ymin < 0:
            raise OutOfBoundsException(
                f"y coord min {ymin} negative, out of raster bounds"
            )
        if ymax > self.array.shape[0] - 1:
            raise OutOfBoundsException(
                f"y coord max {ymax}, out of raster bounds, shape: {self.array.shape}"
            )

    def elevation(
        self, lons: ArrayLike, lats: ArrayLike, interpolation: str = "bilinear"
    ) -> Union[float, list[float], NDArray[np.float32]]:
        """
        Gives the altitude of a (list of) point(s).

        Args:
            lons, lats: longitude (or list of longitudes) and latitude (or list of latitudes)
            interpolation (str): if 'bilinear' (default) returns the height bilinearily interpolated,
                else if 'nearest' returns the nearest neighbor value
        Returns:
            alts: height (or list/array of heights) in meters above the ellipsoid

        Raises:
            OutOfBoundsException if the DEM is not sufficiently big to allow
            querying for points
        """
        is_input_iterable = isinstance(lons, Iterable)

        lats_arr = np.atleast_1d(np.asarray(lats))
        lons_arr = np.atleast_1d(np.asarray(lons))
        assert len(lons_arr) == len(lats_arr), "arguments must have same length"

        geo_coords = np.array([lons_arr, lats_arr])
        # the transform's convention is pixel is area so we shift half a pixel
        img_coords = np.around(~self.transform * geo_coords, 6) - 0.5

        xmin = img_coords[0].min()
        xmax = img_coords[0].max()
        ymin = img_coords[1].min()
        ymax = img_coords[1].max()
        self._assert_in_raster(xmin, xmax, ymin, ymax)

        if interpolation == "nearest":
            alts = np.array(
                [self.array[int(round(y)), int(round(x))] for x, y in zip(*img_coords)]
            )
        else:
            dem_subparts = []
            for x, y in zip(img_coords[0], img_coords[1]):
                xx = int(x)
                yy = int(y)
                window = self.array[yy : yy + 2, xx : xx + 2]
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

        Note:
            The bounds should be well inside the dem. Near the border, the behaviour is not exactly well defind for now.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            np.ndarray: raster DEM crop array
            affine.Affine: transform
            rasterio.crs.CRS: CRS

        Raises:
            OutOfBoundsException if the DEM is not big enough to crop at specified bounds.
        """

        xmin, ymin = np.array(~self.transform * np.array((bounds[0], bounds[3])))
        xmax, ymax = np.array(~self.transform * np.array((bounds[2], bounds[1])))

        xmin = int(np.floor(xmin))
        ymin = int(np.floor(ymin))
        xmax = int(np.floor(xmax))
        ymax = int(np.floor(ymax))

        self._assert_in_raster(xmin, xmax, ymin, ymax)

        array = self.array[ymin : ymax + 1, xmin : xmax + 1]
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

        Raises:
            OutOfBoundsException if the DEM is not big enough to subset at specified bounds.
        """
        array, transform, _ = self.crop(bounds)
        if copy:
            array = array.copy()
        return DEM(array=array, transform=transform)

    def fill_nan(self, value: float = 0.0):
        new_array = self.array.copy()
        new_array[np.isnan(new_array)] = value
        return DEM(array=new_array, transform=self.transform, crs=self.crs)

    @staticmethod
    def from_rasterio_dataset(dataset: rasterio.DatasetReader):
        array = dataset.read(1)
        profile: dict[str, Any] = dataset.profile
        transform = profile["transform"]
        assert profile["crs"] == "EPSG:4326"
        return DEM(array=array, transform=transform)

    def get_extent(self):
        return rasterio.transform.array_bounds(
            self.array.shape[0], self.array.shape[1], self.transform
        )


class DEMSource(abc.ABC):
    """
    Notes:
        The only supported datum is ellipsoidal.
    """

    @abc.abstractmethod
    def fetch_dem(self, bounds: Bounds) -> DEM:
        """
        Return a DEM instance on the given `bounds`.

        Note:
            Depending on the actual DEMSource, there is currently no guarantees that the bounds are exactly contained in the resulting DEM.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            dem: eos.dem.DEM instance
        """


@dataclass(frozen=True)
class SRTM4Source(DEMSource):
    nan_value: Optional[float] = None

    def fetch_dem(self, bounds: Bounds) -> DEM:
        array, transform, crs = srtm4.crop(bounds, datum="ellipsoidal")
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert transform is not None
        assert crs == "EPSG:4326"
        dem = DEM(array=array, transform=transform)
        if self.nan_value is not None:
            dem = dem.fill_nan(value=self.nan_value)
        return dem

    def elevation(self, lons, lats, interpolation="bilinear"):
        warnings.warn(
            "DEMSource.elevation is deprecated. Use DEMSource.fetch_dem(bounds).elevation(lons, lats).",
            DeprecationWarning,
        )
        assert interpolation == "bilinear"
        return srtm4.srtm4(lons, lats)

    def crop(self, bounds):
        warnings.warn(
            "DEMSource.crop is deprecated. Use DEMSource.fetch_dem(bounds).crop(bounds).",
            DeprecationWarning,
        )
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
        array, transform, crs = multidem.crop(
            bounds, source=self.demsource, datum="ellipsoidal"
        )
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert crs == "EPSG:4326"
        return DEM(array=array, transform=transform)

    def elevation(self, lons, lats, interpolation="bilinear"):
        warnings.warn(
            "DEMSource.elevation is deprecated. Use DEMSource.fetch_dem(bounds).elevation(lons, lats).",
            DeprecationWarning,
        )
        return multidem.elevation(
            lons,
            lats,
            interpolation=interpolation,
            source=self.demsource,
            datum="ellipsoidal",
        )

    def crop(self, bounds):
        warnings.warn(
            "DEMSource.crop is deprecated. Use DEMSource.fetch_dem(bounds).crop(bounds).",
            DeprecationWarning,
        )
        return multidem.crop(bounds, source=self.demsource, datum="ellipsoidal")


@dataclass(frozen=True)
class DEMStitcherSource(DEMSource):
    dem_name: str = "glo_30"
    fill_in_glo_30: bool = True
    dst_resolution: Optional[Union[float, tuple[float]]] = None
    """ see https://github.com/ACCESS-Cloud-Based-InSAR/dem-stitcher#dems-supported """
    tiles_cache_dir: Optional[Path] = None
    """ Directory to where dem-stitch will cache the DEM tiles.
    If the directory does not exist, dem_stitcher will create it.
    """

    def fetch_dem(self, bounds: Bounds) -> DEM:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            array, profile = dem_stitcher.stitch_dem(
                bounds=list(bounds),
                dem_name=self.dem_name,
                merge_nodata_value=0,
                dst_resolution=self.dst_resolution,
                fill_in_glo_30=self.fill_in_glo_30,
                dst_tile_dir=self.tiles_cache_dir,
            )
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert profile["crs"] == "EPSG:4326"
        transform = profile["transform"]
        return DEM(array=array, transform=transform)


def get_any_source() -> DEMSource:
    if has_multidem:
        demsource = os.environ.get("EOS_SAR_MULTIDEM_SOURCE", "SRTM30")
        return MultidemSource(demsource)
    if has_srtm4:
        return SRTM4Source()
    if has_demstitcher:
        return DEMStitcherSource()
    raise RuntimeError(
        "couldn't find a DEM source; please install multidem, srtm4 or dem-stitcher."
    )


### Personal class for DEM stored locally
### author: Arthur Hauck 22/01/2024
class MyDEMSource(DEMSource):
    def __init__(
        self,
        path_to_dem: str,
        margin: Optional[float] = None,
        set_nan: bool = True,
        geoid_name: Optional[str] = None,
    ):
        """
        Instantiate a MyDEMSource object.

        Parameters
        ----------
        path_to_dem : str
            Path to the DEM file. The DEM has to be projected on EPSG:4326 or EPSG:4979.
        margin : float, optional
            If not None, Extent of the margins for the padding around the DEM.
            The unit is degree (from EPSG:4326).
        set_nan : bool
            Set to True if you want to fill NoData Values of the DEM with np.nan.
            The default is True.
        geoid_name : str, optional
            Name of the geoid relative to which the altitudes of the DEM are given.
            The height of the geoid will be removed to get ellipsoidal heights (ie. relative to the WGS84 ellipsoid).
            The default is None. Requires the dem-stitcher dependency.
        """
        # Store metadata
        dem_reader = rasterio.open(path_to_dem)
        transform = dem_reader.meta["transform"]
        nodata = dem_reader.meta["nodata"]
        if set_nan:
            nodata = np.nan

        #if dem_reader.meta["crs"] != rasterio.CRS.from_epsg(4326) or dem_reader.meta["crs"] != rasterio.CRS.from_epsg(4979):
        #    raise ValueError(f"CRS of '{path_to_dem}'")

        # Get the DEM
        if set_nan:
            dem = dem_reader.read(1, masked=True).filled(np.nan)
        else:
            dem = dem_reader.read(1)

        # Set to ellipsoidal height if the given DEM is referenced relative to a geoid
        if geoid_name is not None:
            assert has_demstitcher, (
                "Should have dem_stitcher to convert altitudes from the geoid to the ellipsoid"
            )
            dem = dem_stitcher.geoid.remove_geoid(
                dem_arr=dem,
                dem_profile=dem_reader.profile,
                geoid_name=geoid_name,
                dem_area_or_point="Area",
            )

        if margin is not None:
            # Pad the DEM (add NoData around)
            h, w = dem.shape
            lon_min = transform[2]
            lon_max = transform[2] + (w - 1) * transform[0]
            lat_max = transform[5]
            lat_min = transform[5] + (h - 1) * transform[4]
            j_padding, i_padding = ~transform * np.array(
                [
                    [lon_min - margin, lon_max + margin],
                    [lat_min - margin, lat_max + margin],
                ]
            )
            jmin, jmax = np.abs(np.round(j_padding).astype(int))
            imax, imin = np.abs(np.round(i_padding).astype(int))
            padded_dem = np.full(
                (imax + imin + 1, jmax + jmin + 1), nodata, dtype=np.float32
            )
            padded_dem[imin : imin + h, jmin : jmin + w] = dem
            dem = padded_dem

            # Update the affine.Affine.transform
            orig_lon, orig_lat = transform * (-jmin, -imin)
            transform = affine.Affine(
                transform[0],
                transform[1],
                float(orig_lon),
                transform[3],
                transform[4],
                float(orig_lat),
            )

        # Keep the DEM as an attribute
        self._dem = DEM(array=dem, transform=transform)

    def fetch_dem(self, bounds: Bounds) -> DEM:
        return self._dem.subset(bounds, copy=True)

    def get_extent(self):
        return self._dem.get_extent()
