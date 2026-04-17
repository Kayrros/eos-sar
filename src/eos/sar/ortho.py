from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar

import cv2
import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
import shapely.geometry
from numpy.typing import NDArray

import eos.dem
import eos.sar
from eos.sar import model
from eos.sar.roi import Roi

T = TypeVar("T", np.float32, np.complex64)


@dataclass(frozen=True)
class Interpolation:
    cv2_flag: int


NearestInterpolation = Interpolation(cv2.INTER_NEAREST)
LinearInterpolation = Interpolation(cv2.INTER_LINEAR)
CubicInterpolation = Interpolation(cv2.INTER_CUBIC)
AreaInterpolation = Interpolation(cv2.INTER_AREA)
LanczosInterpolation = Interpolation(cv2.INTER_LANCZOS4)


class AlignmentError(Exception):
    pass


def _compute_transform_shape(crs, res, bbox, align=None):
    """from aws-lambda/function-s2-crop utils_geo.py

    Compute a transform and a shape given a lon lat bbox and a resolution

    Parameters
    ----------
    crs: str
    res: float
        Positive (the y-resolution will be negative)
    bbox: tuple of float
    align: float or None, optional

    Returns
    -------
    affine.Affine
        Transform of the bbox
    tuple of int
        Shape of the bbox
    tuple of float
        Extent of the bbox in the given CRS
    """
    left, bottom, right, top = rasterio.warp.transform_bounds("epsg:4326", crs, *bbox)

    if align and (align % res > 0):
        raise AlignmentError

    if align is None:
        align = res

    if align > 0:
        left = align * np.floor(left / align)
        right = align * np.ceil(right / align)
        bottom = align * np.floor(bottom / align)
        top = align * np.ceil(top / align)

    transform = rasterio.Affine(res, 0, left, 0, -res, top)
    shape = int((top - bottom) / res), int((right - left) / res)
    extent = left, bottom, right, top

    return transform, shape, extent


def _utm_zone_of_bbox(bbox):
    zone = int(((bbox[0] + bbox[2]) / 2 + 180) // 6 + 1)
    const = 32600 if bbox[1] + bbox[3] > 0 else 32700
    epsg = const + zone
    return f"epsg:{epsg}"


class _DEMInfo:
    """The goal of this class is to precompute (and share this precomputation) the raster_xy_grid.
    This is not a huge speed-up compared to the projection for example."""

    x: np.ndarray
    y: np.ndarray
    alt: np.ndarray
    transform: rasterio.Affine
    crs: rasterio.CRS
    shape: tuple

    @staticmethod
    def from_dem(dem: eos.dem.DEM):
        x, y = eos.sar.utils.raster_xy_grid(
            dem.array.shape, dem.transform, px_is_area=True
        )

        deminfo = _DEMInfo()
        deminfo.shape = dem.array.shape
        deminfo.x = x.flatten()
        deminfo.y = y.flatten()
        deminfo.alt = dem.array.flatten()
        deminfo.transform = dem.transform
        deminfo.crs = dem.crs
        return deminfo


def _make_orthorectifier(
    proj_model: model.SensorModel,
    deminfo: _DEMInfo,
    origin_col: int,
    origin_row: int,
    dst_crs: rasterio.CRS,
    dst_transform: rasterio.Affine,
    dst_shape: tuple[int, int],
) -> Orthorectifier:
    rows, cols, _ = proj_model.projection(deminfo.x, deminfo.y, deminfo.alt)
    rows = rows.reshape(deminfo.shape)
    cols = cols.reshape(deminfo.shape)
    rows -= origin_row
    cols -= origin_col

    coordinate_map = np.full((2, *dst_shape), np.nan, dtype=np.float32)

    rasterio.warp.reproject(
        cols,
        coordinate_map[0],
        dtype=np.float32,
        src_crs=deminfo.crs,
        src_transform=deminfo.transform,
        src_nodata=np.nan,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        dst_nodata=np.nan,
        resampling=rasterio.warp.Resampling.bilinear,
    )
    rasterio.warp.reproject(
        rows,
        coordinate_map[1],
        dtype=np.float32,
        src_crs=deminfo.crs,
        src_transform=deminfo.transform,
        src_nodata=np.nan,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        dst_nodata=np.nan,
        resampling=rasterio.warp.Resampling.bilinear,
    )

    return Orthorectifier(
        _deminfo=deminfo,
        shape=dst_shape,
        crs=dst_crs,
        transform=dst_transform,
        coordinate_map=coordinate_map,
    )


@dataclass(frozen=True)
class Orthorectifier:
    """
    Allows to orthorectify a SAR raster onto a CRS.

    Two functions are available to create an Orthorectifier:
    - Orthorectifier.from_roi:
        Takes a sensor model, a Roi, a dem, a desired resolution, and optionally a CRS and an alignment.
        This function estimates an adequate (transform, shape) tuple according to the desired resolution, CRS, and bounding box (derived from the sensor model + the roi (SensorModel.get_buffered_geom)).
        The function will determine a relevant UTM zone if the CRS is not provided; in this case, the resolution should be in meters.
    - Orthorectifier.from_transform:
        Similar to from_roi, but it takes a CRS/transform/shape as input, which might not exactly correspond to the roi that is given.
        In general, it is recommended to use this function when possible, as it avoids computing a 4326 bounding box based on the sensor model + roi (which is always ricky).

    In both cases, the (transform,shape) tuple is then used to subset the DEM: a bounding box that contains the (transform,shape) in 4326 is estimated and `DEM.subset` is used.
    Each point of the DEM is projected to sensor geometry, which provides row/col points in the DEM CRS (likely 4326). These maps row/col are then reprojected to the desired CRS/transform/shape. The result is coordinate maps that allows to 'pull' the SAR signal to their ground coordinates.

    The DEM used for orthorectification does not need to be larger than the 4326 bounding box that contains the destination geometry, but the roi has to be larger depending on the altitudes. This computation is not done in the Orthorectifier, the user has to take care of this aspect themself.

    The method `apply` allows to orthorectify a raster (complex64 or float32).
    This step is simply the warping of the input raster according to the coordinate maps that were precomputed.
    The resulting array has a geometry defined by the orthorectifier `shape`, `transform` and `crs` fields.

    The method `on_multilooked_raster` can be used to create a new Orthorectifier that is adapted to the multilooked rasters. The Orthorectifier itself will not perform any multilooking, it is the user's responsibility to ensure that the raster that is given to `apply` has been multilooked with the same factors as those provided to `on_multilooked_raster`.

    An Orthorecifier can be reused for multiple orthorectification, as long as the projection model (SensorModel) and Roi is the same.
    This is typically true for different polarizations of a single product, or when the products in a timeseries are all registered to a common geometry (for example in an interferometric stack).
    """

    shape: tuple[int, int]
    transform: rasterio.Affine
    crs: rasterio.CRS
    coordinate_map: NDArray[np.float32]
    """ shape is (2, self.shape[0], self.shape[1]). coordinate_map[0] is for X, coordinate_map[1] is for Y """

    _deminfo: Optional[_DEMInfo] = None

    @staticmethod
    def from_roi(
        proj_model: model.SensorModel,
        roi: Roi,
        resolution: float,
        dem: eos.dem.DEM,
        crs: Optional[rasterio.CRS] = None,
        align: Optional[float] = None,
    ) -> Orthorectifier:
        """
        Make sure to use a resolution in the same units as the CRS. For UTM (when crs=None), the resolution should be in meters.
        """
        coords = proj_model.get_buffered_geom(roi=roi, dem=dem)
        geometry = shapely.geometry.Polygon(coords)
        bbox = geometry.bounds

        if crs is None:
            crs = _utm_zone_of_bbox(bbox)

        transform, shape, bbox = _compute_transform_shape(crs, resolution, bbox, align)

        # subset the dem to the bbox of the desired shape/transform
        bbox = rasterio.warp.transform_bounds(crs, dem.crs, *bbox)
        dem = dem.subset(bbox)
        deminfo = _DEMInfo.from_dem(dem)

        origin_col, origin_row = roi.get_origin()
        ortho = _make_orthorectifier(
            proj_model, deminfo, origin_col, origin_row, crs, transform, shape
        )
        return ortho

    @staticmethod
    def from_transform(
        proj_model: model.SensorModel,
        roi: Roi,
        crs: rasterio.CRS,
        transform: rasterio.Affine,
        shape: tuple[int, int],
        dem: eos.dem.DEM,
        previous_orthorectifier: Optional[Orthorectifier] = None,
    ) -> Orthorectifier:
        if previous_orthorectifier and previous_orthorectifier._deminfo:
            assert previous_orthorectifier.crs == crs
            assert previous_orthorectifier.transform == transform
            assert previous_orthorectifier.shape == shape
            deminfo = previous_orthorectifier._deminfo
        else:
            # subset the dem to the desired shape/transform
            bounds = rasterio.transform.array_bounds(*shape, transform)
            bounds = rasterio.warp.transform_bounds(crs, dem.crs, *bounds)
            dem = dem.subset(bounds)
            deminfo = _DEMInfo.from_dem(dem)

        origin_col, origin_row = roi.get_origin()
        ortho = _make_orthorectifier(
            proj_model, deminfo, origin_col, origin_row, crs, transform, shape
        )
        return ortho

    def apply(self, raster: NDArray[T], interpolation: Interpolation) -> NDArray[T]:
        if raster.dtype == np.complex64:
            real_out = self.apply(np.real(raster), interpolation)
            imag_out = self.apply(np.imag(raster), interpolation)
            out = real_out + 1j * imag_out
        else:
            assert raster.dtype == np.float32
            out = cv2.remap(
                raster,  # type: ignore[arg-type]
                self.coordinate_map[0],
                self.coordinate_map[1],
                interpolation=interpolation.cv2_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(np.nan,),
            )
        return out

    def on_multilooked_raster(self, azimuth: float, range: float) -> Orthorectifier:
        """
        Create a new Orthorectifier that is a multilooked version of the current one.
        The new Orthorectifier can be used to orthorectify a multilooked raster that was generated from the same sensor model and roi.
        """
        new_coordinate_map = self.coordinate_map.copy()
        new_coordinate_map[0] /= range
        new_coordinate_map[1] /= azimuth
        return Orthorectifier(
            shape=self.shape,
            transform=self.transform,
            crs=self.crs,
            coordinate_map=new_coordinate_map,
            _deminfo=self._deminfo,
        )
