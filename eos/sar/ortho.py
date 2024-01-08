from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import cv2
import numpy as np
import rasterio
import rasterio.warp
import shapely.geometry
from numpy.typing import NDArray

import eos.dem
from eos.sar import model, regist

T = TypeVar("T", np.float32, np.complex64)


@dataclass
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
        Extent of the bbox
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
    x: np.ndarray
    y: np.ndarray
    alt: np.ndarray
    transform: rasterio.Affine
    crs: rasterio.CRS
    shape: tuple

    @staticmethod
    def from_proj_model(proj_model, roi, dem: eos.dem.DEM):
        refined_geom, _, _ = proj_model.get_approx_geom(roi=roi, dem=dem)
        x, y, alt, transform, crs = regist.dem_points(refined_geom, dem=dem)

        deminfo = _DEMInfo()
        deminfo.shape = alt.shape
        deminfo.x = x.flatten()
        deminfo.y = y.flatten()
        deminfo.alt = alt.flatten()
        deminfo.transform = transform
        deminfo.crs = crs
        return deminfo


class Orthorectifier:
    shape: tuple
    transform: rasterio.Affine
    crs: rasterio.CRS

    @staticmethod
    def from_roi(
        proj_model, roi, resolution, dem: eos.dem.DEM, crs=None, align=None
    ) -> Orthorectifier:
        coords, _, _ = proj_model.get_approx_geom(roi=roi, dem=dem)
        geometry = shapely.geometry.Polygon(coords)
        bbox = geometry.bounds

        if crs is None:
            crs = _utm_zone_of_bbox(bbox)

        transform, shape, _ = _compute_transform_shape(crs, resolution, bbox, align)
        origin_col, origin_row = roi.get_origin()
        deminfo = _DEMInfo.from_proj_model(proj_model, roi, dem=dem)
        ortho = Orthorectifier(
            proj_model, deminfo, origin_col, origin_row, crs, transform, shape
        )
        return ortho

    @staticmethod
    def from_transform(
        proj_model,
        roi,
        crs,
        transform,
        shape,
        dem: eos.dem.DEM,
        previous_orthorectifier=None,
    ) -> Orthorectifier:
        if previous_orthorectifier:
            assert previous_orthorectifier.crs == crs
            assert previous_orthorectifier.transform == transform
            assert previous_orthorectifier.shape == shape
            deminfo = previous_orthorectifier._deminfo
        else:
            deminfo = _DEMInfo.from_proj_model(proj_model, roi, dem=dem)

        origin_col, origin_row = roi.get_origin()
        ortho = Orthorectifier(
            proj_model, deminfo, origin_col, origin_row, crs, transform, shape
        )
        return ortho

    def __init__(
        self,
        proj_model: model.SensorModel,
        deminfo: _DEMInfo,
        origin_col,
        origin_row,
        dst_crs,
        dst_transform,
        dst_shape,
    ):
        rows, cols, _ = proj_model.projection(deminfo.x, deminfo.y, deminfo.alt)
        rows = rows.reshape(deminfo.shape)
        cols = cols.reshape(deminfo.shape)
        rows -= origin_row
        cols -= origin_col

        self._map_x = np.full(dst_shape, np.nan, dtype=np.float32)
        self._map_y = np.full(dst_shape, np.nan, dtype=np.float32)

        rasterio.warp.reproject(
            cols,
            self._map_x,
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
            self._map_y,
            dtype=np.float32,
            src_crs=deminfo.crs,
            src_transform=deminfo.transform,
            src_nodata=np.nan,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear,
        )

        self._deminfo = deminfo
        self.shape = dst_shape
        self.crs = dst_crs
        self.transform = dst_transform

    def apply(self, raster: NDArray[T], interpolation: Interpolation) -> NDArray[T]:
        if raster.dtype == np.complex64:
            real_out = self.apply(np.real(raster), interpolation)
            imag_out = self.apply(np.imag(raster), interpolation)
            out = real_out + 1j * imag_out
        else:
            assert raster.dtype == np.float32
            out = cv2.remap(
                raster,
                self._map_x,
                self._map_y,
                interpolation=interpolation.cv2_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(np.nan,),
            )
        return out

    def apply_stack(
        self, rasters: Sequence[NDArray[T]], interpolation: Interpolation
    ) -> NDArray[T]:
        n = len(rasters)
        assert n > 0
        result = np.empty((n, *rasters[0].shape), dtype=rasters[0].dtype)
        for i in range(n):
            result[i] = self.apply(rasters[i], interpolation)
        return result
