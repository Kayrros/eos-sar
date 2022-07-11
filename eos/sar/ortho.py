from dataclasses import dataclass

import cv2
import shapely.geometry
import numpy as np
import rasterio
import rasterio.warp

from eos.sar import model, roi, regist


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
    left, bottom, right, top = rasterio.warp.transform_bounds(
        'epsg:4326', crs, *bbox)

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
    return f'epsg:{epsg}'


class Orthorectifier:

    @staticmethod
    def from_roi(proj_model, roi, resolution, crs=None, align=None):
        coords, _, _ = proj_model.get_approx_geom(roi=roi)
        geometry = shapely.geometry.Polygon(coords)
        bbox = geometry.bounds

        if crs is None:
            crs = _utm_zone_of_bbox(bbox)

        transform, shape, _ = _compute_transform_shape(crs, resolution, bbox, align)
        ortho = Orthorectifier(proj_model, roi, crs, transform, shape)
        return ortho

    @staticmethod
    def from_transform(proj_model, roi, crs, transform, shape):
        ortho = Orthorectifier(proj_model, roi, crs, transform, shape)
        return ortho

    def __init__(self,
                 proj_model: model.SensorModel,
                 roi: roi.Roi,
                 dst_crs,
                 dst_transform,
                 dst_shape,
                 ):
        origin_col, origin_row = roi.get_origin()
        refined_geom, _, _ = proj_model.get_approx_geom(roi=roi)
        x, y, alt, transform, crs = regist.dem_points(refined_geom)

        shape = alt.shape
        rows, cols, _ = proj_model.projection(x.flatten(), y.flatten(), alt.flatten())
        rows = rows.reshape(shape)
        cols = cols.reshape(shape)
        rows -= origin_row
        cols -= origin_col

        dst_map_r = np.full(dst_shape, np.nan, dtype=np.float32)
        dst_map_c = np.full(dst_shape, np.nan, dtype=np.float32)

        rasterio.warp.reproject(
            cols, dst_map_c,
            dtype=np.float32,
            src_crs=crs,
            src_transform=transform,
            src_nodata=np.nan,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear,
        )
        rasterio.warp.reproject(
            rows, dst_map_r,
            dtype=np.float32,
            src_crs=crs,
            src_transform=transform,
            src_nodata=np.nan,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear,
        )

        self.map_x = dst_map_c.astype(np.float32)
        self.map_y = dst_map_r.astype(np.float32)
        self.crs = dst_crs
        self.transform = dst_transform

    def apply(self, raster, interpolation: Interpolation):
        assert raster.dtype == np.float32
        out = cv2.remap(
            raster,
            self.map_x,
            self.map_y,
            interpolation=interpolation.cv2_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan
        )
        return out
