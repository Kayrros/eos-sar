"""
High-level module to generate Analysis-Ready-Data (ARD) crops of Sentinel-1 IW GRD.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
import shapely.geometry
from numpy.typing import NDArray
from typing_extensions import assert_never, override

import eos.dem
import eos.sar
from eos.products import sentinel1
from eos.products.sentinel1 import orbit_catalog
from eos.sar.model import SensorModel
from eos.sar.ortho import Orthorectifier
from eos.sar.roi import Roi

logger = logging.getLogger(__name__)

Calibration = Literal["sigma", "gamma", "beta"]
Polarization = Literal["VV", "VH", "HV", "HH"]


@dataclass(frozen=True)
class Params:
    polarizations: list[Polarization]
    calibration: Optional[Calibration]
    orthorectify: Literal[True]
    rtc: None
    filtering: None


class InputProduct(abc.ABC):
    @abc.abstractmethod
    def into_product_info(self) -> sentinel1.product.Sentinel1GRDProductInfo: ...


@dataclass(frozen=True)
class PhoenixInputProduct(InputProduct):
    item: Any
    """ Phoenix catalog Item from the GRD collection. """

    @override
    def into_product_info(self) -> sentinel1.product.Sentinel1GRDProductInfo:
        return sentinel1.product.PhoenixSentinel1GRDProductInfo(
            self.item,
            image_opener=eos.sar.io.open_image,
        )


@dataclass(frozen=True)
class ShapeTransformDestinationGeometry:
    """
    For users that know their target shape/transform/CRS.
    """

    shape: tuple[int, int]
    transform: rasterio.Affine
    crs: rasterio.CRS


@dataclass(frozen=True)
class BboxDestinationGeometry:
    """
    For users that are interested in a bbox defined in 4326.
    """

    bbox: tuple[float, float, float, float]
    resolution: float
    align: Optional[float]
    crs: Optional[rasterio.CRS]
    """ If not set, pick the best UTM zone. """


@dataclass(frozen=True)
class FromImageRoiDestinationGeometry:
    """
    For users that know beforehand which is the ROI of interest in the image.
    """

    roi: Roi
    resolution: float
    align: Optional[float]
    crs: Optional[rasterio.CRS]
    """ If not set, pick the best UTM zone. """


DestinationGeometry = Union[
    ShapeTransformDestinationGeometry,
    BboxDestinationGeometry,
    FromImageRoiDestinationGeometry,
]


@dataclass(frozen=True)
class FilesystemResultDestination:
    paths: dict[Polarization, Path]


@dataclass
class MemoryResultDestination:
    """
    Note: this is a mutable object, to be instanciated by users with 'make_empty'.
    It can be read after going through the `process` function.
    """

    arrays: dict[Polarization, NDArray[np.float32]]
    _profile: dict[str, Any]

    @staticmethod
    def make_empty() -> MemoryResultDestination:
        return MemoryResultDestination(arrays={}, _profile={})

    @property
    def crs(self) -> rasterio.CRS:
        return self._profile["crs"]

    @property
    def transform(self) -> rasterio.Affine:
        return self._profile["transform"]

    @property
    def nodata(self) -> float:
        return self._profile["nodata"]

    @property
    def rasterio_profile(self) -> dict[str, Any]:
        return self._profile


ResultDestination = Union[FilesystemResultDestination, MemoryResultDestination]


@dataclass(frozen=True)
class CropperInput:
    products: list[InputProduct]
    """GRD products from the same datatake"""
    destination_geometry: DestinationGeometry
    params: Params
    result_destination: ResultDestination

    dem_source: eos.dem.DEMSource
    orbit_catalog_backend: orbit_catalog.Sentinel1OrbitCatalogBackend


def get_cdse_orbit_catalog_backend(
    username: str, password: str
) -> orbit_catalog.Sentinel1OrbitCatalogBackend:
    return orbit_catalog.CDSESentinel1OrbitCatalogBackend(username, password)


def get_phoenix_orbit_catalog_backend(
    client: Any,
) -> orbit_catalog.Sentinel1OrbitCatalogBackend:
    return orbit_catalog.PhoenixSentinel1OrbitCatalogBackend(
        collection_source=client.get_collection("esa-sentinel-1-csar-aux").at(
            "aws:proxima:kayrros-prod-sentinel-aux"
        )
    )


def geom_to_roi(geometry, proj_model: SensorModel, dem: eos.dem.DEM) -> Roi:
    geom_coords = geometry.exterior.coords[:]
    lons = [c[0] for c in geom_coords]
    lats = [c[1] for c in geom_coords]
    alts = np.nan_to_num(dem.elevation(lons, lats))
    rows, cols, _ = proj_model.projection(lons, lats, alts)
    roi = Roi.from_bounds_tuple(Roi.points_to_bbox(rows, cols))
    return roi


def _utm_zone_of_bbox(bbox: tuple[float, float, float, float]) -> rasterio.CRS:
    zone = int(((bbox[0] + bbox[2]) / 2 + 180) // 6 + 1)
    const = 32600 if bbox[1] + bbox[3] > 0 else 32700
    epsg = const + zone
    return rasterio.CRS.from_epsg(epsg)


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
        raise Exception("AlignmentError")

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


# TODO: check potential failure cases (invalid AOI vs. product footprint, I/O error, ...)
def process(input: CropperInput) -> None:
    # TODO: support GRD assembly
    if len(input.products) != 1:
        raise Exception("CropperInput.products should contain only one element")
    product = input.products[0].into_product_info()
    product_id = product.product_id

    query = orbit_catalog.Sentinel1OrbitCatalogQuery(
        product_ids=[product_id],
        quality=orbit_catalog.BestEffort,
    )

    statevectors = orbit_catalog.search(input.orbit_catalog_backend, query).single()

    pol = input.params.polarizations[0]
    xml = product.get_xml_annotation(pol)
    meta = sentinel1.metadata.extract_grd_metadata(xml)
    if statevectors:
        meta = meta.with_new_state_vectors(statevectors, "")
    else:
        logger.warn(
            f"couldn't find orbit file for {product_id}, continuing with the product metadata"
        )

    orbit = eos.sar.orbit.Orbit(meta.state_vectors)
    corr = [
        eos.sar.atmospheric_correction.ApdCorrection(orbit),
    ]
    corrector = eos.sar.projection_correction.Corrector(corr)

    proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit, corrector)

    roi: Roi
    dst_geom = input.destination_geometry
    if isinstance(dst_geom, FromImageRoiDestinationGeometry):
        roi = dst_geom.roi
        dem = proj_model.fetch_dem(input.dem_source, roi)
        orthorectifier = Orthorectifier.from_roi(
            proj_model,
            roi,
            resolution=dst_geom.resolution,
            dem=dem,
            crs=dst_geom.crs,
            align=dst_geom.align,
        )
    else:
        bbox: tuple[float, float, float, float]
        if isinstance(dst_geom, BboxDestinationGeometry):
            bbox = dst_geom.bbox
            crs = dst_geom.crs
            if crs is None:
                crs = _utm_zone_of_bbox(bbox)

            transform, shape, _ = _compute_transform_shape(
                crs, dst_geom.resolution, bbox, dst_geom.align
            )
        elif isinstance(dst_geom, ShapeTransformDestinationGeometry):
            transform = dst_geom.transform
            shape = dst_geom.shape
            crs = dst_geom.crs

            # compute a bbox, used for fetching the DEM and identifying a ROI
            bbox = rasterio.transform.array_bounds(*dst_geom.shape, dst_geom.transform)
            bbox = rasterio.warp.transform_bounds(crs, "epsg:4326", *bbox)
        else:
            assert_never(dst_geom)

        geometry = shapely.geometry.box(*bbox)

        # download a dem, with a large buffer to ensure we can use localize_without_alt
        # for example, 0.5 is half the tile size of GLO30, this make sure we have plenty of DEM available
        dem = input.dem_source.fetch_dem(geometry.buffer(0.5).bounds)

        # estimate a Roi for the requested geometry
        roi = geom_to_roi(geometry, proj_model, dem)

        orthorectifier = Orthorectifier.from_transform(
            proj_model,
            roi,
            transform=transform,
            shape=shape,
            dem=dem,
            crs=crs,
        )

    for pol in input.params.polarizations:
        reader = product.get_image_reader(pol)

        if cal := input.params.calibration:
            cal_xml = product.get_xml_calibration(pol)
            noise_xml = product.get_xml_noise(pol)
            ipf = product.ipf
            calibrator = sentinel1.calibration.Sentinel1Calibrator(
                cal_xml, noise_xml, ipf
            )
            reader = sentinel1.calibration.CalibrationReader(
                reader, calibrator, method=cal
            )

        raster = eos.sar.io.read_window(
            reader, roi, get_complex=False, out_dtype=np.float32, boundless=True
        )
        mask = sentinel1.border_noise_grd.compute_border_mask(raster)
        raster = sentinel1.border_noise_grd.apply_border_mask(raster, mask)

        raster = orthorectifier.apply(raster, eos.sar.ortho.LanczosInterpolation)

        assert len(raster.shape) == 2
        assert raster.dtype == np.float32
        profile = dict(
            driver="GTiff",
            width=raster.shape[1],
            height=raster.shape[0],
            count=1,
            dtype=raster.dtype,
            nodata=np.nan,
            crs=orthorectifier.crs,
            transform=orthorectifier.transform,
        )

        storage = input.result_destination
        if isinstance(storage, FilesystemResultDestination):
            output = storage.paths[pol]
            with rasterio.open(output, "w+", **profile) as dst:
                dst.write(raster, 1)
        elif isinstance(storage, MemoryResultDestination):
            storage.arrays[pol] = raster
            storage._profile = profile
        else:
            assert_never(storage)
