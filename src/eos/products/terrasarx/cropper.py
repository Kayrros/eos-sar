import logging
import os
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray

from eos.dem import DEM, DEMSource
from eos.products.terrasarx.metadata import TSXMetadata, parse_tsx_metadata
from eos.products.terrasarx.model import TSXModel
from eos.sar.atmospheric_correction import ApdCorrection
from eos.sar.io import open_image, read_window
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector
from eos.sar.regist import (
    apply_affine,
    change_resamp_mat_orig,
    get_registration_dem_pts,
    orbital_registration,
)
from eos.sar.roi import Roi
from eos.sar.roi_provider import RoiProvider

logger = logging.getLogger(__name__)

ArrayRealCmpx = NDArray[Union[np.float32, np.complex64]]


def pid_from_xml_path(xml_path: str) -> str:
    return os.path.splitext(os.path.basename(xml_path))[0]


@dataclass(frozen=True)
class TSXCrop:
    product_id: str
    model: TSXModel
    meta: TSXMetadata
    array: ArrayRealCmpx
    """Resampled array. No resampling for primary array"""
    roi: Roi
    """roi for current image, i.e. primary_roi only for primary_image"""
    resampling_matrix: NDArray[np.float64]
    """ 3x3 matrix. (Identity for the primary product) from primary_roi to secondary_roi"""


# TODO separate in PrimaryPipeline and SecondaryPipeline
# TODO implement a TSX product which gives xmlcontent and reader, instead of inputting paths
def crop_images(
    xml_metadata_files: list[str],
    raster_paths: list[str],
    primary_id: int,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex=True,
) -> tuple[list[TSXCrop], DEM]:
    backgeocoding_reference = xml_metadata_files[primary_id]

    primary_metadata = parse_tsx_metadata(backgeocoding_reference)
    primary_orbit = Orbit(sv=primary_metadata.state_vectors, degree=11)
    """
    TODO explore more corrections ?, apd by itself did not help a lot with the jitter 
    there is not a big altitude change in the scene, jitter should not be due to prallax
    From a quick read of 
    "In-Depth Verification of Sentinel-1 and TerraSAR-X Geolocation Accuracy
    Using the Australian Corner Reflector Array"
    TSX should not need any correction besides APD (tropo + to some extent iono)
    negligible alt_fm_mismatch
    and otherwise tidal effects
    """
    primary_corrector = Corrector([ApdCorrection(primary_orbit)])
    primary_model = TSXModel.from_metadata(
        primary_metadata, primary_orbit, primary_corrector
    )

    primary_roi, _, _ = roi_provider.get_roi(primary_model, dem_source)

    # TODO Not sure if necessary, I follow here teosar strategy of first downloading a dem for roi
    # then re-defining the bounds and downloading a dem again
    dem = primary_model.fetch_dem(dem_source, roi=primary_roi)

    # Read primary and crop
    primary_raster_path = raster_paths[primary_id]

    primary_product_id = pid_from_xml_path(backgeocoding_reference)
    logger.info(f"Processing {primary_product_id}")

    # TODO implement and use a calibration reader
    primary_reader = open_image(primary_raster_path)
    primary_array = read_window(
        primary_reader, primary_roi, get_complex=get_complex, boundless=True
    )

    primary_crop = TSXCrop(
        primary_product_id,
        primary_model,
        primary_metadata,
        primary_array,
        primary_roi,
        resampling_matrix=np.eye(3, dtype=np.float64),
    )

    x_sampled, y_sampled, raster_sampled, crs = get_registration_dem_pts(
        primary_model, primary_roi, sampling_ratio=dem_sampling_ratio, dem=dem
    )
    row_primary, col_primary, _ = primary_model.projection(
        x_sampled, y_sampled, raster_sampled
    )

    crops = []
    for secondary_xml, secondary_raster_path in zip(xml_metadata_files, raster_paths):
        # skip primary image
        if secondary_xml == backgeocoding_reference:
            continue
        secondary_product_id = pid_from_xml_path(secondary_xml)

        logger.info(f"Processing {secondary_product_id}")
        secondary_metadata = parse_tsx_metadata(secondary_xml)
        secondary_orbit = Orbit(sv=secondary_metadata.state_vectors, degree=11)
        secondary_corrector = Corrector([ApdCorrection(secondary_orbit)])
        secondary_model = TSXModel.from_metadata(
            secondary_metadata, secondary_orbit, secondary_corrector
        )

        # origin of affinity here is the full image origin
        A_init = orbital_registration(
            row_primary,
            col_primary,
            secondary_model,
            x_sampled,
            y_sampled,
            raster_sampled,
            crs,
        )

        # transform roi into secondary and add hardcoded margin
        roi_in_secondary = primary_roi.warp(A_init).add_margin(50)

        # Change origins
        col_dst, row_dst = primary_roi.get_origin()
        col_src, row_src = roi_in_secondary.get_origin()
        A_crop = change_resamp_mat_orig(row_dst, col_dst, row_src, col_src, A_init)

        # Read
        secondary_reader = open_image(secondary_raster_path)
        secondary_array = read_window(
            secondary_reader, roi_in_secondary, get_complex=get_complex, boundless=True
        )
        # TODO deramp, reramp

        # resample
        secondary_resampled = apply_affine(
            secondary_array, A_crop, primary_roi.get_shape()
        )

        secondary_crop = TSXCrop(
            secondary_product_id,
            secondary_model,
            secondary_metadata,
            secondary_resampled,
            roi_in_secondary,
            A_crop,
        )

        crops.append(secondary_crop)
        # TODO refine the registration with image based alignement

    crops.insert(primary_id, primary_crop)

    return (crops, dem)
