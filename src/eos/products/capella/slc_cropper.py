"""
The code in this module is heavily based on the eos.products.terrasarx.cropper module.
TODO: Eventually, one could imagine moving some of the common code to eos.sar.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import rasterio
from numpy.typing import NDArray

from eos.dem import DEM, DEMSource
from eos.products.capella.doppler_info import CapellaDoppler
from eos.products.capella.metadata import (
    CapellaSLCMetadata,
    parse_metadata,
)
from eos.products.capella.proj_model import CapellaSLCModel
from eos.products.capella.resampler import CapellaResample
from eos.sar.atmospheric_correction import ApdCorrection
from eos.sar.io import open_image, read_window
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector
from eos.sar.regist import (
    change_resamp_mat_orig,
    get_registration_dem_pts,
    orbital_registration,
    phase_correlation_on_amplitude,
    translation_matrix,
)
from eos.sar.roi import Roi
from eos.sar.roi_provider import RoiProvider

logger = logging.getLogger(__name__)


def meta_from_slc_tif(slc_tif_path: str) -> CapellaSLCMetadata:
    with rasterio.open(slc_tif_path, "r") as db:
        meta = parse_metadata(db.get_tag_item("TIFFTAG_IMAGEDESCRIPTION"))
    assert isinstance(meta, CapellaSLCMetadata)
    return meta


def proj_model_from_meta(
    metadata: CapellaSLCMetadata, use_apd: bool = True
) -> CapellaSLCModel:
    orbit = Orbit(sv=metadata.state_vectors, degree=11)

    corrections = [ApdCorrection(orbit)] if use_apd else []

    model = CapellaSLCModel.from_metadata(
        metadata,
        orbit,
        corrector=Corrector(corrections),
        max_iterations=20,
        tolerance=0.0001,
    )
    return model


def pid_from_tif_path(path: str) -> str:
    # take only basename
    basename = os.path.basename(path)
    # Capella product id should be 51 characters long
    # this line of code removes the .tif extension and any suffix that is added to the product id
    return basename[:51]


@dataclass(frozen=True)
class CapellaCrop:
    product_id: str
    model: CapellaSLCModel
    meta: CapellaSLCMetadata
    array: NDArray[Union[np.float32, np.complex64]]
    """Resampled array. No resampling for primary array"""
    roi: Roi
    """roi for current image, i.e. primary_roi only for primary_image"""
    resampling_matrix: NDArray[np.float64]
    """ 3x3 matrix. (Identity for the primary product) from primary_roi to secondary_roi"""
    translation: tuple[float, float] = (0.0, 0.0)
    """
    image based translation
    """


def get_primary_crop(
    primary_raster_path: str,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    calibrate: bool = True,
) -> CapellaCrop:
    primary_metadata = meta_from_slc_tif(primary_raster_path)
    primary_model = proj_model_from_meta(primary_metadata, use_apd=use_apd)

    primary_roi, _, _ = roi_provider.get_roi(primary_model, dem_source)

    # Read primary and crop
    primary_product_id = pid_from_tif_path(primary_raster_path)
    logger.info(f"Processing {primary_product_id}")

    # TODO implement and use a calibration reader
    primary_reader = open_image(primary_raster_path)
    primary_array = read_window(
        primary_reader, primary_roi, get_complex=get_complex, boundless=True
    )
    if calibrate:
        primary_array = primary_array * primary_metadata.scale_factor

    primary_crop = CapellaCrop(
        primary_product_id,
        primary_model,
        primary_metadata,
        primary_array,
        primary_roi,
        resampling_matrix=np.eye(3, dtype=np.float64),
    )

    return primary_crop


@dataclass(frozen=True)
class RegistrationLUT:
    """
    Some 3D coords sampled on the DEM with their CRS
    and their 2D coords in the image. All arrays should have the same shape, and
    are intended to be 1D.
    """

    x_sampled: NDArray[np.float64]
    y_sampled: NDArray[np.float64]
    raster_sampled: NDArray[np.float64]
    crs: str
    row: NDArray[np.float64]
    col: NDArray[np.float64]

    def __post_init__(self):
        array_shapes = [
            self.x_sampled.shape,
            self.y_sampled.shape,
            self.raster_sampled.shape,
            self.row.shape,
            self.col.shape,
        ]

        # both conditions are equivalent to having all arrays of equal shapes and 1D
        assert len(array_shapes[0]) == 1
        assert [arr == array_shapes[0] for arr in array_shapes[1:]]


def get_primary_registLUT(
    primary_model: CapellaSLCModel,
    primary_roi: Roi,
    dem: DEM,
    dem_sampling_ratio: float = 0.3,
) -> RegistrationLUT:
    x_sampled, y_sampled, raster_sampled, crs = get_registration_dem_pts(
        primary_model, primary_roi, sampling_ratio=dem_sampling_ratio, dem=dem
    )
    row_primary, col_primary, _ = primary_model.projection(
        x_sampled, y_sampled, raster_sampled, crs=crs
    )
    return RegistrationLUT(
        x_sampled, y_sampled, raster_sampled, crs, row_primary, col_primary
    )


def get_primary_crop_dem_registLUT(
    primary_raster_path: str,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    calibrate: bool = True,
) -> tuple[CapellaCrop, DEM, RegistrationLUT]:
    primary_crop = get_primary_crop(
        primary_raster_path,
        roi_provider,
        dem_source,
        get_complex=get_complex,
        use_apd=use_apd,
        calibrate=calibrate,
    )

    dem = primary_crop.model.fetch_dem(dem_source, roi=primary_crop.roi)

    primary_registration_LUT = get_primary_registLUT(
        primary_crop.model, primary_crop.roi, dem, dem_sampling_ratio
    )

    return primary_crop, dem, primary_registration_LUT


def get_secondary_crop(
    secondary_raster_path: str,
    primary_roi: Roi,
    primary_registration_LUT: RegistrationLUT,
    primary_array: Optional[NDArray[Union[np.float32, np.complex64]]] = None,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    calibrate: bool = True,
) -> CapellaCrop:
    """
    Important ! registration refinement !
        if primary_array is provided (not None):
        -> the registration will be refined with image based alignement.
    """

    secondary_product_id = pid_from_tif_path(secondary_raster_path)
    logger.info(f"Processing {secondary_product_id}")
    secondary_metadata = meta_from_slc_tif(secondary_raster_path)
    secondary_model = proj_model_from_meta(secondary_metadata, use_apd=use_apd)

    # origin of affinity here is the full image origin
    A_init = orbital_registration(
        primary_registration_LUT.row,
        primary_registration_LUT.col,
        secondary_model,
        primary_registration_LUT.x_sampled,
        primary_registration_LUT.y_sampled,
        primary_registration_LUT.raster_sampled,
        primary_registration_LUT.crs,
    )

    # transform roi into secondary and add hardcoded margin
    # the margin is big enough to allow registration potential
    # registration refinement (additional translation)
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
    if calibrate:
        secondary_array = secondary_array * secondary_metadata.scale_factor

    # create a resampler
    doppler = CapellaDoppler.from_metadata(secondary_metadata)

    resampler = CapellaResample(
        A_crop, roi_in_secondary, primary_roi.get_shape(), doppler
    )

    # resample
    secondary_resampled = resampler.resample(secondary_array)

    if primary_array is not None:
        tcol, trow = phase_correlation_on_amplitude(
            np.abs(primary_array), np.abs(secondary_resampled)
        )
        A = translation_matrix(-tcol, -trow)

        # resample again
        # by adapting the resampling matrix with the translation
        resampler = CapellaResample(
            A.dot(A_crop), roi_in_secondary, primary_roi.get_shape(), doppler
        )
        secondary_resampled = resampler.resample(secondary_array)

        translation = (-tcol, -trow)
    else:
        translation = (0.0, 0.0)
    secondary_crop = CapellaCrop(
        secondary_product_id,
        secondary_model,
        secondary_metadata,
        secondary_resampled,
        roi_in_secondary,
        A_crop,
        translation,
    )

    return secondary_crop


# TODO implement a Capella product which gives jsoncontent and reader, instead of inputing paths
def crop_images(
    raster_paths: list[str],
    primary_id: int,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    refine_regist: bool = True,
    calibrate: bool = True,
) -> tuple[list[CapellaCrop], DEM]:
    """
    Crop images and align with a primary image. A DEM covering the images is also returned.
    Basic implementation intended for small aois and a limited number of images for memory considerations:
        Indeed, it stacks the results in a list for each date and returns it in the end.
    The arrays are treated sequentially with no parallelism.
    For heavier inputs, consider adapting this function to store the result of each array on the disk.
    """
    primary_raster_path = raster_paths[primary_id]
    primary_crop, dem, primary_registration_LUT = get_primary_crop_dem_registLUT(
        primary_raster_path,
        roi_provider,
        dem_source,
        dem_sampling_ratio,
        get_complex=get_complex,
        use_apd=use_apd,
        calibrate=calibrate,
    )

    crops = []
    for i, secondary_raster_path in enumerate(raster_paths):
        # skip primary image
        if i == primary_id:
            continue
        secondary_crop = get_secondary_crop(
            secondary_raster_path,
            primary_crop.roi,
            primary_registration_LUT,
            primary_array=primary_crop.array if refine_regist else None,
            get_complex=get_complex,
            use_apd=use_apd,
            calibrate=calibrate,
        )

        crops.append(secondary_crop)

    crops.insert(primary_id, primary_crop)

    return (crops, dem)
