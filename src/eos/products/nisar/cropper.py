import logging
from dataclasses import dataclass
from typing import Literal, Optional, Union, cast

import numpy as np
from boto3.session import Session as S3Session
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from eos.dem import DEM, DEMSource
from eos.products.nisar.metadata import (
    DatasetNotFoundError,
    Frequency,
    NisarRSLCMetadata,
    Polarization,
)
from eos.products.nisar.proj_model import NisarModel
from eos.sar.atmospheric_correction import ApdCorrection
from eos.sar.io import open_netcdf_osio, read_hdf5_window
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector
from eos.sar.regist import (
    apply_affine,
    change_resamp_mat_orig,
    get_registration_dem_pts,
    orbital_registration,
    phase_correlation_on_amplitude,
    translation_matrix,
)
from eos.sar.roi import Roi
from eos.sar.roi_provider import RoiProvider

Calibration: TypeAlias = Literal["beta", "sigma", "gamma"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NisarCrop:
    product_id: str
    frequency: Frequency
    polarization: Polarization
    model: NisarModel
    meta: NisarRSLCMetadata
    array: NDArray[Union[np.float32, np.complex64]]
    roi: Roi
    resampling_matrix: NDArray[np.float64]
    translation: tuple[float, float] = (0.0, 0.0)

    @property
    def amplitude(self) -> NDArray[np.float32]:
        return get_amplitude(self.array)


def get_amplitude(
    array: NDArray[Union[np.float32, np.complex64]],
) -> NDArray[np.float32]:
    """
    The function checks if array is already floating and returns it unchanged.
    Otherwise, if dtype is complex, the numpy.abs function is used to get amplitude.
    """
    assert array.dtype in [np.float32, np.complex64], "Unrecognized array type"

    if array.dtype == np.float32:
        return cast(NDArray[np.float32], array)
    else:
        return np.abs(array)


def get_primary_crop(
    primary_h5_path: str,
    frequency: Frequency,
    polarization: Polarization,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    calibration: Optional[Calibration] = None,
    s3_session: Optional[S3Session] = None,
) -> NisarCrop:
    if primary_h5_path.startswith("s3://") and s3_session is None:
        logger.warning(f"No s3_session provided to read {primary_h5_path}")

    reader_options = (
        {"session": s3_session} if primary_h5_path.startswith("s3://") else {}
    )

    primary_ds = open_netcdf_osio(primary_h5_path, **reader_options)
    primary_metadata = NisarRSLCMetadata.parse_metadata(primary_ds)
    primary_product_id = primary_metadata.product_id
    logger.info(f"Processing {primary_product_id}")

    orbit = Orbit(sv=primary_metadata.state_vectors, degree=11)
    corrections = [ApdCorrection(orbit)] if use_apd else []
    primary_model = NisarModel.from_metadata(
        primary_metadata, frequency, orbit, corrector=Corrector(corrections)
    )
    primary_roi, _, _ = roi_provider.get_roi(primary_model, dem_source)

    dataset = f"science/LSAR/RSLC/swaths/frequency{frequency}/{polarization}"
    if dataset not in primary_ds.keys():
        raise DatasetNotFoundError(
            f"Dataset {dataset} not found in {primary_product_id}"
        )
    primary_array = read_hdf5_window(
        primary_ds[dataset], primary_roi, get_complex=get_complex, boundless=True
    )
    primary_ds.close()

    if calibration is not None:
        raise NotImplementedError("Calibration not implemented yet.")

    return NisarCrop(
        product_id=primary_product_id,
        frequency=frequency,
        polarization=polarization,
        model=primary_model,
        meta=primary_metadata,
        array=primary_array,
        roi=primary_roi,
        resampling_matrix=np.eye(3, dtype=np.float64),
    )


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
    primary_model: NisarModel,
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
    primary_h5_path: str,
    frequency: Frequency,
    polarization: Polarization,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    calibration: Optional[Calibration] = None,
    s3_session: Optional[S3Session] = None,
) -> tuple[NisarCrop, DEM, RegistrationLUT]:
    primary_crop = get_primary_crop(
        primary_h5_path,
        frequency,
        polarization,
        roi_provider,
        dem_source,
        get_complex=get_complex,
        use_apd=use_apd,
        calibration=calibration,
        s3_session=s3_session,
    )

    dem = primary_crop.model.fetch_dem(dem_source, roi=primary_crop.roi)

    primary_registration_LUT = get_primary_registLUT(
        primary_crop.model, primary_crop.roi, dem, dem_sampling_ratio
    )

    return primary_crop, dem, primary_registration_LUT


def get_secondary_crop(
    secondary_h5_path: str,
    frequency: Frequency,
    polarization: Polarization,
    primary_roi: Roi,
    primary_registration_LUT: RegistrationLUT,
    primary_array_amp: Optional[NDArray[np.float32]] = None,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    calibration: Optional[Calibration] = None,
    s3_session: Optional[S3Session] = None,
) -> NisarCrop:
    if secondary_h5_path.startswith("s3://") and s3_session is None:
        logger.warning(f"No s3_session provided to read {secondary_h5_path}")

    reader_options = (
        {"session": s3_session} if secondary_h5_path.startswith("s3://") else {}
    )

    secondary_ds = open_netcdf_osio(secondary_h5_path, **reader_options)
    secondary_metadata = NisarRSLCMetadata.parse_metadata(secondary_ds)
    secondary_product_id = secondary_metadata.product_id
    logger.info(f"Processing {secondary_product_id}")

    orbit = Orbit(sv=secondary_metadata.state_vectors, degree=11)
    corrections = [ApdCorrection(orbit)] if use_apd else []
    secondary_model = NisarModel.from_metadata(
        secondary_metadata, frequency, orbit, corrector=Corrector(corrections)
    )

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
    dataset = f"science/LSAR/RSLC/swaths/frequency{frequency}/{polarization}"
    if dataset not in secondary_ds.keys():
        raise DatasetNotFoundError(
            f"Dataset {dataset} not found in {secondary_product_id}"
        )
    secondary_array = read_hdf5_window(
        secondary_ds[dataset], roi_in_secondary, get_complex=get_complex, boundless=True
    )
    secondary_ds.close()

    if calibration is not None:
        raise NotImplementedError("Calibration not implemented yet.")

    if get_complex:
        logger.warning(
            "Proper complex data resampling is not fully implemented "
            "(Doppler centroid estimation and deramping)."
            "The data will be resampled anyway and the result may be "
            "imprecise depending on how big the Doppler centroid is."
        )
    # resample
    secondary_resampled = apply_affine(secondary_array, A_crop, primary_roi.get_shape())

    if primary_array_amp is not None:
        tcol, trow = phase_correlation_on_amplitude(
            primary_array_amp, get_amplitude(secondary_resampled)
        )
        A = translation_matrix(-tcol, -trow)

        # resample again
        # by adapting the resampling matrix with the translation
        secondary_resampled = apply_affine(
            secondary_array, A.dot(A_crop), primary_roi.get_shape()
        )

        translation = (-tcol, -trow)
    else:
        translation = (0.0, 0.0)

    return NisarCrop(
        secondary_product_id,
        frequency,
        polarization,
        secondary_model,
        secondary_metadata,
        secondary_resampled,
        roi_in_secondary,
        A_crop,
        translation,
    )


def crop_images(
    h5_paths: list[str],
    primary_id: int,
    frequency: Frequency,
    polarization: Polarization,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    refine_regist: bool = True,
    calibration: Optional[Calibration] = None,
    s3_session: Optional[S3Session] = None,
) -> tuple[list[NisarCrop], DEM]:
    """
    Crop images and align with a primary image. A DEM covering the images is also returned.
    Basic implementation intended for small aois and a limited number of images for memory considerations:
        Indeed, it stacks the results in a list for each date and returns it in the end.
    The arrays are treated sequentially with no parallelism.
    For heavier inputs, consider adapting this function to store the result of each array on the disk.
    """
    primary_h5_path = h5_paths[primary_id]
    primary_crop, dem, primary_registration_LUT = get_primary_crop_dem_registLUT(
        primary_h5_path,
        frequency,
        polarization,
        roi_provider,
        dem_source,
        dem_sampling_ratio,
        get_complex=get_complex,
        use_apd=use_apd,
        calibration=calibration,
        s3_session=s3_session,
    )

    crops = []
    for i, secondary_h5_path in enumerate(h5_paths):
        # skip primary image
        if i == primary_id:
            continue
        secondary_crop = get_secondary_crop(
            secondary_h5_path,
            frequency,
            polarization,
            primary_crop.roi,
            primary_registration_LUT,
            primary_array_amp=primary_crop.amplitude if refine_regist else None,
            get_complex=get_complex,
            use_apd=use_apd,
            calibration=calibration,
            s3_session=s3_session,
        )

        crops.append(secondary_crop)

    crops.insert(primary_id, primary_crop)

    return (crops, dem)
