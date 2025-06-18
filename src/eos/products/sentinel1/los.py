import numpy as np

from eos.products.sentinel1.burst_resamp import Sentinel1BurstResample
from eos.products.sentinel1.proj_model import Sentinel1MosaicModel
from eos.sar.geoconfig import (
    LOSPredictor,
    convert_arrays_to_enu,
    get_grid,
    get_los_squinted,
    localize_on_ellipsoid,
)
from eos.sar.model import Arrayf64
from eos.sar.roi import Roi
from eos.sar.utils import stitch_arrays


def get_points_3D_azt_rng(
    rows_roi: Arrayf64,
    cols_roi: Arrayf64,
    roi_in_mosaic: Roi,
    cropped_mosaic_model: Sentinel1MosaicModel,
    ellipsoid_alt: float,
) -> tuple[Arrayf64, Arrayf64, Arrayf64]:
    rows_in_mosaic = rows_roi + roi_in_mosaic.row
    cols_in_mosaic = cols_roi + roi_in_mosaic.col
    # 3D points on ellipsoid at Zero Doppler condition
    points_3D = localize_on_ellipsoid(
        cropped_mosaic_model, rows_in_mosaic, cols_in_mosaic, ellipsoid_alt
    )

    # coordinates in image to azt time and range
    azt, rng = cropped_mosaic_model.to_azt_rng(rows_in_mosaic, cols_in_mosaic)
    return points_3D, azt, rng


def get_dop_centroid_freq_delta_t(
    resampler_on_roi: Sentinel1BurstResample,
    rows_on_roi: Arrayf64,
    cols_on_roi: Arrayf64,
) -> tuple[Arrayf64, Arrayf64]:
    eta, ref_time, dop_centroid, dop_rate = resampler_on_roi.get_doppler_params(
        rows_on_roi,
        cols_on_roi,
        resampler_on_roi.src_roi_in_burst.get_origin(),
        resampler_on_roi.matrix,
    )

    dop_centroid_freq = dop_centroid + dop_rate * (eta - ref_time)
    alpha = resampler_on_roi.doppler.krot / dop_rate

    # approximate time between sensing and zero doppler
    delta_t = (eta - ref_time) * (1 / alpha - 1)

    return dop_centroid_freq, delta_t


def get_los_on_roi(
    roi_in_mosaic: Roi,
    resampler_on_roi: Sentinel1BurstResample,
    cropped_mosaic_model: Sentinel1MosaicModel,
    grid_size_col: int = 50,
    grid_size_row: int = 50,
    polynom_degree: int = 7,
    ellipsoid_alt: float = 0.0,
    normalized: bool = True,
    estimate_in_ENU: bool = False,
) -> Arrayf64:
    assert roi_in_mosaic.get_shape() == resampler_on_roi.dst_roi_in_burst.get_shape()

    # Get a meshgrid inside the roi_in_mosaic
    cols, rows = get_grid(
        roi_in_mosaic.w,
        roi_in_mosaic.h,
        grid_size_col=grid_size_col,
        grid_size_row=grid_size_row,
    )

    rows = rows.ravel()
    cols = cols.ravel()

    points_3D, azt, _ = get_points_3D_azt_rng(
        rows, cols, roi_in_mosaic, cropped_mosaic_model, ellipsoid_alt
    )

    # Get Doppler properties on grid as well
    doppler_centroid_freq, delta_t = get_dop_centroid_freq_delta_t(
        resampler_on_roi, rows, cols
    )

    # Compute LOS on grid
    los_squinted = get_los_squinted(
        points_3D,
        azt,
        cropped_mosaic_model.orbit,
        doppler_centroid_freq,
        cropped_mosaic_model.wavelength,
        delta_azt=delta_t,
        normalized=normalized,
    )

    if estimate_in_ENU:
        los_squinted = convert_arrays_to_enu(
            los_squinted, points_3D, ellipsoid_alt == 0
        )

    # Interpolate with a polynomial predictor
    los_predictor = LOSPredictor.from_los_grid_coords(
        los_squinted, rows, cols, degree=polynom_degree
    )
    los_on_roi = los_predictor.predict_los(
        np.arange(roi_in_mosaic.h), np.arange(roi_in_mosaic.w), grid_eval=True
    ).reshape((roi_in_mosaic.h, roi_in_mosaic.w, 3))
    return los_on_roi


def get_los_squinted_mosaic(
    rois_in_mosaic: dict[str, Roi],
    resamplers_on_roi: dict[str, Sentinel1BurstResample],
    mosaic_height: int,
    mosaic_width: int,
    mosaic_proj_model: Sentinel1MosaicModel,
    grid_size_col: int = 50,
    grid_size_row: int = 50,
    polynom_degree: int = 7,
    ellipsoid_alt: float = 0.0,
    *,
    normalized: bool = True,
    estimate_in_ENU: bool = False,
) -> Arrayf64:
    out_shape = (mosaic_height, mosaic_width, 3)
    out = np.full(out_shape, np.nan, dtype=np.float64)

    def gen():
        for bsid, roi_in_mosaic in rois_in_mosaic.items():
            los_on_roi = get_los_on_roi(
                roi_in_mosaic,
                resamplers_on_roi[bsid],
                mosaic_proj_model,
                grid_size_col=grid_size_col,
                grid_size_row=grid_size_row,
                polynom_degree=polynom_degree,
                ellipsoid_alt=ellipsoid_alt,
                normalized=normalized,
                estimate_in_ENU=estimate_in_ENU,
            )
            yield los_on_roi, roi_in_mosaic

    los_mosaic = stitch_arrays(gen(), out_shape, out)

    return los_mosaic


def get_los_ZeroDoppler_mosaic(
    mosaic_height: int,
    mosaic_width: int,
    mosaic_proj_model: Sentinel1MosaicModel,
    grid_size_col: int = 50,
    grid_size_row: int = 50,
    polynom_degree: int = 7,
    ellipsoid_alt: float = 0.0,
    *,
    normalized: bool = True,
    estimate_in_ENU: bool = False,
) -> Arrayf64:
    ZeroDoppler_los_predictor = LOSPredictor.from_proj_model_grid_size(
        mosaic_proj_model,
        grid_size_col,
        grid_size_row,
        degree=polynom_degree,
        alt=ellipsoid_alt,
        normalized=normalized,
        estimate_in_enu=estimate_in_ENU,
    )

    los_mosaic = ZeroDoppler_los_predictor.predict_los(
        np.arange(mosaic_height), np.arange(mosaic_width), grid_eval=True
    ).reshape(mosaic_height, mosaic_width, 3)

    return los_mosaic
