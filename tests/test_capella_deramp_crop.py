import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import shape

import eos.sar
from eos.products.capella.doppler_info import CapellaDoppler
from eos.products.capella.metadata import (
    CapellaSLCMetadata,
    parse_metadata,
)
from eos.products.capella.resampler import CapellaResample
from eos.products.capella.slc_cropper import crop_images
from eos.sar.dem_to_radar import dem_radarcoding
from eos.sar.io import read_window
from eos.sar.regist import apply_affine
from eos.sar.roi import Roi
from eos.sar.roi_provider import GeometryRoiProvider

PREFIX = "https://capella-open-data.s3.amazonaws.com"
TIF_FILES = [
    f"{PREFIX}/data/2022/8/30/CAPELLA_C08_SM_SLC_HH_20220830054712_20220830054716/CAPELLA_C08_SM_SLC_HH_20220830054712_20220830054716.tif",
    f"{PREFIX}/data/2022/9/16/CAPELLA_C08_SM_SLC_HH_20220916054645_20220916054649/CAPELLA_C08_SM_SLC_HH_20220916054645_20220916054649.tif",
]


@pytest.mark.parametrize("tif_path", TIF_FILES)
def test_deramping_reramping(tif_path: str):
    reader = rasterio.open(tif_path, "r")
    json_content_str = reader.get_tag_item("TIFFTAG_IMAGEDESCRIPTION")
    meta = parse_metadata(json_content_str)
    assert isinstance(meta, CapellaSLCMetadata)
    doppler = CapellaDoppler.from_metadata(meta)

    h, w = 100, 100

    # create an affine matrix
    trf = (
        Affine.rotation(45, pivot=(h / 2, w / 2))
        * Affine.shear(20)
        * Affine.translation(5, -3)
    )
    inverse_mat = np.array(trf).reshape((3, 3))

    # Roi for reading
    src_roi_in_dop_frame = Roi(500, 800, w, h)
    dst_roi = src_roi_in_dop_frame

    array = read_window(reader, src_roi_in_dop_frame, get_complex=True)
    reader.close()
    assert array.dtype == np.complex64
    array = array.astype(np.complex64)  # for mypy

    resampler = CapellaResample(
        inverse_mat, src_roi_in_dop_frame, dst_roi.get_shape(), doppler
    )

    deramping_phase = resampler.deramping_phase()

    # check deramping
    deramped = resampler.deramp(array)
    assert deramped.dtype == np.complex64
    assert np.all(deramped == array * np.exp(1j * deramping_phase).astype(np.complex64))

    # check reramping phase equal to (- resampled deramping phase)
    reramping_phase = resampler.reramping_phase()
    deramping_resampled = apply_affine(
        deramping_phase, inverse_mat, resampler.dst_shape
    )

    nanmask = np.isnan(deramping_resampled)
    np.testing.assert_allclose(
        -deramping_resampled[~nanmask], reramping_phase[~nanmask], atol=1e-2, rtol=1e-2
    )

    # Check that everything can work in one go
    resampled = resampler.resample(array)
    resampled_bis = apply_affine(deramped, inverse_mat, resampler.dst_shape) * np.exp(
        1j * reramping_phase
    ).astype(np.complex64)
    np.testing.assert_allclose(resampled, resampled_bis)


# copied from teosar utils
def estimate_corrections(primary_proj_model, roi, secondary_proj_model, heights):
    topo = eos.sar.geom_phase.TopoCorrection(
        primary_proj_model,
        [secondary_proj_model],
        grid_size=50,
        degree=7,
    )
    # predict flat earth
    flat_earth_phase = topo.flat_earth_image(roi)
    # predict topographic phase
    topo_phase = topo.topo_phase_image(heights, primary_roi=roi)

    return flat_earth_phase[0], topo_phase[0]


def test_crop_insar():
    raster_paths = TIF_FILES
    primary_id = 0
    secondary_id = 1

    context = {
        "type": "Polygon",
        "coordinates": [
            [
                [-119.183126349377375, 40.782904348630971],
                [-119.183795737679915, 40.781479053021485],
                [-119.180878352643091, 40.780899218036019],
                [-119.180354965814175, 40.782619814069278],
                [-119.183126349377375, 40.782904348630971],
            ]
        ],
    }

    geometry = shape(context)
    roi_provider = GeometryRoiProvider(geometry, min_width=0, min_height=0)

    dem_source = eos.dem.DEMStitcherSource()
    dem_sampling_ratio = 0.2

    get_complex = True
    use_apd = True
    refine_regist = True

    (crops, dem) = crop_images(
        raster_paths,
        primary_id,
        roi_provider,
        dem_source,
        dem_sampling_ratio,
        get_complex=get_complex,
        use_apd=use_apd,
        refine_regist=refine_regist,
    )

    primary_model = crops[primary_id].model
    roi = crops[primary_id].roi

    # radarcode the dem
    heights = dem_radarcoding(dem, primary_model, roi, margin=100)

    # simulate flat and topo phase
    secondary_model = crops[secondary_id].model

    flat_earth_phase, topo_phase = estimate_corrections(
        primary_model, roi, secondary_model, heights
    )

    # do interferograms
    ifg = crops[primary_id].array * np.conj(crops[secondary_id].array)
    ifg = ifg * np.exp(-1j * (flat_earth_phase + topo_phase)).astype(np.complex64)
