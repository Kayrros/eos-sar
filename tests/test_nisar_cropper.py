from typing import Any

import numpy as np
import pytest
import shapely

from eos.dem import DEM, SRTM4Source
from eos.products.nisar.cropper import (
    NisarCrop,
    crop_images,
    get_subswath_valid_mask,
)
from eos.products.nisar.metadata import DatasetNotFoundError, NisarRSLCMetadata
from eos.sar.io import RemoteH5Loader
from eos.sar.regist import phase_correlation_on_amplitude
from eos.sar.roi import Roi
from eos.sar.roi_provider import GeometryRoiProvider, PrescribedRoiProvider

RSLC_SAMPLE_PATHS = [
    "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5",
    "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_002_030_A_019_2800_SHNA_A_20081127T061000_20081127T061014_D00404_N_F_J_001/NISAR_L1_PR_RSLC_002_030_A_019_2800_SHNA_A_20081127T061000_20081127T061014_D00404_N_F_J_001.h5",
]


def test_cropper():
    geom = shapely.geometry.shape(
        {
            "coordinates": [
                [
                    [-118.07773384956742, 34.80618038625743],
                    [-118.07544854671534, 34.8061889740553],
                    [-118.07542762861146, 34.80472474158923],
                    [-118.07773384956742, 34.80469038978216],
                    [-118.07773384956742, 34.80618038625743],
                ]
            ],
            "type": "Polygon",
        }
    )
    cropper_input: dict[str, Any] = {
        "h5_loaders": [RemoteH5Loader(s3path) for s3path in RSLC_SAMPLE_PATHS],
        "primary_id": 0,
        "frequency": "A",
        "polarization": "HH",
        "roi_provider": GeometryRoiProvider(
            geometry=geom, min_height=500, min_width=500
        ),
        "dem_source": SRTM4Source(),
        "get_complex": False,
    }

    crops, dem = crop_images(**cropper_input)
    assert isinstance(dem, DEM)
    assert isinstance(crops, list)
    assert len(crops) == 2
    assert isinstance(crops[0], NisarCrop)
    assert isinstance(crops[1], NisarCrop)
    assert crops[0].array.shape == crops[1].array.shape
    assert abs(crops[1].translation[0]) < 0.15
    assert abs(crops[1].translation[1]) < 0.1
    assert crops[0].array.dtype == crops[1].array.dtype == np.float32
    for crop in crops:
        assert crop.valid_mask.shape == crop.array.shape
        assert crop.valid_mask.dtype == bool
        assert crop.valid_mask.any()
        # partially focused samples are discarded
        assert np.isnan(crop.array[~crop.valid_mask]).all()

    cropper_input["get_complex"] = True  # check that get_complex is working
    # also have roi exceed image limits to check for boundless behavior
    cropper_input["roi_provider"] = PrescribedRoiProvider(roi=Roi(-10, -10, 200, 300))
    crops, dem = crop_images(**cropper_input)
    assert crops[0].array.shape == (300, 200)
    assert crops[1].array.shape == (300, 200)
    assert np.isnan(
        crops[0].array[:10, :]
    ).all()  # check that the first 10 rows are nodata
    assert np.isnan(
        crops[0].array[:, :10]
    ).all()  # check that the first 10 cols are nodata
    # samples outside of the image are not valid
    assert not crops[0].valid_mask[:10, :].any()
    assert not crops[0].valid_mask[:, :10].any()
    for crop in crops:
        assert np.isnan(crop.array[~crop.valid_mask]).all()

    assert crops[0].array.dtype == crops[1].array.dtype == np.complex64

    # rerun offset computation
    tcol2, trow2 = phase_correlation_on_amplitude(
        crops[0].amplitude, crops[1].amplitude
    )
    # check that new offsets smaller than before
    assert abs(tcol2) < abs(crops[1].translation[0])
    assert abs(trow2) < abs(crops[1].translation[1])
    assert abs(tcol2) < 0.05
    assert abs(trow2) < 0.05

    # without masking, the mask is still returned and the primary valid samples
    # are unchanged (secondaries are not compared since the registration
    # refinement depends on the masked amplitude)
    crops_unmasked, _ = crop_images(**cropper_input, mask_partially_focused=False)
    assert crops_unmasked[1].valid_mask.shape == crops_unmasked[1].array.shape
    np.testing.assert_array_equal(crops[0].valid_mask, crops_unmasked[0].valid_mask)
    np.testing.assert_array_equal(
        crops[0].array[crops[0].valid_mask],
        crops_unmasked[0].array[crops[0].valid_mask],
    )

    with pytest.raises(DatasetNotFoundError):
        cropper_input["polarization"] = "VV"  # not present in sample files
        crops, dem = crop_images(**cropper_input)


def test_subswath_valid_mask():
    with RemoteH5Loader(RSLC_SAMPLE_PATHS[0]) as ds:
        meta = NisarRSLCMetadata.parse_metadata(ds)
    width = meta.frequency_a.width
    row_start = meta.height // 2
    starts, ends = meta.frequency_a.get_subswath_extents(row_start, row_start + 20)

    # full width of a few lines, exceeding the image on the right
    roi = Roi(0, row_start, width + 10, 20)
    mask = get_subswath_valid_mask(meta, "A", roi)
    assert mask.shape == roi.get_shape()
    assert mask.dtype == bool
    assert not mask[:, width:].any()
    assert mask.any()

    # the mask is the union of [start, end] of each subswath
    cols = np.arange(roi.w)
    for i in range(roi.h):
        expected = np.zeros(roi.w, dtype=bool)
        for start, end in zip(starts[:, i], ends[:, i]):
            expected |= (cols >= start) & (cols <= end)
        np.testing.assert_array_equal(mask[i], expected)

    # roi partially and completely outside of the image
    partial = get_subswath_valid_mask(meta, "A", Roi(-10, -10, 30, 30))
    assert not partial[:10].any() and not partial[:, :10].any()
    np.testing.assert_array_equal(
        partial[10:, 10:], get_subswath_valid_mask(meta, "A", Roi(0, 0, 20, 20))
    )
    assert not get_subswath_valid_mask(meta, "A", Roi(-50, -50, 20, 20)).any()
