from typing import Any

import numpy as np
import pytest
import shapely

from eos.dem import DEM, SRTM4Source
from eos.products.nisar.cropper import NisarCrop, crop_images
from eos.products.nisar.metadata import DatasetNotFoundError
from eos.sar.io import RemoteH5Loader
from eos.sar.regist import phase_correlation_on_amplitude
from eos.sar.roi import Roi
from eos.sar.roi_provider import GeometryRoiProvider, PrescribedRoiProvider

RSLC_SAMPLE_PATHS = [
    "s3://kayrros-dev-satellite-test-data/NISAR/simulated_samples/l1_rslc/sample1/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5",
    "s3://kayrros-dev-satellite-test-data/NISAR/simulated_samples/l1_rslc/sample2/NISAR_L1_PR_RSLC_002_030_A_019_2800_SHNA_A_20081127T061000_20081127T061014_D00404_N_F_J_001.h5",
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

    with pytest.raises(DatasetNotFoundError):
        cropper_input["polarization"] = "VV"  # not present in sample files
        crops, dem = crop_images(**cropper_input)
