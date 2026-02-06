import json

import pytest

from eos.products.nisar import metadata
from eos.sar.io import RemoteH5Loader

NISAR_RSLC_SAMPLE_PATHS = [
    "s3://kayrros-dev-satellite-test-data/NISAR/simulated_samples/l1_rslc/sample1/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5",
    "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5",
]


def rslc_meta_from_h5(h5_path: str) -> metadata.NisarRSLCMetadata:
    with RemoteH5Loader(h5_path) as ds:
        meta = metadata.NisarRSLCMetadata.parse_metadata(ds)
    return meta


@pytest.mark.parametrize("h5_path", NISAR_RSLC_SAMPLE_PATHS)
def test_rslc_meta_from_h5(h5_path: str):
    meta = rslc_meta_from_h5(h5_path)
    assert isinstance(meta, metadata.NisarRSLCMetadata)

    assert meta.radar_band == "L"
    assert meta.look_side == "right"
    assert meta.height == 19760

    meta_dict = meta.to_dict()
    meta_from_dict = metadata.NisarRSLCMetadata.from_dict(meta_dict)
    assert meta == meta_from_dict

    meta_from_dict_bis = metadata.NisarRSLCMetadata.from_dict(
        json.loads(json.dumps(meta_dict))
    )

    assert meta_from_dict_bis == meta
