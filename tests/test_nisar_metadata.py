import json

import boto3

from eos.products.nisar import metadata
from eos.sar.io import open_netcdf_osio

NISAR_RSLC_SAMPLE_PATH = "s3://kayrros-dev-satellite-test-data/NISAR/simulated_samples/l1_rslc/sample1/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5"


def rslc_meta_from_h5(h5_s3_path: str) -> metadata.NisarRSLCMetadata:
    osio_options = {"session": boto3.session.Session()}
    with open_netcdf_osio(h5_s3_path, **osio_options) as ds:
        meta = metadata.NisarRSLCMetadata.parse_metadata(ds)
    return meta


def test_rslc_meta_from_h5():
    meta = rslc_meta_from_h5(NISAR_RSLC_SAMPLE_PATH)
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
