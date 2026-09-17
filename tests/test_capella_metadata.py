import json
import os
from contextlib import nullcontext as does_not_raise
from typing import Union

import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.io import DatasetReader

from eos.products.capella import metadata, polynomial
from eos.sar.io import open_image, read_file_as_str


def test_meta_tifftags_vs_jsonfile():
    product_id = "CAPELLA_C02_SS_SLC_HH_20210204153042_20210204153058"

    s3_path = f"s3://capella-open-data/data/2021/2/4/{product_id}/"
    image_path = os.path.join(s3_path, f"{product_id}.tif")
    meta_json_path = os.path.join(s3_path, f"{product_id}_extended.json")

    db: DatasetReader = open_image(
        image_path, rasterio_session_kwargs={"aws_unsigned": True}
    )
    tags = db.tags()
    db.close()

    meta_dict_from_tags = json.loads(tags["TIFFTAG_IMAGEDESCRIPTION"])

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    json_content = read_file_as_str(meta_json_path, s3_client=s3)
    meta_dict_from_file = json.loads(json_content)

    assert meta_dict_from_file == meta_dict_from_tags

    meta_from_tiff = metadata.parse_metadata(tags["TIFFTAG_IMAGEDESCRIPTION"])
    assert isinstance(meta_from_tiff, metadata.CapellaSLCMetadata)

    meta_from_json = metadata.parse_metadata(json_content)
    assert isinstance(meta_from_json, metadata.CapellaSLCMetadata)

    assert meta_from_tiff == meta_from_json


SLC_META_PATHS = [
    "./tests/data/capella/CAPELLA_C13_SS_SLC_HH_20260819153732_20260819153747_extended.json",
    "./tests/data/capella/CAPELLA_C18_SM_SLC_HH_20260818222745_20260818222750_extended.json",
]

SLC_SPOT_PATH = "./tests/data/capella/CAPELLA_C20_SP_SLC_HH_20260823125202_20260823125209_extended.json"

GEC_META_PATHS = [
    "./tests/data/capella/CAPELLA_C13_SS_GEC_HH_20260819153732_20260819153747_extended.json",
    "./tests/data/capella/CAPELLA_C18_SM_GEC_HH_20260818222745_20260818222750_extended.json",
]

GEC_SPOT_PATH = "./tests/data/capella/CAPELLA_C20_SP_GEC_HH_20260823125202_20260823125209_extended.json"


def meta_from_capella_path(
    capella_path: str,
) -> Union[metadata.CapellaSLCMetadata, metadata.CapellaGECMetadata]:
    json_content = read_file_as_str(capella_path)
    meta_from_json = metadata.parse_metadata(json_content)
    return meta_from_json


@pytest.mark.parametrize(
    "meta_json_path,expected_meta_type,expectation",
    [
        (slc_path, metadata.CapellaSLCMetadata, does_not_raise())
        for slc_path in SLC_META_PATHS
    ]
    # geometry type pfa unsupported, seems that SP SLC products have pfa geometry
    + [
        (
            SLC_SPOT_PATH,
            metadata.CapellaSLCMetadata,
            pytest.raises(AssertionError, match="Unsupported image geometry type: pfa"),
        )
    ]
    + [
        (gec_path, metadata.CapellaGECMetadata, does_not_raise())
        for gec_path in GEC_META_PATHS
    ]
    # The SP gec product passes, because the exception is only raised for SLC products (here the geometry type is geotransform)
    + [(GEC_SPOT_PATH, metadata.CapellaGECMetadata, does_not_raise())],
)
def test_meta(meta_json_path: str, expected_meta_type, expectation):
    with expectation:
        meta = meta_from_capella_path(meta_json_path)
        assert isinstance(meta, expected_meta_type)


@pytest.mark.parametrize("slc_path", SLC_META_PATHS)
def test_capella_polynomial_creation(slc_path: str):
    meta = meta_from_capella_path(slc_path)
    assert isinstance(meta, metadata.CapellaSLCMetadata)
    # the post init has some asserts that should run
    polynomial.CapellaPolynomial2D.from_poly_meta(meta.fdop_cen_poly2d_meta)
