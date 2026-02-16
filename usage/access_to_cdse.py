"""
uv run --env-file .env usage/access_to_cdse.py

The .env file shoudl contain:
    CDSE_ACCESS_KEY_ID
    CDSE_SECRET_ACCESS_KEY
"""

import os

import boto3

from eos.products.sentinel1.catalog import CDSESentinel1SLCCatalogBackend
from eos.products.sentinel1.product import CDSEUnzippedSafeSentinel1SLCProductInfo

session = boto3.Session(
    aws_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
)

product_id = "S1A_IW_SLC__1SDV_20230205T174135_20230205T174151_047104_05A6A7_AADA"

backend = CDSESentinel1SLCCatalogBackend()
product = CDSEUnzippedSafeSentinel1SLCProductInfo.from_product_id(
    cdse_backend=backend,
    s3_session=session,
    product_id=product_id,
)

assert (
    product.s3_path
    == "s3://eodata/Sentinel-1/SAR/SLC/2023/02/05/S1A_IW_SLC__1SDV_20230205T174135_20230205T174151_047104_05A6A7_AADA.SAFE"
)

print(len(product.get_xml_noise("IW3", "VV")))
print(product.get_image_reader("IW3", "VV").read(1, window=((2, 8), (10, 20))))
