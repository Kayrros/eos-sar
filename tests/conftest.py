import os

import pytest


@pytest.fixture(scope="session")
def phx_client(maybe_skip_phx):
    import phoenix.catalog

    phx_client = phoenix.catalog.Client()
    return phx_client


@pytest.fixture(scope="session")
def cdse_auth(maybe_skip_cdse):
    username = os.environ["CDSE_USERNAME"]
    password = os.environ["CDSE_PASSWORD"]
    return (username, password)


try:
    import boto3

    _s3_client = boto3.client("s3")
    # the account requires read access to:
    # - s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/
    s = _s3_client.list_objects_v2(
        Bucket="kayrros-dev-satellite-test-data",
        Prefix="sentinel-1/eos_test_data/",
        MaxKeys=0,
    )
except Exception:
    _s3_client = None

    # disable multidem because it implicitly requires AWS credentials
    import eos.dem

    eos.dem.has_multidem = False


@pytest.fixture(scope="session")
def s3_client(maybe_skip_s3):
    return _s3_client


@pytest.fixture(scope="session")
def maybe_skip_cdse():
    if "CDSE_USERNAME" not in os.environ:
        pytest.skip("skipped because CDSE credentials are not found")


@pytest.fixture(scope="session")
def maybe_skip_phx():
    if "PHX_USERNAME" not in os.environ:
        pytest.skip("skipped because Phoenix credentials are not found")


@pytest.fixture(scope="session")
def maybe_skip_s3():
    if _s3_client is None:
        pytest.skip("skipped because AWS S3 credentials are not found")
