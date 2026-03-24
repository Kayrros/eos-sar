import os

import boto3
import pytest


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            "cdse",
            marks=[pytest.mark.cdse, pytest.mark.flaky(reruns=5, reruns_delay=30)],
        )
    ],
)
def cdse_auth(maybe_skip_cdse):
    username = os.environ["CDSE_USERNAME"]
    password = os.environ["CDSE_PASSWORD"]
    return (username, password)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            "cdse",
            marks=[pytest.mark.cdse, pytest.mark.flaky(reruns=5, reruns_delay=30)],
        )
    ],
)
def cdse_s3_session(maybe_skip_cdse_s3):
    return boto3.Session(
        aws_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
    )


@pytest.fixture(scope="session")
def maybe_skip_cdse():
    if ("CDSE_USERNAME" not in os.environ) or ("CDSE_PASSWORD" not in os.environ):
        pytest.skip(
            "skipped because CDSE credentials are not found (CDSE_USERNAME, CDSE_PASSWORD)"
        )


@pytest.fixture(scope="session")
def maybe_skip_cdse_s3():
    if ("CDSE_ACCESS_KEY_ID" not in os.environ) or (
        "CDSE_SECRET_ACCESS_KEY" not in os.environ
    ):
        pytest.skip(
            "skipped because CDSE S3 credentials are not found (CDSE_ACCESS_KEY_ID, CDSE_SECRET_ACCESS_KEY)"
        )
