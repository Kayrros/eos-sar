import os

import pytest
import boto3
import numpy as np

from eos.products import sentinel1


@pytest.fixture
def client_s3():
    client_s3 = boto3.client("s3")
    return client_s3


def test_update_statevectors_using_our_bucket(client_s3):
    annotation = open('tests/data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml').read()
    date = '20190803T164007'
    mission = 'S1B'
    burst = sentinel1.metadata.extract_burst_metadata(annotation, burst_id=1)

    assert burst['state_vectors_origin'] == 'orbpre'
    assert sentinel1.orbits.update_statevectors_using_our_bucket(client_s3, (date, mission), burst) == 'orbpoe'
    assert burst['state_vectors_origin'] == 'orbpoe'


def test_update_statevectors_using_our_bucket2(client_s3):
    product_id = 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
    annotation = open(f'tests/data/{product_id}.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml').read()
    burst = sentinel1.metadata.extract_burst_metadata(annotation, burst_id=1)

    assert burst['state_vectors_origin'] == 'orbpre'
    assert sentinel1.orbits.update_statevectors_using_our_bucket(client_s3, product_id, burst) == 'orbpoe'
    assert burst['state_vectors_origin'] == 'orbpoe'


def test_update_statevectors_using_our_bucket2_manybursts(client_s3):
    product_id = 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
    annotation = open(f'tests/data/{product_id}.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml').read()
    bursts = sentinel1.metadata.extract_bursts_metadata(annotation)

    assert all(b['state_vectors_origin'] == 'orbpre' for b in bursts)
    assert sentinel1.orbits.update_statevectors_using_our_bucket(client_s3, product_id, bursts) == 'orbpoe'
    assert all(b['state_vectors_origin'] == 'orbpoe' for b in bursts)


def test_update_statevectors_using_our_bucket_forceres(client_s3):
    product_id = 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
    annotation = open(f'tests/data/{product_id}.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml').read()
    burst = sentinel1.metadata.extract_burst_metadata(annotation, burst_id=1)

    assert burst['state_vectors_origin'] == 'orbpre'
    assert sentinel1.orbits.update_statevectors_using_our_bucket(client_s3, product_id, burst, force_type='res') == 'orbres'
    assert sentinel1.orbits.update_statevectors_using_our_bucket(client_s3, product_id, burst, force_type='orbres') == 'orbres'
    assert burst['state_vectors_origin'] == 'orbres'


def test_update_statevectors_using_our_bucket_invalid(client_s3):
    # fake product, too old
    product_id = 'S1A_IW_SLC__1SDV_20120216T151206_20210216T151233_036617_044D40_8650'

    with pytest.raises(FileNotFoundError):
        sentinel1.orbits.update_statevectors_using_our_bucket(client_s3, product_id, {})
