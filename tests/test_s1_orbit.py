import pytest
import phoenix.catalog

from eos.products import sentinel1


@pytest.fixture
def phx_client():
    phx_client = phoenix.catalog.Client()
    return phx_client


def test_update_statevectors_using_phoenix(phx_client):
    annotation = open('tests/data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml').read()
    date = '20190803T164007'
    mission = 'S1B'
    burst = sentinel1.metadata.extract_burst_metadata(annotation, burst_id=1)

    assert burst.state_vectors_origin == 'orbpre'
    sv, orig = sentinel1.orbits.retrieve_statevectors_using_phoenix(phx_client, (date, mission), burst)
    assert orig == 'orbpoe'
    burst = burst.with_new_state_vectors(sv, orig)
    assert burst.state_vectors_origin == 'orbpoe'


def test_update_statevectors_using_phoenix2(phx_client):
    product_id = 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
    annotation = open(f'tests/data/{product_id}.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml').read()
    burst = sentinel1.metadata.extract_burst_metadata(annotation, burst_id=1)

    _, orig = sentinel1.orbits.retrieve_statevectors_using_phoenix(phx_client, product_id, burst)
    assert orig == 'orbpoe'


def test_update_statevectors_using_phoenix_forceres(phx_client):
    product_id = 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
    annotation = open(f'tests/data/{product_id}.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml').read()
    burst = sentinel1.metadata.extract_burst_metadata(annotation, burst_id=1)
    assert sentinel1.orbits.retrieve_statevectors_using_phoenix(phx_client, product_id, burst, force_type='res')[1] == 'orbres'
    assert sentinel1.orbits.retrieve_statevectors_using_phoenix(phx_client, product_id, burst, force_type='orbres')[1] == 'orbres'


def test_update_statevectors_using_phoenix_invalid(phx_client):
    # fake product, too old
    product_id = 'S1A_IW_SLC__1SDV_20120216T151206_20210216T151233_036617_044D40_8650'

    with pytest.raises(FileNotFoundError):
        sentinel1.orbits.retrieve_statevectors_using_phoenix(phx_client, product_id, {})


try:
    from eos.products.sentinel1.product import PhoenixSentinel1GRDProductInfo
except ImportError:
    pass
else:
    def test_grd_assemble_metadata(phx_client):
        product_id = 'S1A_IW_GRDH_1SDV_20220908T170044_20220908T170109_044916_055D72_82EF'
        product = PhoenixSentinel1GRDProductInfo.from_product_id(product_id)
        xml = product.get_xml_annotation('vv')
        meta = sentinel1.metadata.extract_grd_metadata(xml)

        assert sentinel1.orbits.retrieve_statevectors_using_phoenix(phx_client, product_id, meta)[1] == 'orbpoe'
