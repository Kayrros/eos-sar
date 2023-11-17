import pytest

from eos.products.sentinel1 import orbit_catalog
from eos.products.sentinel1.orbit_catalog import (
    BestEffort,
    CDSESentinel1OrbitCatalogBackend,
    OrbitFileNotFound,
    PhoenixSentinel1OrbitCatalogBackend,
    Sentinel1OrbitCatalogQuery,
)


@pytest.fixture
def phx_backend(phx_client):
    backend = PhoenixSentinel1OrbitCatalogBackend(
        collection_source=phx_client.get_collection("esa-sentinel-1-csar-aux").at(
            "aws:proxima:kayrros-prod-sentinel-aux"
        )
    )
    return backend


@pytest.fixture
def cdse_backend(cdse_auth):
    backend = CDSESentinel1OrbitCatalogBackend(
        username=cdse_auth[0],
        password=cdse_auth[1],
    )
    return backend


def test_phx_catalog_backend_single(phx_backend):
    product_id = "S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=BestEffort)
    result = orbit_catalog.search(phx_backend, query)
    assert result.single()


def test_cdse_catalog_backend_single(cdse_backend):
    product_id = "S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=BestEffort)
    result = orbit_catalog.search(cdse_backend, query)
    assert result.single()


def test_phx_catalog_backend_many(phx_backend):
    product_ids = [
        "S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885",
        "S1A_IW_SLC__1SDV_20190110T230559_20190110T230627_025421_02D0E7_5EFE",
        "S1A_IW_SLC__1SDV_20190122T230559_20190122T230627_025596_02D74D_0EBA",
        "S1B_IW_SLC__1SDV_20190128T230512_20190128T230539_014700_01B682_D729",
        "S1A_IW_SLC__1SDV_20190203T230558_20190203T230626_025771_02DDA8_85BB",
        "S1B_IW_SLC__1SDV_20190209T230512_20190209T230539_014875_01BC41_A036",
        "S1A_IW_SLC__1SDV_20190215T230558_20190215T230622_025946_02E3DC_44B8",
        "S1B_IW_SLC__1SDV_20190221T230512_20190221T230539_015050_01C1FA_C1EF",
        "S1A_IW_SLC__1SDV_20190227T230558_20190227T230626_026121_02EA15_19A2",
        "S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14",
        "S1A_IW_SLC__1SDV_20190311T230558_20190311T230626_026296_02F06F_A19C",
        "S1B_IW_SLC__1SDV_20190317T230512_20190317T230539_015400_01CD6B_8DB9",
        "S1A_IW_SLC__1SDV_20190323T230558_20190323T230626_026471_02F6E6_C653",
        "S1B_IW_SLC__1SDV_20190329T230512_20190329T230539_015575_01D324_27D9",
    ]
    query = Sentinel1OrbitCatalogQuery(product_ids=product_ids, quality=BestEffort)
    result = orbit_catalog.search(phx_backend, query)
    for pid in product_ids:
        assert result.for_product_id(pid) is not None


def test_cdse_catalog_backend_many(cdse_backend):
    product_ids = [
        "S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885",
        "S1A_IW_SLC__1SDV_20190110T230559_20190110T230627_025421_02D0E7_5EFE",
        "S1A_IW_SLC__1SDV_20190122T230559_20190122T230627_025596_02D74D_0EBA",
        "S1B_IW_SLC__1SDV_20190128T230512_20190128T230539_014700_01B682_D729",
        "S1A_IW_SLC__1SDV_20190203T230558_20190203T230626_025771_02DDA8_85BB",
        "S1B_IW_SLC__1SDV_20190209T230512_20190209T230539_014875_01BC41_A036",
        "S1A_IW_SLC__1SDV_20190215T230558_20190215T230622_025946_02E3DC_44B8",
        "S1B_IW_SLC__1SDV_20190221T230512_20190221T230539_015050_01C1FA_C1EF",
        "S1A_IW_SLC__1SDV_20190227T230558_20190227T230626_026121_02EA15_19A2",
        "S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14",
        "S1A_IW_SLC__1SDV_20190311T230558_20190311T230626_026296_02F06F_A19C",
        "S1B_IW_SLC__1SDV_20190317T230512_20190317T230539_015400_01CD6B_8DB9",
        "S1A_IW_SLC__1SDV_20190323T230558_20190323T230626_026471_02F6E6_C653",
        "S1B_IW_SLC__1SDV_20190329T230512_20190329T230539_015575_01D324_27D9",
    ]
    query = Sentinel1OrbitCatalogQuery(product_ids=product_ids, quality=BestEffort)
    result = orbit_catalog.search(cdse_backend, query)
    for pid in product_ids:
        assert result.for_product_id(pid) is not None


def test_phx_catalog_backend_noquality(phx_backend):
    product_id = "S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=[])
    result = orbit_catalog.search(phx_backend, query)
    assert result.single() is None


def test_cdse_catalog_backend_noquality(cdse_backend):
    product_id = "S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=[])
    result = orbit_catalog.search(cdse_backend, query)
    assert result.single() is None


def test_phx_catalog_backend_invalid(phx_backend):
    # fake product, too old
    product_id = "S1A_IW_SLC__1SDV_20120216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=BestEffort)
    with pytest.raises(OrbitFileNotFound):
        orbit_catalog.search(phx_backend, query)


def test_cdse_catalog_backend_invalid(cdse_backend):
    # fake product, too old
    product_id = "S1A_IW_SLC__1SDV_20120216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=BestEffort)
    with pytest.raises(OrbitFileNotFound):
        orbit_catalog.search(cdse_backend, query)
