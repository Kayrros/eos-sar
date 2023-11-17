import pytest

from eos.products.sentinel1 import orbit_catalog
from eos.products.sentinel1.orbit_catalog import (
    BestEffort,
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


def test_phx_catalog_backend(phx_backend):
    product_id = "S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=BestEffort)
    result = orbit_catalog.search(phx_backend, query)
    assert result.single()


def test_phx_catalog_backend_noquality(phx_backend):
    product_id = "S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=[])
    result = orbit_catalog.search(phx_backend, query)
    assert result.single() is None


def test_phx_catalog_backend_invalid(phx_backend):
    # fake product, too old
    product_id = "S1A_IW_SLC__1SDV_20120216T151206_20210216T151233_036617_044D40_8650"
    query = Sentinel1OrbitCatalogQuery(product_ids=[product_id], quality=BestEffort)
    with pytest.raises(OrbitFileNotFound):
        orbit_catalog.search(phx_backend, query)
