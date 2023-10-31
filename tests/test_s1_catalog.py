import datetime

import pytest
import shapely.geometry

from eos.products.sentinel1.catalog import (
    CDSESentinel1CatalogBackend,
    PhoenixSentinel1CatalogBackend,
    Sentinel1Catalog,
    Sentinel1CatalogQuery,
)

try:
    import phoenix.catalog as phx

    has_phx = True
except ModuleNotFoundError:
    has_phx = False


query = Sentinel1CatalogQuery(
    geometry=shapely.geometry.Point(-68.374028, -23.563574),
    relative_orbit_number=149,
    start_date=datetime.datetime(2019, 1, 1),
    end_date=datetime.datetime(2019, 4, 1),
    polarization=["SV", "DV"],
)

expected = [
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


@pytest.mark.skipif(not has_phx, reason="phoenix not installed")
def test_phx_catalog():
    client = phx.Client()
    collection = client.get_collection("esa-sentinel-1-csar-l1-slc").at(
        "asf:daac:sentinel-1"
    )
    catalog = Sentinel1Catalog(
        backend=PhoenixSentinel1CatalogBackend(collection_source=collection)
    )
    result = catalog.search_slc(query)
    assert result.product_ids == expected


def test_cdse_catalog():
    catalog = Sentinel1Catalog(backend=CDSESentinel1CatalogBackend())
    result = catalog.search_slc(query)
    assert result.product_ids == expected
