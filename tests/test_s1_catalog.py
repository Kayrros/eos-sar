import datetime

import pytest
import shapely.geometry

from eos.products.sentinel1.catalog import (
    CDSESentinel1GRDCatalogBackend,
    CDSESentinel1SLCCatalogBackend,
    PhoenixSentinel1GRDCatalogBackend,
    PhoenixSentinel1SLCCatalogBackend,
    Sentinel1CatalogQuery,
    search_grd,
    search_slc,
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

expected_slc = [
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


expected_grd = [
    "S1B_IW_GRDH_1SDV_20190104T230514_20190104T230539_014350_01AB40_31F1",
    "S1A_IW_GRDH_1SDV_20190110T230600_20190110T230625_025421_02D0E7_7B47",
    "S1A_IW_GRDH_1SDV_20190122T230600_20190122T230625_025596_02D74D_9AC6",
    "S1B_IW_GRDH_1SDV_20190128T230513_20190128T230538_014700_01B682_3033",
    "S1A_IW_GRDH_1SDV_20190203T230559_20190203T230624_025771_02DDA8_A806",
    "S1B_IW_GRDH_1SDV_20190209T230513_20190209T230538_014875_01BC41_B2D3",
    "S1A_IW_GRDH_1SDV_20190215T230559_20190215T230622_025946_02E3DC_3113",
    "S1B_IW_GRDH_1SDV_20190221T230513_20190221T230538_015050_01C1FA_50F2",
    "S1A_IW_GRDH_1SDV_20190227T230559_20190227T230624_026121_02EA15_DFE8",
    "S1B_IW_GRDH_1SDV_20190305T230513_20190305T230538_015225_01C7C5_14D0",
    "S1A_IW_GRDH_1SDV_20190311T230559_20190311T230624_026296_02F06F_2121",
    "S1B_IW_GRDH_1SDV_20190317T230513_20190317T230538_015400_01CD6B_9117",
    "S1A_IW_GRDH_1SDV_20190323T230600_20190323T230625_026471_02F6E6_AC56",
    "S1B_IW_GRDH_1SDV_20190329T230513_20190329T230538_015575_01D324_E3AB",
]


@pytest.mark.skipif(not has_phx, reason="phoenix not installed")
def test_phx_catalog_slc():
    client = phx.Client()
    collection = client.get_collection("esa-sentinel-1-csar-l1-slc").at(
        "asf:daac:sentinel-1"
    )
    backend = PhoenixSentinel1SLCCatalogBackend(collection_source=collection)
    result = search_slc(backend, query)
    assert result.product_ids == expected_slc


@pytest.mark.skipif(not has_phx, reason="phoenix not installed")
def test_phx_catalog_grd():
    client = phx.Client()
    collection = client.get_collection("esa-sentinel-1-csar-l1-grd").at(
        "asf:daac:sentinel-1"
    )
    backend = PhoenixSentinel1GRDCatalogBackend(collection_source=collection)
    result = search_grd(backend, query)
    print(result.product_ids)
    assert result.product_ids == expected_grd


def test_cdse_catalog_slc():
    backend = CDSESentinel1SLCCatalogBackend()
    result = search_slc(backend, query)
    assert result.product_ids == expected_slc


def test_cdse_catalog_grd():
    backend = CDSESentinel1GRDCatalogBackend()
    result = search_grd(backend, query)
    assert result.product_ids == expected_grd


def test_cdse_catalog_slc_get_item():
    backend = CDSESentinel1SLCCatalogBackend()
    item = backend.get_cdse_item(
        "S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14"
    )
    assert (
        item["S3Path"]
        == "/eodata/Sentinel-1/SAR/SLC/2019/03/05/S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14.SAFE"
    )


def test_cdse_catalog_grd_get_item():
    backend = CDSESentinel1GRDCatalogBackend()
    item = backend.get_cdse_item(
        "S1B_IW_GRDH_1SDV_20190329T230513_20190329T230538_015575_01D324_E3AB"
    )
    assert (
        item["S3Path"]
        == "/eodata/Sentinel-1/SAR/GRD/2019/03/29/S1B_IW_GRDH_1SDV_20190329T230513_20190329T230538_015575_01D324_E3AB.SAFE"
    )
