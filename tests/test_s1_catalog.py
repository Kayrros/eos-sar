import datetime

import pytest
import requests
import shapely.geometry

from eos.products.sentinel1.catalog import (
    CDSESentinel1GRDCatalogBackend,
    CDSESentinel1SLCCatalogBackend,
    Sentinel1CatalogQuery,
    search_grd,
    search_slc,
)

QUERY = Sentinel1CatalogQuery(
    geometry=shapely.geometry.Point(-68.374028, -23.563574),
    relative_orbit_number=149,
    start_date=datetime.datetime(2019, 1, 1),
    end_date=datetime.datetime(2019, 4, 1),
    polarization=["SV", "DV"],
)

EXPECTED_SLC = [
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


EXPECTED_GRD = [
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

QUERY_NEW = Sentinel1CatalogQuery(
    geometry=shapely.geometry.Point(150.666, -26.842),
    relative_orbit_number=111,
    start_date=datetime.datetime(2025, 3, 1),
    end_date=datetime.datetime(2025, 4, 29),
    polarization=["SV", "DV", "DH", "SH"],
)

EXPECTED_SLC_NEW = [
    "S1A_IW_SLC__1SSV_20250301T083323_20250301T083350_058108_072D23_874B",
    "S1A_IW_SLC__1SSH_20250313T083326_20250313T083353_058283_07343E_B0A2",
    "S1A_IW_SLC__1SSV_20250325T083306_20250325T083334_058458_073B1E_CBB3",
    "S1C_IW_SLC__1SDV_20250331T083159_20250331T083230_001682_002CE8_D50A",
    "S1A_IW_SLC__1SSH_20250406T083307_20250406T083335_058633_074243_0532",
    "S1C_IW_SLC__1SDV_20250412T083159_20250412T083230_001857_00381E_A63D",
    "S1A_IW_SLC__1SSV_20250418T083307_20250418T083335_058808_074969_CCA4",
    "S1C_IW_SLC__1SDH_20250424T083200_20250424T083231_002032_004292_AC89",
]


EXPECTED_GRD_NEW = [
    "S1A_IW_GRDH_1SSV_20250301T083324_20250301T083349_058108_072D23_3637",
    "S1A_IW_GRDH_1SSH_20250313T083327_20250313T083352_058283_07343E_D336",
    "S1A_IW_GRDH_1SSV_20250325T083307_20250325T083332_058458_073B1E_5E50",
    "S1C_IW_GRDH_1SDV_20250331T083159_20250331T083229_001682_002CE8_60DB",
    "S1A_IW_GRDH_1SSH_20250406T083308_20250406T083333_058633_074243_084E",
    "S1C_IW_GRDH_1SDV_20250412T083159_20250412T083229_001857_00381E_2D13",
    "S1A_IW_GRDH_1SSV_20250418T083308_20250418T083333_058808_074969_9FA7",
    "S1C_IW_GRDH_1SDH_20250424T083200_20250424T083229_002032_004292_E22D",
]


def test_phx_catalog_slc(phx_client):
    query = QUERY
    expected_slc = EXPECTED_SLC
    from eos.products.sentinel1.catalog import PhoenixSentinel1SLCCatalogBackend

    collection = phx_client.get_collection("esa-sentinel-1-csar-l1-slc").at(
        "asf:daac:sentinel-1"
    )
    backend = PhoenixSentinel1SLCCatalogBackend(collection_source=collection)
    result = search_slc(backend, query)
    assert result.product_ids == expected_slc


def test_phx_catalog_grd(phx_client):
    query = QUERY
    expected_grd = EXPECTED_GRD
    from eos.products.sentinel1.catalog import PhoenixSentinel1GRDCatalogBackend

    collection = phx_client.get_collection("esa-sentinel-1-csar-l1-grd").at(
        "asf:daac:sentinel-1"
    )

    backend = PhoenixSentinel1GRDCatalogBackend(collection_source=collection)
    result = search_grd(backend, query)
    print(result.product_ids)
    assert result.product_ids == expected_grd


@pytest.mark.parametrize(
    "query,expected_slc", [(QUERY, EXPECTED_SLC), (QUERY_NEW, EXPECTED_SLC_NEW)]
)
@pytest.mark.xfail(raises=requests.exceptions.RequestException, strict=False)
def test_cdse_catalog_slc(query, expected_slc):
    backend = CDSESentinel1SLCCatalogBackend()
    result = search_slc(backend, query)
    assert result.product_ids == expected_slc


@pytest.mark.parametrize(
    "query,expected_grd", [(QUERY, EXPECTED_GRD), (QUERY_NEW, EXPECTED_GRD_NEW)]
)
@pytest.mark.xfail(raises=requests.exceptions.RequestException, strict=False)
def test_cdse_catalog_grd(query, expected_grd):
    backend = CDSESentinel1GRDCatalogBackend()
    result = search_grd(backend, query)
    assert result.product_ids == expected_grd


@pytest.mark.xfail(raises=requests.exceptions.RequestException, strict=False)
def test_cdse_catalog_slc_get_item():
    backend = CDSESentinel1SLCCatalogBackend()
    item = backend.get_cdse_item(
        "S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14"
    )
    assert (
        item["S3Path"]
        == "/eodata/Sentinel-1/SAR/SLC/2019/03/05/S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14.SAFE"
    )


@pytest.mark.xfail(raises=requests.exceptions.RequestException, strict=False)
def test_cdse_catalog_grd_get_item():
    backend = CDSESentinel1GRDCatalogBackend()
    item = backend.get_cdse_item(
        "S1B_IW_GRDH_1SDV_20190329T230513_20190329T230538_015575_01D324_E3AB"
    )
    assert (
        item["S3Path"]
        == "/eodata/Sentinel-1/SAR/GRD/2019/03/29/S1B_IW_GRDH_1SDV_20190329T230513_20190329T230538_015575_01D324_E3AB.SAFE"
    )
