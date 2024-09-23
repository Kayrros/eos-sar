import os
from pathlib import Path

import rasterio

import eos.dem
from eos.products.sentinel1.grd_cropper import (
    BboxDestinationGeometry,
    CropperInput,
    FilesystemResultDestination,
    Params,
    PhoenixInputProduct,
    get_phoenix_orbit_catalog_backend,
    process,
)


def test_grd_cropper(phx_client, tmp_path):
    pid = "S1A_IW_GRDH_1SDV_20240125T041457_20240125T041522_052258_065159_C088"

    collection = phx_client.get_collection("esa-sentinel-1-csar-l1-grd").at(
        "aws:proxima:sentinel-s1-l1c"
    )
    item = collection.get_item(pid)

    params = Params(
        polarizations=["VV", "VH"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[PhoenixInputProduct(item)],
        params=params,
        destination_geometry=BboxDestinationGeometry(
            bbox=(29.00, 41.00, 29.10, 41.10),
            resolution=10,
            align=60,
            crs=rasterio.CRS.from_user_input("epsg:32635"),
        ),
        result_destination=FilesystemResultDestination(
            {
                "VV": Path(f"{tmp_path}/vv.tif"),
                "VH": Path(f"{tmp_path}/vh.tif"),
            }
        ),
        dem_source=eos.dem.DEMStitcherSource(),
        orbit_catalog_backend=get_phoenix_orbit_catalog_backend(client=phx_client),
    )

    process(input)
    assert os.path.exists(f"{tmp_path}/vv.tif")
    assert os.path.exists(f"{tmp_path}/vh.tif")


def test_grd_cropper_2(phx_client, tmp_path):
    pid = "S1A_IW_GRDH_1SDV_20221205T015438_20221205T015503_046190_0587C2_F48B"

    collection = phx_client.get_collection("esa-sentinel-1-csar-l1-grd").at(
        "aws:proxima:sentinel-s1-l1c"
    )
    item = collection.get_item(pid)

    params = Params(
        polarizations=["VV", "VH"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[PhoenixInputProduct(item)],
        params=params,
        destination_geometry=BboxDestinationGeometry(
            bbox=(73.73, 70.96, 73.92, 71.02),
            resolution=10,
            align=None,
            crs=None,
        ),
        result_destination=FilesystemResultDestination(
            {
                "VV": Path(f"{tmp_path}/vv.tif"),
                "VH": Path(f"{tmp_path}/vh.tif"),
            }
        ),
        dem_source=eos.dem.DEMStitcherSource(),
        orbit_catalog_backend=get_phoenix_orbit_catalog_backend(client=phx_client),
    )

    process(input)
    assert os.path.exists(f"{tmp_path}/vv.tif")
    assert os.path.exists(f"{tmp_path}/vh.tif")
