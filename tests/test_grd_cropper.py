import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely.geometry

import eos.dem
from eos.products.sentinel1.catalog import CDSESentinel1GRDCatalogBackend
from eos.products.sentinel1.grd_cropper import (
    BboxDestinationGeometry,
    CDSEInputProduct,
    CropperInput,
    FilesystemResultDestination,
    MemoryResultDestination,
    Params,
    ProductsAreFromDifferentDatatakes,
    get_cdse_orbit_catalog_backend,
    process,
)


def test_grd_cropper(cdse_auth, cdse_s3_session, tmp_path):
    pid = "S1A_IW_GRDH_1SDV_20240125T041457_20240125T041522_052258_065159_C088"

    cdse_backend = CDSESentinel1GRDCatalogBackend()

    params = Params(
        polarizations=["VV", "VH"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[
            CDSEInputProduct(
                product_id=pid,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            )
        ],
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
        orbit_catalog_backend=get_cdse_orbit_catalog_backend(*cdse_auth),
    )

    metadata = process(input)
    assert os.path.exists(f"{tmp_path}/vv.tif")
    assert os.path.exists(f"{tmp_path}/vh.tif")

    r = rasterio.open(f"{tmp_path}/vv.tif").read(1)
    assert np.isnan(r).sum() == 0

    np.testing.assert_allclose(
        metadata.los_angles.los,
        (-0.5239635338760896, 0.09782481122050468, -0.8461043206825931),
    )
    np.testing.assert_allclose(metadata.los_angles.altitude, 107.51067352294922)
    np.testing.assert_allclose(metadata.los_angles.azimuth_angle, 280.57545555)
    np.testing.assert_allclose(
        metadata.los_angles.incidence_angle,
        32.20955155179675,
    )


def test_grd_cropper_2(cdse_auth, cdse_s3_session, tmp_path):
    pid = "S1A_IW_GRDH_1SDV_20221205T015438_20221205T015503_046190_0587C2_F48B"

    cdse_backend = CDSESentinel1GRDCatalogBackend()

    params = Params(
        polarizations=["VV", "VH"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[
            CDSEInputProduct(
                product_id=pid,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            )
        ],
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
        orbit_catalog_backend=get_cdse_orbit_catalog_backend(*cdse_auth),
    )

    process(input)
    assert os.path.exists(f"{tmp_path}/vv.tif")
    assert os.path.exists(f"{tmp_path}/vh.tif")

    r = rasterio.open(f"{tmp_path}/vv.tif").read(1)
    assert np.isnan(r).sum() == 0


def test_grd_cropper_completely_outside(cdse_auth, cdse_s3_session, tmp_path):
    pid = "S1A_IW_GRDH_1SDV_20221205T015438_20221205T015503_046190_0587C2_F48B"

    cdse_backend = CDSESentinel1GRDCatalogBackend()

    params = Params(
        polarizations=["VV", "VH"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[
            CDSEInputProduct(
                product_id=pid,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            )
        ],
        params=params,
        destination_geometry=BboxDestinationGeometry(
            bbox=(53.73, 50.96, 53.92, 51.02),
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
        orbit_catalog_backend=get_cdse_orbit_catalog_backend(*cdse_auth),
    )

    process(input)
    assert os.path.exists(f"{tmp_path}/vv.tif")
    assert os.path.exists(f"{tmp_path}/vh.tif")

    r = rasterio.open(f"{tmp_path}/vv.tif").read(1)
    assert np.isnan(r).sum() == r.size


# this is also a test for the _COG GRD products on CDSE
def test_grd_cropper_assembly(tmp_path, cdse_auth, cdse_s3_session):
    pid1 = "S1A_IW_GRDH_1SDV_20241201T132452_20241201T132517_056799_06F90F_0273_COG"
    pid2 = "S1A_IW_GRDH_1SDV_20241201T132517_20241201T132542_056799_06F90F_8DED_COG"

    cdse_backend = CDSESentinel1GRDCatalogBackend()

    geom = shapely.geometry.shape(
        {
            "coordinates": [
                [
                    [-108.93034707948136, 44.8850185093938],
                    [-108.93034707948136, 44.634934452915445],
                    [-108.57357229127953, 44.634934452915445],
                    [-108.57357229127953, 44.8850185093938],
                    [-108.93034707948136, 44.8850185093938],
                ]
            ],
            "type": "Polygon",
        }
    )
    bbox = geom.bounds

    params = Params(
        polarizations=["VV"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[
            CDSEInputProduct(
                product_id=pid1,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            ),
            CDSEInputProduct(
                product_id=pid2,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            ),
        ],
        params=params,
        destination_geometry=BboxDestinationGeometry(
            bbox=bbox,
            resolution=10,
            align=None,
            crs=None,
        ),
        result_destination=FilesystemResultDestination(
            {"VV": Path(f"{tmp_path}/vv.tif")}
        ),
        dem_source=eos.dem.SRTM4Source(),
        orbit_catalog_backend=get_cdse_orbit_catalog_backend(*cdse_auth),
    )

    process(input)

    # make sure we don't get nans, thanks to the assembly
    r = rasterio.open(f"{tmp_path}/vv.tif").read(1)
    assert np.isnan(r).sum() == 0


def test_grd_cropper_multiple_datatakes(cdse_auth, cdse_s3_session):
    pid1 = "S1A_IW_GRDH_1SDV_20221205T015438_20221205T015503_046190_0587C2_F48B"
    pid2 = "S1A_IW_GRDH_1SDV_20240125T041457_20240125T041522_052258_065159_C088"

    cdse_backend = CDSESentinel1GRDCatalogBackend()

    params = Params(
        polarizations=["VV", "VH"],
        calibration="gamma",
        orthorectify=True,
        rtc=None,
        filtering=None,
    )
    input = CropperInput(
        products=[
            CDSEInputProduct(
                product_id=pid1,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            ),
            CDSEInputProduct(
                product_id=pid2,
                cdse_backend=cdse_backend,
                s3_session=cdse_s3_session,
            ),
        ],
        params=params,
        destination_geometry=BboxDestinationGeometry(
            bbox=(73.73, 70.96, 73.92, 71.02),
            resolution=10,
            align=None,
            crs=None,
        ),
        result_destination=MemoryResultDestination.make_empty(),
        dem_source=eos.dem.DEMStitcherSource(),
        orbit_catalog_backend=get_cdse_orbit_catalog_backend(*cdse_auth),
    )

    with pytest.raises(ProductsAreFromDifferentDatatakes):
        process(input)
