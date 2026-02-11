import datetime

from eos.products.sentinel1.catalog import (
    CDSESentinel1GRDCatalogBackend,
    CDSESentinel1SLCCatalogBackend,
)
from eos.products.sentinel1.product import (
    CDSEUnzippedSafeSentinel1GRDProductInfo,
    CDSEUnzippedSafeSentinel1SLCProductInfo,
)


def test_product_properties_slc(cdse_s3_session):
    product_id = "S1A_IW_SLC__1SDV_20250313T055953_20250313T060021_058282_07342C_D57A"
    catalog_backend = CDSESentinel1SLCCatalogBackend()
    product = CDSEUnzippedSafeSentinel1SLCProductInfo.from_product_id(
        catalog_backend, cdse_s3_session, product_id
    )

    props = product.get_properties()
    assert props.footprint == [
        (46.431999, 4.073224),
        (46.832890, 0.731536),
        (48.505569, 1.112442),
        (48.103569, 4.565003),
    ]
    assert props.platform == "S1A"
    assert props.ipf_version == "003.91"
    assert props.cycle_number == 347
    assert props.relative_orbit_number == 110
    assert props.absolute_orbit_number == 58282
    assert props.orbit_direction == "desc"
    assert props.anx_time == datetime.datetime(2025, 3, 13, 5, 23, 41, 789654)
    assert not props.crossing_anx


def test_product_properties_grd(cdse_s3_session):
    product_id = "S1A_IW_GRDH_1SDV_20230103T003252_20230103T003321_046612_059621_25B2"
    catalog_backend = CDSESentinel1GRDCatalogBackend()
    product = CDSEUnzippedSafeSentinel1GRDProductInfo.from_product_id(
        catalog_backend, cdse_s3_session, product_id
    )
    props = product.get_properties()
    assert props.footprint == [
        (38.921623, 84.360428),
        (39.321037, 81.436714),
        (41.062702, 81.805511),
        (40.664528, 84.804825),
    ]
    assert props.platform == "S1A"
    assert props.ipf_version == "003.52"
    assert props.cycle_number == 280
    assert props.relative_orbit_number == 165
    assert props.absolute_orbit_number == 46612
    assert props.orbit_direction == "desc"
    assert props.anx_time == datetime.datetime(2023, 1, 2, 23, 54, 36, 762504)
    assert not props.crossing_anx
