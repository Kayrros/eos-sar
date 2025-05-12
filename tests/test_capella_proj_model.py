import os
from datetime import datetime, timezone
from math import ceil, floor

import numpy as np
import pytest
import rasterio
import requests
from numpy.typing import NDArray

from eos.products.capella.metadata import CapellaSLCMetadata, parse_metadata
from eos.products.capella.proj_model import CapellaSLCModel
from eos.sar.atmospheric_correction import ApdCorrection
from eos.sar.fourier_zoom import fourier_zoom
from eos.sar.io import read_window
from eos.sar.max_finding import sub_pixel_maxima
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector
from eos.sar.roi import Roi

BASE_URLS = [
    "https://capella-open-data.s3.amazonaws.com/data/2022/3/19/",
    "https://capella-open-data.s3.amazonaws.com/data/2022/3/18/",
]

PRODUCT_IDS = [
    "CAPELLA_C02_SM_SLC_HH_20220319141314_20220319141318",
    "CAPELLA_C06_SM_SLC_HH_20220318142552_20220318142558",
]

"""
We could have worked with this product in theory
    base_url = "https://capella-open-data.s3.amazonaws.com/data/2021/10/9/" 
    product_id = "CAPELLA_C06_SP_SLC_HH_20211009234314_20211009234317"

However, 
We remove it from the test suite because it seems its metadata is wrong,
namely the delta_line_time (in azimuth) seems to be wrong 
which gives an incorrect proj_model. 
This is supported by the fact that delta_line_time is inconsitent with the azimuth pixel size in meters 

when you multiply the time with the state vector velocity:
 print(capella_meta.delta_line_time * np.mean([np.linalg.norm(sv.velocity) for sv in capella_meta.state_vectors])) 
 > 0.7072591961053496
 
when you perform localization of point at col and col + eps: 
 import pyproj
 eps = 1e-4

 # use centroid to get estimate of scene height
 centroid = capella_meta.center_pixel_target_position 
 transformer = pyproj.transformer.Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
 lon, lat, alt = transformer.transform(*centroid)

 # localize first pixel
 p1 = proj_model.localization(0, 0, alt, crs="epsg:4978")
 p2 = proj_model.localization(0, eps, alt, crs="epsg:4978")
 print(np.linalg.norm(np.array(p2) - np.array(p1)) / eps)
 > 0.7999147961463473
 
And you compare to     
 print(capella_meta.azimuth_pixel_size)
 > 0.21297066003081772

It is probable that capella_meta.azimuth_pixel_size is correct but capella_meta.delta_line_time is wrong for this product

It is also convenient to skip it from the tests,
because the other two products are stripmap with small doppler centroids (verified manually), so we can 
zoom the complex values without deramping without a big loss of precision.
"""


def cr_gda2020_to_itrf2014(
    Cr_records: NDArray[np.float64], date: datetime
) -> NDArray[np.float64]:
    """
    Convert coordinates from GDA2020 to ITRF2014.

    Parameters
    ----------
    cr_records : ndarray (Nrecords, 6)
        each record (x, y, z, vx, vy, vz).
    date : datetime
        date onto which we transform.

    Returns
    -------
    ndarray (N, 3)
        N (x, y, z) records at the date.

    """
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    dt = (date - t0).days / 365.25
    return Cr_records[:, :3] + Cr_records[:, 3:] * dt


def itrf2014_to_itrf2008(
    Cr_records: NDArray[np.float64], date: datetime
) -> NDArray[np.float64]:
    """
    Convert coordinates from ITRF2014 to ITRF2008 at a given date.

    Args:
        Cr_records (ndarray (N, 3)): (x, y, z) cartesian coordinates in ITRF2014
        date (datetime): date at which we want the ITRF2008 coordinates

    Returns:
        (N, 3) ndarray with the x, y, z coordinates in ITRF2008 @ t
    """
    # The ITRF2014 origin is defined in such a way that there are zero
    # translation parameters at epoch 2010.0
    # http://itrf.ensg.ign.fr/ITRF_solutions/2014/frame_ITRF2014.php
    t0 = datetime(2010, 1, 1, tzinfo=timezone.utc)
    dt = (date - t0).days / 365.25

    a = np.array([1.6, 1.9, 2.4, -0.02], dtype=float)
    b = np.array([0, 0, -0.1, 0.03], dtype=float)

    tx, ty, tz, sc = a + b * dt
    tx *= 1e-3
    ty *= 1e-3
    tz *= 1e-3
    sc *= 1e-9
    return np.array([tx, ty, tz]) + (1 + sc) * Cr_records


def get_cr_coords_itrf(
    coords: NDArray[np.float64], date: datetime, use_2008: bool = True
) -> NDArray[np.float64]:
    """
    Convert coordinates to itrf (either 2014 or 2008).

    Parameters
    ----------
    coords : NDArray[np.float64]
        (N, 6) where each record is (x, y, z, vx, vy, vz).
    date : datetime
        date at which we want the ITRF coordinates.
    use_2008 : bool, optional
        If True, ITRF2008. Else, ITRF2014. The default is True.

    Returns
    -------
    NDArray[np.float64]
        (N, 3) with the (x, y, z) coordinates in ITRF.

    """
    cr_2014 = cr_gda2020_to_itrf2014(coords, date)

    if use_2008:
        cr_2008 = itrf2014_to_itrf2008(cr_2014, date)
        return cr_2008
    else:
        return cr_2014


@pytest.mark.parametrize("base_url,product_id", zip(BASE_URLS, PRODUCT_IDS))
def test_corner_reflector_41_surat(base_url, product_id):
    # setup urls
    meta_url = os.path.join(base_url, product_id, f"{product_id}_extended.json")
    geotiff_url = os.path.join(base_url, product_id, f"{product_id}.tif")

    # read corner reflector from file
    with open("./tests/data/QLD_corner_reflector_positions_GDA2020.txt", "r") as f:
        lines = f.readlines()

    assert lines[0].split() == [
        "Name",
        "Latitude",
        "Longitude",
        "Height",
        "X",
        "Y",
        "Z",
        "veloX",
        "veloY",
        "veloZ",
        "Azimuth",
        "Elevation",
    ]
    # only work on 41, because it is the only one in the products chosen
    CR_41_line = lines[39].split()
    assert CR_41_line[0] == "SB41-CRApex"

    coords = list(map(float, CR_41_line[4:10]))

    # compute the date of the product
    start = len("CAPELLA_C02_SM_SLC_HH_")
    end = len("CAPELLA_C02_SM_SLC_HH_20220319")
    date_str = product_id[start:end]
    date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)

    # convert to itrf for product date
    coords_numpy = np.array(coords, dtype=np.float64)[None, :]  # (1, 6) array
    coords_itrf = get_cr_coords_itrf(coords_numpy, date).squeeze()

    # download metadata and parse
    response = requests.get(meta_url)
    capella_meta = parse_metadata(response.text)
    assert isinstance(capella_meta, CapellaSLCMetadata)

    # get proj model with apd correction
    orbit = Orbit(capella_meta.state_vectors, degree=11)
    proj_model = CapellaSLCModel.from_metadata(
        capella_meta,
        orbit,
        corrector=Corrector([ApdCorrection(orbit)]),
        max_iterations=20,
        tolerance=0.0001,
    )

    # project CR 41
    r, c, _ = proj_model.projection(
        coords_itrf[0], coords_itrf[1], coords_itrf[2], crs="epsg:4978"
    )

    # assert in image
    assert r >= 0 and r <= proj_model.h - 1 and c >= 0 and c <= proj_model.w - 1, (
        "CR outside image"
    )

    crop_size = 64
    zoom_factor = 16

    # do a crop around CR
    col = round(c) - crop_size // 2
    row = round(r) - crop_size // 2

    # take roi around the prediction
    roi = Roi(col, row, crop_size, crop_size).make_valid((proj_model.h, proj_model.w))

    assert roi != Roi(0, 0, 0, 0), "Roi outside image"

    col_pred, row_pred = c - col, r - row

    with rasterio.open(geotiff_url, "r") as db:
        array = read_window(db, roi, get_complex=True)

    """
    # Checking the centering of the azimuth spectrum at this stage
    # could be done with the following
    
    fft = np.fft.fftshift(np.fft.fft2(array))
    abs_az = np.abs(fft).mean(axis=1)
    
    from matplotlib import pyplot as plt 
    plt.figure()
    plt.plot(abs_az)
    plt.show()
    """

    # zoom complex, assuming centered azimuth spectrum
    zoomed = fourier_zoom(array, z=zoom_factor)
    # Then the intensity
    intensity_zoomed = np.real(zoomed) ** 2 + np.imag(zoomed) ** 2

    # Now find the max of the intensity and compare with prediction
    search_roi = Roi.from_bounds_tuple(
        (
            floor(col_pred - 3),
            floor(row_pred - 3),
            ceil(col_pred + 3),
            ceil(row_pred + 3),
        )
    )

    subpix_max_measured, _ = sub_pixel_maxima(
        intensity_zoomed, search_roi, zoom_factor=zoom_factor
    )

    assert len(subpix_max_measured), "No local max found in search region"

    # then just take the most significant maximum
    (row_measured, col_measured), _ = subpix_max_measured[0]

    assert row_measured is not None, (
        "Quadratic polynomial fitting failed around prediction"
    )

    # compute Absolute location error in pixels
    row_ale = float(row_pred - row_measured)
    col_ale = float(col_pred - col_measured)

    # in meters
    az_ale_meters = row_ale * capella_meta.azimuth_pixel_size
    rng_ale_meters = col_ale * capella_meta.range_pixel_size

    # set tolerance to 3 meters for azimuth and range
    # We mostly need the tolerance in range because of uncompensated effects and simplistic APD
    # But azimuth is not perfect as well
    assert abs(az_ale_meters) < 3, "More than 3 meters error in azimuth"
    assert abs(rng_ale_meters) < 3, "More than 3 meters error in range"


def test_proj_model_slc():
    base_url = BASE_URLS[0]
    product_id = PRODUCT_IDS[0]

    meta_url = os.path.join(base_url, product_id, f"{product_id}_extended.json")

    # download metadata and parse
    response = requests.get(meta_url)
    capella_meta = parse_metadata(response.text)
    assert isinstance(capella_meta, CapellaSLCMetadata)

    # get proj model with apd correction
    orbit = Orbit(capella_meta.state_vectors, degree=11)
    proj_model = CapellaSLCModel.from_metadata(
        capella_meta,
        orbit,
        corrector=Corrector([ApdCorrection(orbit)]),
        max_iterations=20,
        tolerance=0.0001,
    )

    # create a grid of points
    cols_grid, rows_grid = np.meshgrid(
        np.linspace(-1000, proj_model.w + 1000, 100),
        np.linspace(-1000, proj_model.h + 1000, 100),
    )
    cols, rows = cols_grid.ravel(), rows_grid.ravel()
    alts = np.zeros_like(cols)

    # localize the points
    lon, lat, alt = proj_model.localization(rows, cols, alts)

    # check if localized points are at alt = 0
    np.testing.assert_allclose(alts, alt, atol=1e-5)

    # now project these points back in the burst
    rows_pred, cols_pred, _ = proj_model.projection(lon, lat, alt)

    # check if point fall back in the same location
    np.testing.assert_allclose(cols_pred, cols, rtol=1e-3)
    np.testing.assert_allclose(rows_pred, rows, rtol=1e-3)

    # check ability to query one point
    ptlon, ptlat, ptalt = proj_model.localization(rows[0], cols[0], alts[0])
    assert isinstance(ptlon, float), (
        "vectorized localization func failed on scalar input"
    )
    assert isinstance(ptlat, float), (
        "vectorized localization func failed on scalar input"
    )
    assert isinstance(ptalt, float), (
        "vectorized localization func failed on scalar input"
    )

    # check ability to query one point
    ptrow, ptcol, pti = proj_model.projection(lon[0], lat[0], alt[0])
    assert isinstance(ptrow, float), "vectorized projection func failed on scalar input"
    assert isinstance(ptcol, float), "vectorized projection func failed on scalar input"
    assert isinstance(pti, float), "vectorized projection func failed on scalar input"

    # check validity of cropped proj model
    roi = Roi(50, 75, 200, 300)
    proj_model_on_roi = proj_model.to_cropped_model(roi)

    rows_pred_shifted, cols_pred_shifted, _ = proj_model_on_roi.projection(
        lon, lat, alt
    )

    np.testing.assert_allclose(rows_pred_shifted, rows_pred - roi.row, atol=1e-5)

    np.testing.assert_allclose(cols_pred_shifted, cols_pred - roi.col, atol=1e-5)
