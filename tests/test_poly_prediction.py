import os

import numpy as np

import eos.products.sentinel1 as s1
from eos.sar import geoconfig, io, poly
from eos.sar.orbit import Orbit


def get_normalization(vec):
    """
    normalize between -2 & 2
    """
    a = np.amin(vec, axis=0)
    b = np.amax(vec, axis=0)
    off = (b + a) / 2
    scale = (b - a) / 4
    return off, scale


def test_poly():
    npoints = 1000
    coeffs = np.random.random_sample(3) * 50
    x = np.random.uniform(low=0, high=10000, size=npoints)
    y = np.random.uniform(low=0, high=10000, size=npoints)

    z = coeffs[0] + coeffs[1] * x + coeffs[2] * y
    z = z.reshape(npoints, -1)
    xoff, xscale = get_normalization(x)
    yoff, yscale = get_normalization(y)
    zoff, zscale = get_normalization(z)
    # ground truth poly on normalized coords
    cst = (coeffs[0] + coeffs[1] * xoff + coeffs[2] * yoff - zoff) / zscale
    xcoef = coeffs[1] * xscale / zscale
    ycoef = coeffs[2] * yscale / zscale
    gt_coeffs = np.array([cst, xcoef, ycoef])
    fitted = poly.polymodel(degree=1)
    fitted.fit_poly(x, y, z)
    np.testing.assert_allclose(gt_coeffs, fitted.coeffs)
    zeval = fitted.eval_poly(x, y)
    rmse = np.sqrt(np.mean((zeval - z) ** 2))
    np.testing.assert_allclose(rmse, 0, atol=1e-3)


def test_baseline_predictions():
    xml_folder = (
        "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/annotation"
    )
    xml_basenames = [
        "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml",
        "s1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml",
    ]
    # list of our xmls
    xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames]
    # read the xmls as strings
    xml_content = []
    for xml_path in xml_paths:
        xml_content.append(io.read_xml_file(xml_path))

    # Now extract the needed metadata
    primary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[0])
    secondary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[1])

    primary_orbit = Orbit(s1.metadata.unique_sv_from_bursts_meta(primary_bursts_meta))
    primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
        primary_bursts_meta, primary_orbit
    )

    secondary_orbit = Orbit(
        s1.metadata.unique_sv_from_bursts_meta(secondary_bursts_meta)
    )
    secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
        secondary_bursts_meta, secondary_orbit
    )

    pred = geoconfig.GeometryPredictor(
        primary_swath_model, [secondary_swath_model], grid_size=20, degree=7
    )

    npts = 50
    rows = np.random.uniform(0, primary_swath_model.h, size=npts)
    cols = np.random.uniform(0, primary_swath_model.w, size=npts)

    par_baseline = pred.predict_par_baseline(rows, cols)
    perp_baseline = pred.predict_perp_baseline(rows, cols)
    inc = pred.predict_incidence(rows, cols)
    assert par_baseline.shape == (npts, 1)
    assert perp_baseline.shape == (npts, 1)
    assert inc.shape == (npts, 1)
    par_baseline = pred.predict_par_baseline(rows, cols, grid_eval=True)
    perp_baseline = pred.predict_perp_baseline(rows, cols, grid_eval=True)
    inc = pred.predict_incidence(rows, cols, grid_eval=True)
    assert par_baseline.shape == (npts**2, 1)
    assert perp_baseline.shape == (npts**2, 1)
    assert inc.shape == (npts**2, 1)
