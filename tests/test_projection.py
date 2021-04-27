import s1m
from eos.products import sentinel1
import numpy as np
import sys
sys.path.append('../')

xml_path = './data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
s1model = s1m.Sentinel1Model(xml=xml_path)
bmod = sentinel1.burst_model.burst_model_from_s1m(
    s1model, burst=1, apd_correction=True,
    bistatic_correction=True)

# create a grid of points
x, y, w, h = bmod.burst_roi
Cols, Rows = np.meshgrid(np.linspace(0, w-1, 10), np.linspace(0, h-1, 10))
cols, rows = Cols.ravel(), Rows.ravel()
alts = np.zeros_like(cols)

# localize the points
lon, lat, alt = bmod.localization(cols, rows, alts)

# check if localized points are at alt = 0
np.testing.assert_allclose(alts, alt, atol=1e-5)

# now project these points back in the burst
cols_pred, rows_pred, i_pred = bmod.projection(lon, lat, alt)

# check if point fall back in the same location
np.testing.assert_allclose(cols_pred, cols, atol=1e-3)
np.testing.assert_allclose(rows_pred, rows, atol=1e-3)

# verify projection vs s1m projection
s1_cols_pred, s1_row_pred, s1_i_pred = s1m.main_projection(s1model, lon, lat, alt, error_when_outside=False,
                                                           apd_correction=True, bistatic_correction=True, verbose=False)

# check similarity of x coordinate referenced to first col in raster
np.testing.assert_allclose(s1_cols_pred + s1model.x_min,
                           cols_pred + bmod.burst_roi[0])

# check similarity of azimuth time
azt_pred, _ = bmod.to_azt_rng(rows_pred, cols_pred)
np.testing.assert_allclose(
    s1_row_pred/s1model.azimuth_frequency + s1model.burst_times[0][1], 
    azt_pred)
