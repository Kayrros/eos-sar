import numpy as np
import os
from eos.sar import model, io, roi
from eos.products import sentinel1

remote_test = True

if remote_test: 
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
else: 
    xml_folder = '../tests/data'

basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'    
xml_path = os.path.join(xml_folder, basename)

# read xml
xml_content = io.read_xml_file( xml_path)

burst_id = 1

# extract the burst metadata
burst_meta = sentinel1.metadata.extract_burst_metadata(
    xml_content, burst_id)

# create a Sentinel1BurstModel
bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta,
                                                        degree=11,
                                                        bistatic_correction=True,
                                                        apd_correction=True,
                                                        max_iterations=20,
                                                        tolerance=0.001)

#%%
rows = np.round(np.random.rand(5) * 1000)
cols = np.round(np.random.rand(5) *20000)

#%% test recursively shrinking the interval on a single point
am1, am2, ad1, ad2, masks = model.recursive_shrink_interval(
    sensor_model=bmod, row=0, col=0, alt_min=-10000, alt_max=10000,
    num_alt=50, max_iter=10, eps=1e-1, verbosity=True)
print("Minimum altitude: ", am1)
print("Diff from srtm at  min altitude: ", ad1)
print("Maximum altitude: ", am2)
print("Diff from srtm at  max altitude: ", ad2)
print("Num of converged points: ", masks["converged"].sum())
print("Num of points where opt exactly 0: ", masks["zeros"].sum())
print("Num of points where opt not found in interval: ", masks["invalid"].sum())

#%% test recursively shrinking the interval on a set of points
am1, am2, ad1, ad2, masks = model.recursive_shrink_interval(
    sensor_model=bmod, row=rows, col=cols, alt_min=-10000, alt_max=10000,
    num_alt=50, max_iter=10, eps=1e-1, verbosity=True)
print("Minimum altitude: ", am1)
print("Diff from srtm at  min altitude: ", ad1)
print("Maximum altitude: ", am2)
print("Diff from srtm at  max altitude: ", ad2)
print("Num of converged points: ", masks["converged"].sum())
print("Num of points where opt exactly 0: ", masks["zeros"].sum())
print("Num of points where opt not found in interval: ", masks["invalid"].sum())

#%% test localizing without alt 
lon, lat, alt, masks = bmod.localize_without_alt(rows, cols, max_iter=5, eps=1,
                             alt_min=-1000, alt_max=9000, num_alt=100,
                             verbosity=True)
print("lon: ", lon)
print("lat: ", lat)
print("alt: ", alt)
print("Num of converged points: ", masks["converged"].sum())
print("Num of points where opt exactly 0: ", masks["zeros"].sum())
print("Num of points where opt not found in interval: ", masks["invalid"].sum())
#%% reproject and compare with rows and cols 
rows_pred, cols_pred, _ = bmod.projection(lon, lat, alt)
print("Row error : ", rows_pred - rows)
print("Col error : ", cols_pred - cols)

#%% Localize a Region of interest without height
roi_geom = roi.Roi(50,10, 800, 1000)
# here this is just to get the ground truth bounding points 
rows_roi, cols_roi = roi_geom.to_bounding_points()
# Localize to get the geometry, altitudes, validity masks
approx_geom, alts, masks = bmod.get_approx_geom(roi_geom)

print("Approx geom: ", approx_geom)
print("alts :", alts)
print("Num of converged points: ", masks["converged"].sum())
print("Num of points where opt exactly 0: ", masks["zeros"].sum())
print("Num of points where opt not found in interval: ", masks["invalid"].sum())

# reproject and compare with the ground truth 
projected = [bmod.projection(*a, alt) for (a, alt) in zip(approx_geom, alts)]

row_err = rows_roi - np.array([p[0] for p in projected])
col_err = cols_roi - np.array([p[1] for p in projected])

print("Row err: ", row_err)
print("Col err:", col_err)
