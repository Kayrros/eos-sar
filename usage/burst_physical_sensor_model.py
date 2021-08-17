import numpy as np
import os
import eos.products.sentinel1
import eos.sar

remote_test = True

if remote_test: 
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    # prepare oio config 
    prof_name = 'oio'
    en_url = 'https://s3.kayrros.org'
else: 
    xml_folder = '../tests/data'
    # just set oio vars to None in this case
    prof_name = en_url = None
basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'    
xml_path = os.path.join(xml_folder, basename)

# read xml
xml_content = eos.sar.io.read_xml_file(
    xml_path, profile_name=prof_name,
    endpoint_url=en_url)

burst_id = 1

# extract the burst metadata
burst_meta = eos.products.sentinel1.metadata.extract_burst_metadata(
    xml_content, burst_id)

# create a Sentinel1BurstModel
bmod = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(burst_meta,
                                                                     degree=11,
                                                                     bistatic_correction=True,
                                                                     apd_correction=True,
                                                                     max_iterations=20,
                                                                     tolerance=0.001)
# create a grid of points
x, y, w, h = bmod.burst_roi
cols_grid, rows_grid = np.meshgrid(np.linspace(0, w-1, 10), np.linspace(0, h-1, 10))
cols, rows = cols_grid.ravel(), rows_grid.ravel()
alts = np.zeros_like(cols)

# localize the points
lon, lat, alt = bmod.localization(rows, cols, alts)


# now project these points back in the burst
rows_pred, cols_pred, i_pred = bmod.projection(lon, lat, alt)
