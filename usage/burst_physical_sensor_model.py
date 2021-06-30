import numpy as np
import os
import eos.products.sentinel1

xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'

xml_path = os.path.join(xml_folder, basename)

burst_id = 1

# read the content of the xml
with open(xml_path) as f:
    xml_content = f.read()

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
Cols, Rows = np.meshgrid(np.linspace(0, w-1, 10), np.linspace(0, h-1, 10))
cols, rows = Cols.ravel(), Rows.ravel()
alts = np.zeros_like(cols)

# localize the points
lon, lat, alt = bmod.localization(rows, cols, alts)


# now project these points back in the burst
rows_pred, cols_pred, i_pred = bmod.projection(lon, lat, alt)
