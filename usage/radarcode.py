import os
import numpy as np
from eos.sar import io, dem_to_radar, regist, roi
from eos.products import sentinel1
from matplotlib import pyplot as plt 
#%%
remote_test = True

if remote_test: 
    
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'
   
    
else: 
    
    xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
    tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'
   
xml_basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
tiff_basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff'
                 
# list of our xmls
xml_path = os.path.join(xml_folder, xml_basename)

tiff_path = os.path.join(tiff_folder, tiff_basename)

image_reader = io.open_image(tiff_path)

# read xml
xml_content = io.read_xml_file( xml_path)

burst_id = 5
# extract the burst metadata
burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, 5)

# create a Sentinel1BurstModel
bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta,
                                                        degree=11,
                                                        bistatic_correction=True,
                                                        apd_correction=True,
                                                        max_iterations=20,
                                                        tolerance=0.001)


margin = 10

# define a region of interest where geocoding should occur
crop_roi = roi.Roi(5000,500, 1500, 500)

# get a good approximation of the geometry of the crop_roi
# with a margin of 10 pixels
refined_geom, alts, mask = bmod.get_approx_geom(_roi=crop_roi, margin=margin)

# get a dem on the previously estimated geometry
x, y, raster, transform, crs = regist.dem_points(refined_geom)
# estimate dem only on roi
crop_dem = dem_to_radar.dem_radarcoding(raster, transform, bmod,
                                        roi=crop_roi,
                                        approx_geometry=refined_geom,
                                        margin=margin, 
                                        get_xy=True)

#%% reffine the lon, lats with localization 
cols_grid, rows_grid = crop_roi.get_meshgrid()
x, y, z = bmod.localization(rows_grid.ravel(), cols_grid.ravel(), crop_dem[:,:,0].ravel(), 
                            x_init=crop_dem[:, :, 1].ravel(), 
                            y_init=crop_dem[:, :, 2].ravel(),
                            z_init=crop_dem[:,:,0].ravel())
#%%
localized=np.stack([z, x, y], axis=-1).reshape(*crop_roi.get_shape(), 3)
#%% read the image
read_roi = crop_roi.translate_roi(*bmod.burst_roi.to_roi()[:2])
cmplx_img = io.read_window(image_reader, read_roi)
#%% plots
abs_img = np.abs(cmplx_img)
vmin = np.percentile(abs_img, 10)
vmax = np.percentile(abs_img, 90)
plt.figure()
plt.imshow(abs_img, vmin=vmin, vmax=vmax, cmap='gray')
plt.show()
plt.figure()
plt.imshow(crop_dem[:, :, 0], cmap='gray')
plt.show()
plt.figure() 
plt.imshow(localized[:, : , 1])
plt.show()
plt.figure() 
plt.imshow(localized[:, : , 2])
plt.show()
