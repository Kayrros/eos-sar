import matplotlib.pyplot as plt
import numpy as np
import os
import eos.products.sentinel1 as s1
import eos.sar
from eos.sar.roi import Roi 
#%% init swath models 
remote_test = False

if remote_test: 
    
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'

else: 
    
    xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
    tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'

xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                     's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']

tiff_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                  's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']

# list of our xmls
xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames ]

tiff_paths = [os.path.join(tiff_folder, p) for p in tiff_basenames]

image_readers = [eos.sar.io.open_image(p) for p in tiff_paths]

# read the xmls as strings
xml_content = []
for xml_path in xml_paths: 
        xml_content.append( eos.sar.io.read_xml_file(xml_path))


# Now extract the needed metadata
primary_bursts_meta = s1.metadata.extract_bursts_metadata(
    xml_content[0])
secondary_bursts_meta = s1.metadata.extract_bursts_metadata(
    xml_content[1])

# get the indices of the common bursts
prim_burst_ids, sec_burst_ids = s1.deburst.get_bursts_intersection(
    len(primary_bursts_meta),
    primary_bursts_meta[0]['relative_burst_id'],
    len(secondary_bursts_meta),
    secondary_bursts_meta[0]['relative_burst_id']
)

def filter_list(iter_list, ids):
    return list(map(iter_list.__getitem__, ids))

# keep only the bursts intersecting
primary_bursts_meta = filter_list(primary_bursts_meta, prim_burst_ids)
secondary_bursts_meta = filter_list(secondary_bursts_meta, sec_burst_ids)

primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    secondary_bursts_meta)


#%% Now estimate the registration matrix

# get a good approximation of the geometry of the swath
# with a margin of 10 pixels
margin=10
refined_geom, alts, mask = primary_swath_model.get_approx_geom(margin=margin)

# get dem points
x, y, raster, transform, crs = eos.sar.regist.dem_points(refined_geom,
                                                         source='SRTM30',
                                                         datum='ellipsoidal')

# you can mask some pixels to speed up the projection
mask = np.random.binomial(n=1, p=0.01, size=x.shape).astype(bool)
x_masked = x[mask]
y_masked = y[mask]
raster_masked = raster[mask]

# project in primary
row_primary, col_primary, _ = primary_swath_model.projection(
    x_masked.ravel(), y_masked.ravel(), raster_masked.ravel(), crs=crs)

# project in secondary and estimate registration
A_swath = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                              secondary_swath_model, x_masked,
                                              y_masked, raster_masked, crs )
#%% define the roi in the swath 
# define the roi in the primary swath
# Here, if you set a region of interest within the swath
# in the primary burst, only this region will be considered

primary_swath_roi = Roi(10000, 785, 3000, 3000)
# primary_swath_roi = None

#%% deburst primary and secondary images 
# primary debursting
primary_debursted_crop, burst_ids, read_rois, write_rois = s1.deburst.deburst_in_primary_swath(
    primary_swath_model, image_readers[0], primary_swath_roi)


# burst_ids are the burst ids covered by the roi ( O based in the swath)
# rois_read are the regions that were read from the tiff
# rois_write are the rois where the read patches were written in the crop

# Now for the secondary
# estimate the rois where we need to read data
# and the associated resampler
secondary_read_rois, resamplers = s1.deburst.secondary_rois_and_resamplers(
    primary_swath_model, read_rois,
    burst_ids, secondary_swath_model,
    secondary_bursts_meta, A_swath)

# Secondary reading/resampling/ debursting
secondary_debursted_crop = s1.deburst.read_resample_and_deburst(
    image_readers[1], secondary_read_rois,
    resamplers, write_rois, primary_debursted_crop.shape)
#%% do the interferogram 

interf = primary_debursted_crop * np.conj(secondary_debursted_crop)


#%% create a TopoCorrection instance
topo = eos.sar.geom_phase.TopoCorrection(primary_swath_model,
                                         [secondary_swath_model],
                                         grid_size=50, degree=7,
                                         )

#%% predict flat earth and correct it 
flat_earth = topo.flat_earth_image(primary_swath_roi, wrapped=True)

correc = interf * np.exp(- 1j * flat_earth[0]).astype(np.complex64)



#%% Dem projection in radar coordinates
heights = eos.sar.dem_to_radar.dem_radarcoding(raster, transform,
                                               primary_swath_model,
                                               roi=primary_swath_roi,
                                               margin=margin)

#%% predict topographic phase
topo_phase = topo.topo_phase_image(heights, 
                                   primary_roi=primary_swath_roi,
                                   wrapped=False)
wrapped_topo = eos.sar.utils.wrap(topo_phase[0])

#%%  now correct interf
dinterf = correc * np.exp(- 1j * topo_phase[0]).astype(np.complex64)
#%%
plt.figure()
plt.imshow(np.angle(interf), cmap='jet')
plt.show()

plt.figure()
plt.imshow(flat_earth[0], cmap='jet')
plt.show()

plt.figure()
plt.imshow(np.angle(correc), cmap='jet')
plt.show()

plt.figure()
plt.imshow(heights)
plt.show()

plt.figure()
plt.imshow(wrapped_topo, cmap='jet')
plt.show()

plt.figure()
plt.imshow(np.angle(dinterf), cmap='jet')
plt.show()

# # optional
# # save files and check with vpv
# import tifffile
# import subprocess
# rasters = [interf, flat_earth[0], correc, heights, wrapped_topo, dinterf]
# names = ['interf', 'flat_earth', 'corrected_flat', 'heights', 'wrapped_topo', 'Dinterf']
# out_path = os.path.join('/tmp', 'vpv_imgs')
# if not(os.path.exists(out_path)): 
#     os.makedirs(out_path)
# cmd = ["vpv", "ac"]
# for raster, name in zip(rasters, names): 
#     out_img = os.path.join(out_path, f"{name}.tif") 
#     tifffile.imsave(out_img, raster)
#     cmd += [out_img]
# subprocess.Popen(cmd)