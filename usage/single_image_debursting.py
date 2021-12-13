import os
import numpy as np
import matplotlib.pyplot as plt 
import eos.products.sentinel1
from eos.sar.roi import Roi

def extract_keys(big_dict, list_keys): 
    o = {}
    for key in list_keys: 
        o[key] = big_dict[key]
    return o

def get_ref_metas(ref_xml_paths):
    xml_contents = [eos.sar.io.read_xml_file(xml_path) for xml_path in ref_xml_paths]
    keys = ['slant_range_time',
    'samples_per_burst',
    'range_frequency']
    ref_metas = [extract_keys(eos.products.sentinel1.metadata.extract_burst_metadata(
        xml_content, 0), keys) for xml_content in xml_contents]
    return ref_metas

get_complex = True # to get complex debursted images

remote_test = True

if remote_test: 
    
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'
   
    
else: 
    
    xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
    tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'
   
xml_basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
tiff_basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff'
ref_basename = 's1b-iw2-slc-vv-20190803t164006-20190803t164034-017424-020c57-005.xml'
                  
# list of our xmls
xml_path = os.path.join(xml_folder, xml_basename)

tiff_path = os.path.join(tiff_folder, tiff_basename)

image_reader = eos.sar.io.open_image(tiff_path)


# read the xmls as strings
xml_content = eos.sar.io.read_xml_file(xml_path)
# Now extract the needed metadata
primary_bursts_meta = eos.products.sentinel1.metadata.extract_bursts_metadata(
    xml_content)

ref_meta = get_ref_metas([os.path.join(xml_folder, ref_basename)])[0]

# construct primary swath model
primary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

# If you wish to deburst a "crop" defined by a roi in the swath coordinates
roi_in_swath = Roi(500, 750, 1000, 3000)
# if you wish to deburst the whole swath, set to None
# roi_in_swath = None 
# Careful, might be slow due to io
# might be better to test it with remote_test = False

#%%
# get dem points
x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(primary_swath_model)
#%%
burst_ids, read_rois_no_correc, write_rois_no_correc, out_shape = primary_swath_model.get_read_write_rois(
    roi_in_swath)

# construct burst models with appropriate corrections
primary_burst_models = [eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            primary_bursts_meta[bid], bistatic_correction=True,
            full_bistatic_correction_reference=ref_meta,
            apd_correction=True,
            intra_pulse_correction=True) for bid in burst_ids]

rows_no_correc_global, cols_no_correc_global,\
rows_correc_global, cols_correc_global, pts_in_burst_mask,\
    burst_resampling_matrices = \
     eos.products.sentinel1.regist.primary_registration_estimation(
        primary_swath_model, primary_burst_models, x, y, alt, crs, burst_ids)


debursted_crop, read_rois_correc, resamplers =  \
    eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
        read_rois_no_correc, burst_ids, primary_swath_model,
        primary_swath_model, burst_resampling_matrices,
        primary_bursts_meta, image_reader,
        write_rois_no_correc, out_shape,
        get_complex)

#%% plots 
plt.figure() 
plt.imshow(np.abs(debursted_crop), cmap='gray', 
           vmin=np.nanpercentile(np.abs(debursted_crop),10), 
           vmax=np.nanpercentile(np.abs(debursted_crop),90)
    ) 
plt.show()
