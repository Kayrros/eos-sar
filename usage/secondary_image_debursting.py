import numpy as np
import os
from matplotlib import pyplot as plt
import eos.products.sentinel1 as s1
import eos.sar
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

get_complex = False # whether to deal with complex images
global_rows_fit=False # wether to fit a resampling matrix globaly on rows (usefull for esd method)
remote_test = True

if remote_test: 
    
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'

else: 
    
    xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
    tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'

xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                     's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']

ref_basenames = ['s1b-iw2-slc-vv-20190803t164006-20190803t164034-017424-020c57-005.xml',
                 's1a-iw2-slc-vv-20190809t164051-20190809t164117-028495-033896-005.xml']

tiff_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                  's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']

# list of our xmls
xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames ]

tiff_paths = [os.path.join(tiff_folder, p) for p in tiff_basenames]

image_readers = [eos.sar.io.open_image(p) for p in tiff_paths]

# read the xmls as strings
xml_content = [eos.sar.io.read_xml_file(xml_path) for xml_path in xml_paths]

# Now extract the needed metadata
primary_bursts_meta = s1.metadata.extract_bursts_metadata(
    xml_content[0])
secondary_bursts_meta = s1.metadata.extract_bursts_metadata(
    xml_content[1])

ref_metas = get_ref_metas([os.path.join(xml_folder, ref_base) for ref_base in ref_basenames])

# get the indices of the common bursts
prim_burst_ids, sec_burst_ids = s1.deburst.get_bursts_intersection(
    [len(primary_bursts_meta), len(secondary_bursts_meta)], 
    [primary_bursts_meta[0]['relative_burst_id'], secondary_bursts_meta[0]['relative_burst_id']]
)

# keep only the bursts intersecting
primary_bursts_meta = eos.sar.utils.filter_list(primary_bursts_meta, prim_burst_ids)
secondary_bursts_meta = eos.sar.utils.filter_list(secondary_bursts_meta, sec_burst_ids)

#%%

primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    secondary_bursts_meta)

#%%
# get dem points
x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(primary_swath_model)
#%%
# define the roi in the primary swath
# Here, if you set a region of interest within the swath
# in the primary burst, only this region will be considered
primary_swath_roi = Roi(0, 1000, 1000, 3000)

# if you set it to None, the whole swath will be considered
# Change the comment here to see what happens!
# primary_swath_roi = None
burst_ids, read_rois_no_correc, write_rois_no_correc, out_shape = primary_swath_model.get_read_write_rois(
    primary_swath_roi)

# construct burst models with appropriate corrections
primary_burst_models = [eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            primary_bursts_meta[bid], bistatic_correction=True,
            full_bistatic_correction_reference=ref_metas[0],
            apd_correction=True,
            intra_pulse_correction=True) for bid in burst_ids]

rows_no_correc_global, cols_no_correc_global,\
rows_correc_global, cols_correc_global, pts_in_burst_mask,\
    burst_resampling_matrices = \
     eos.products.sentinel1.regist.primary_registration_estimation(
        primary_swath_model, primary_burst_models, x, y, alt, crs, burst_ids)


primary_debursted_crop, read_rois_correc, resamplers =  \
    eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
        read_rois_no_correc, burst_ids, primary_swath_model,
        primary_swath_model, burst_resampling_matrices,
        primary_bursts_meta, image_readers[0],
        write_rois_no_correc, out_shape,
        get_complex)

secondary_burst_models = [eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            secondary_bursts_meta[bid], bistatic_correction=True,
            full_bistatic_correction_reference=ref_metas[1],
            apd_correction=True,
            intra_pulse_correction=True) for bid in burst_ids]

burst_resampling_matrices = \
    eos.products.sentinel1.regist.secondary_registration_estimation(
        secondary_swath_model, secondary_burst_models,  x, y, alt, crs,
        burst_ids, pts_in_burst_mask, primary_swath_model,  rows_no_correc_global, 
        cols_no_correc_global, global_rows_fit=global_rows_fit )
secondary_debursted_crop, read_rois_correc, resamplers = \
    eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
        read_rois_no_correc, burst_ids, primary_swath_model,
        secondary_swath_model, burst_resampling_matrices,
        secondary_bursts_meta, image_readers[1],
        write_rois_no_correc, out_shape,
        get_complex=get_complex)
#%%

if get_complex: 
    # if you wish to do the interferogram
    interf = primary_debursted_crop * np.conj(secondary_debursted_crop)

#%% plots 
plt.figure() 
plt.imshow(np.abs(primary_debursted_crop), cmap='gray', 
           vmin=np.nanpercentile(np.abs(primary_debursted_crop),10), 
           vmax=np.nanpercentile(np.abs(primary_debursted_crop),90)
    ) 
plt.show()

plt.figure() 
plt.imshow(np.abs(secondary_debursted_crop), cmap='gray', 
           vmin=np.nanpercentile(np.abs(secondary_debursted_crop),10), 
           vmax=np.nanpercentile(np.abs(secondary_debursted_crop),90)
    ) 
plt.show()


if get_complex: 
    plt.figure() 
    plt.imshow(np.angle(interf), cmap='jet') 
    plt.show()