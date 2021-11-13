import os
import numpy as np
import glob
import boto3 
import tqdm
import tifffile

import eos.products.sentinel1
import eos.sar
from eos.sar.roi import Roi
#%%
def extract_keys(big_dict, list_keys): 
    o = {}
    for key in list_keys: 
        o[key] = big_dict[key]
    return o

def get_ref_metas(safe_dirs): 
    ref_swath = "iw2"
    xml_paths = [glob_single_file(os.path.join(s, "annotation", f"*{ref_swath}*vv*" ))\
                    for s in safe_dirs]
    xml_paths = sorted([x for x in xml_paths if x], key=get_date)
    
    xml_contents = [eos.sar.io.read_xml_file(xml_path) for xml_path in xml_paths]
    keys = ['slant_range_time',
    'samples_per_burst',
    'range_frequency']
    ref_metas = [extract_keys(eos.products.sentinel1.metadata.extract_burst_metadata(
        xml_content, 0), keys) for xml_content in xml_contents]
    return ref_metas

def get_date(xml_path): 
    return os.path.basename(xml_path).split('-')[4][:8]

def glob_single_file(pattern): 
    list_results = glob.glob(pattern)
    if len(list_results): 
        return list_results[0]
    
def client_s3():
    client_s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url="https://" + os.environ["AWS_S3_ENDPOINT"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    return client_s3

def bool_to_str(tag): 
    if tag: 
        return "True"
    else: 
        return "False"
    
#%%
experiment_path = "/home/rakiki/CMLA/experiments/Swath"

safe_path = os.path.join(experiment_path, 
                         "Newcastle/safe_data")

safe_dirs = glob.glob(os.path.join(safe_path, "S1*.SAFE")) 

ref_metas = get_ref_metas(safe_dirs) # 1 dict per product

swath = 'iw1'

# image readers 
tiff_paths = [glob_single_file(os.path.join(s, "measurement", f"*{swath}*vv*tiff" ))\
                for s in safe_dirs]
tiff_paths = sorted([t for t in tiff_paths if t], key=get_date)
image_readers = [eos.sar.io.open_image(tiff_path) for tiff_path in tiff_paths]

xml_paths = [glob_single_file(os.path.join(s, "annotation", f"*{swath}*vv*" ))\
                for s in safe_dirs]
xml_paths = sorted([x for x in xml_paths if x], key=get_date)

xml_contents = [eos.sar.io.read_xml_file(xml_path) for xml_path in xml_paths]
bursts_metas = [eos.products.sentinel1.metadata.extract_bursts_metadata(
xml_content) for xml_content in xml_contents]
num_bursts = [len(burst_meta) for burst_meta in bursts_metas]
burst_rel_ids = [burst_meta[0]['relative_burst_id'] for burst_meta in bursts_metas]
# get the ids of the common bursts on the time series for all acquisitions in a swath
burst_ids = eos.products.sentinel1.deburst.get_bursts_intersection(num_bursts, burst_rel_ids)
# construct swath models

filtered_burst_metas = [eos.sar.utils.filter_list(burst_meta, bids) \
                    for burst_meta, bids in zip(bursts_metas, burst_ids)]
# download the precise orbits
product_ids = [xml.split('/')[-3].split('.')[0] for xml in xml_paths]
for pid, bmeta in zip(product_ids, filtered_burst_metas): 
    assert all(b['state_vectors_origin'] == 'orbpre' for b in bmeta)
    assert eos.products.sentinel1.orbits.update_statevectors_using_our_bucket(client_s3(), pid, bmeta) == 'orbpoe'
    assert all(b['state_vectors_origin'] == 'orbpoe' for b in bmeta)
 
swath_models = [eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
            b_meta) for b_meta in filtered_burst_metas]


#%% deburst
bistatic_correction = [False, False, True, True]
apd_correction = [False, True, True, True]
intra_pulse_correction = [False, False, False, True]

def save_array(out_path, array, date): 
    out_fname = f"{date}.tif"
    tifffile.imsave(os.path.join(out_path, out_fname), array)   
    
p_id = 0  
# roi_in_swath_no_correc = Roi(865, 8816, 2713, 2193)
# roi_in_swath_no_correc = Roi(500, 0, 100, 15000)
roi_in_swath_no_correc = Roi(8249, 1781, 1409, 1785)
# roi_in_swath_no_correc = None

get_complex = False
global_rows_fit = True
primary_swath_model = swath_models[p_id]
x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(primary_swath_model)

for b_cor, apd_cor, intra_cor in zip(bistatic_correction, apd_correction, intra_pulse_correction):

    out_f_name = f"apd_{bool_to_str(apd_cor)}_bistatic_{bool_to_str(b_cor)}_intrapulse_{bool_to_str(intra_cor)}"
    out_path = os.path.join(experiment_path, "debursted_Newcastle", out_f_name)
    if not os.path.exists(out_path): 
        os.makedirs(out_path)
        
    burst_ids, read_rois_no_correc, write_rois_no_correc, out_shape = primary_swath_model.get_read_write_rois(
        roi_in_swath_no_correc)
    
    # construct burst models with appropriate corrections
    primary_burst_models = [eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
                filtered_burst_metas[p_id][bid], bistatic_correction=b_cor,
                full_bistatic_correction_reference=ref_metas[p_id],
                apd_correction=apd_cor,
                intra_pulse_correction=intra_cor) for bid in burst_ids]
    
    rows_no_correc_global, cols_no_correc_global,\
    rows_correc_global, cols_correc_global, pts_in_burst_mask,\
        burst_resampling_matrices = \
         eos.products.sentinel1.regist.primary_registration_estimation(
            primary_swath_model, primary_burst_models, x, y, alt, crs, burst_ids)
    
    
    debursted_crop =  \
        eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
            read_rois_no_correc, burst_ids, primary_swath_model,
            primary_swath_model, burst_resampling_matrices,
            filtered_burst_metas[p_id], image_readers[p_id],
            write_rois_no_correc, out_shape,
            get_complex)
    
    save_array(out_path, debursted_crop,  get_date(xml_paths[p_id]))

    for i in tqdm.trange(1, len(swath_models)):
        secondary_swath_model = swath_models[i]
        secondary_burst_models = [eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
                    filtered_burst_metas[i][bid], bistatic_correction=b_cor,
                    full_bistatic_correction_reference=ref_metas[i],
                    apd_correction=apd_cor,
                    intra_pulse_correction=intra_cor) for bid in burst_ids]
        
        burst_resampling_matrices = \
            eos.products.sentinel1.regist.secondary_registration_estimation(
                secondary_swath_model, secondary_burst_models,  x, y, alt, crs,
                burst_ids, pts_in_burst_mask, primary_swath_model,  rows_no_correc_global, 
                cols_no_correc_global, global_rows_fit=global_rows_fit )
        debursted_crop = \
            eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
                read_rois_no_correc, burst_ids, primary_swath_model,
                secondary_swath_model, burst_resampling_matrices,
                filtered_burst_metas[i], image_readers[i],
                write_rois_no_correc, out_shape,
                get_complex=get_complex)
        save_array(out_path, debursted_crop,  get_date(xml_paths[i]))
#%% Optional, sanity check, Compute mean img
for b_cor, apd_cor, intra_cor in zip(bistatic_correction, apd_correction, intra_pulse_correction): 
    out_f_name = f"apd_{bool_to_str(apd_cor)}_bistatic_{bool_to_str(b_cor)}_intrapulse_{bool_to_str(intra_cor)}"
    out_path = os.path.join(experiment_path, "mean_debursted_Newcastle", out_f_name)
    if not os.path.exists(out_path): 
        os.makedirs(out_path)
    in_path = os.path.join(experiment_path, "debursted_Newcastle", out_f_name)
    img_paths = sorted(os.listdir(in_path))
    img_paths = [os.path.join(in_path, im_path) for im_path in img_paths]
    mean = 0
    for img in img_paths: 
        im = tifffile.imread(img)
        mean += np.abs(im)**2 
    mean /= len(img_paths)
    mean = np.sqrt(mean)
    tifffile.imsave(os.path.join(out_path, "mean.tif"), mean) 
#%% Optional, sanity check, write burst mask 
mask_val = True
mask = np.zeros(out_shape, dtype=bool)
bmask = []
for roi in write_rois_no_correc:
    if mask_val: 
        bmask.append(np.ones(roi.get_shape(), dtype = bool))
    else: 
        bmask.append(np.zeros(roi.get_shape(), dtype=bool))
    mask_val = not mask_val
fmask = eos.products.sentinel1.deburst.stitch_arrays(bmask, write_rois_no_correc,  out_shape)
tifffile.imsave(os.path.join(experiment_path, "mask.tif"), fmask) 
