import numpy as np
import os
import eos.products.sentinel1 as s1
import eos.sar

remote_test = True

if remote_test: 
    
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'

    tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'
    # prepare oio config 
    prof_name = 'oio'
    en_url = 'https://s3.kayrros.org'
    
else: 
    
    xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
    tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'
    # just set oio vars to None in this case
    prof_name = en_url = None

xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                     's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']

tiff_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                  's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']

# list of our xmls
xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames ]

tiff_paths = [os.path.join(tiff_folder, p) for p in tiff_basenames]

image_readers = [eos.sar.io.open_image(p, profile_name=prof_name, 
                                       endpoint_url=en_url)
                 for p in tiff_paths]

# read the xmls as strings
xml_content = []
for xml_path in xml_paths: 
        xml_content.append( eos.sar.io.read_xml_file(
                                xml_path, profile_name=prof_name,
                                endpoint_url=en_url))

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

#%%
keywords = {'degree': 11,
            'bistatic_correction': True,
            'apd_correction': True,
            'max_iterations': 20,
            'tolerance': 0.001}

primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta, **keywords)

secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    secondary_bursts_meta, **keywords)

# Now estimate the registration matrix
# get dem points
x, y, raster, transform, crs = eos.sar.regist.dem_points(primary_swath_model,
                                                         source='SRTM30',
                                                         datum='ellipsoidal',
                                                         )
# you can mask some pixels to speed up the projection
mask = np.random.binomial(n=1, p=0.1, size=x.shape).astype(bool)
x = x[mask]
y = y[mask]
raster = raster[mask]

# project in primary
row_primary, col_primary, _ = primary_swath_model.projection(
    x.ravel(), y.ravel(), raster.ravel(), crs=crs)

# project in secondary and estimate registration
A_swath = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                              secondary_swath_model, x, y,
                                              raster, crs
                                              )

# define the roi in the primary swath
# Here, if you set a region of interest within the swath
# in the primary burst, only this region will be considered
primary_swath_roi = (0, 1000, 1000, 3000)

# if you set it to None, the whole swath will be considered
# Change the comment here to see what happens!
# primary_swath_roi = None

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

# if you wish to do the interferogram
interf = primary_debursted_crop * np.conj(secondary_debursted_crop)

#%% plots 
plt.figure() 
plt.imshow(np.abs(primary_debursted_crop), cmap='gray', 
           vmin=np.percentile(np.abs(primary_debursted_crop),10), 
           vmax=np.percentile(np.abs(primary_debursted_crop),90)
    ) 
plt.show()

plt.figure() 
plt.imshow(np.abs(secondary_debursted_crop), cmap='gray', 
           vmin=np.nanpercentile(np.abs(secondary_debursted_crop),10), 
           vmax=np.nanpercentile(np.abs(secondary_debursted_crop),90)
    ) 
plt.show()

plt.figure() 
plt.imshow(np.angle(interf), cmap='jet') 
plt.show()
