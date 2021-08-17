import os
import eos.products.sentinel1


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

xml_basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
tiff_basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff'
                 
# list of our xmls
xml_path = os.path.join(xml_folder, xml_basename)

tiff_path = os.path.join(tiff_folder, tiff_basename)

image_reader = eos.sar.io.open_image(tiff_path, profile_name=prof_name, 
                                       endpoint_url=en_url)


# read the xmls as strings
xml_content = eos.sar.io.read_xml_file(xml_path, profile_name=prof_name,
                                endpoint_url=en_url)
# Now extract the needed metadata
primary_bursts_meta = eos.products.sentinel1.metadata.extract_bursts_metadata(
    xml_content)

# construct primary swath model
primary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

# # deburst the whole swath
debursted_swath, burst_ids, rois_read, rois_write = eos.products.sentinel1.deburst.deburst_in_primary_swath(
    primary_swath_model, image_reader)

# If you wish to deburst a "crop" defined by a roi in the swath coordinates
roi_in_swath = (500, 750, 1000, 3000)

# deburst
debursted_crop, burst_ids, rois_read, rois_write = eos.products.sentinel1.deburst.deburst_in_primary_swath(
    primary_swath_model, image_reader, roi_in_swath)

# burst_ids are the burst ids covered by the roi ( O based in the swath)
# rois_read are the regions that were read from the tiff 
# rois_write are the rois where the read patches were written in the crop

#%% plots 
import matplotlib.pyplot as plt 

plt.figure() 
plt.imshow(np.abs(debursted_crop), cmap='gray', 
           vmin=np.percentile(np.abs(debursted_crop),10), 
           vmax=np.percentile(np.abs(debursted_crop),90)
    ) 
plt.show()

plt.figure() 
plt.imshow(np.abs(debursted_swath), cmap='gray', 
           vmin=np.percentile(np.abs(debursted_crop),10), 
           vmax=np.percentile(np.abs(debursted_crop),90)
    ) 
plt.show()
