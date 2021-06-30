import os
import eos.products.sentinel1

xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'

xml_path = os.path.join(xml_folder, basename)

tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'
basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff'
tiff_path = os.path.join(tiff_folder, basename)

# read the xml as a string
with open(xml_path) as f:
    xml_content = f.read()

# Now extract the needed metadata
primary_bursts_meta = eos.products.sentinel1.metadata.extract_bursts_metadata(
    xml_content)

# construct primary swath model
primary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

# deburst the whole swath
debursted_swath = eos.products.sentinel1.deburst.deburst_in_primary_swath(
    primary_swath_model, tiff_path)

# If you wish to deburst a "crop" defined by a roi in the swath coordinates
roi_in_swath = (500, 750, 1000, 3000)
# deburst
debursted_crop, burst_ids, rois_read, rois_write = eos.products.sentinel1.deburst.deburst_in_primary_swath(
    primary_swath_model, tiff_path, roi_in_swath)
