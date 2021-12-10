import numpy as np
import os
import eos.products.sentinel1
import eos.sar

remote_test = True

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

# burst id in subswath
# here, by "chance", the 4th burst is the same geographical location in both products
burst_id = 3

# Now extract the needed metadata
primary_burst_meta = eos.products.sentinel1.metadata.extract_burst_metadata(
    xml_content[0], burst_id)
secondary_burst_meta = eos.products.sentinel1.metadata.extract_burst_metadata(
    xml_content[1], burst_id)

# Now instantiate burst_model instances for projection/localization
primary_burst_model = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
    primary_burst_meta)
secondary_burst_model = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
    secondary_burst_meta)

# Now estimate the registration matrix

# get the sampled dem points 
x, y, raster, crs = eos.sar.regist.get_registration_dem_pts(
    primary_burst_model, sampling_ratio=0.01, 
    dem_source='SRTM30', dem_datum='ellipsoidal' )

# project in primary
row_primary, col_primary, _ = primary_burst_model.projection(
    x.ravel(), y.ravel(), raster.ravel(), crs=crs)

# project in secondary and estimate registration
A = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                        secondary_burst_model, x, y, raster, crs
                                        )
# Now read the secondary array
secondary_burst_array = eos.sar.io.read_window(
    image_readers[1],
    secondary_burst_model.burst_roi)

# resample the complex secondary burst
h, w = primary_burst_model.burst_roi.get_shape()

# create a resampler instance
resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(
    secondary_burst_meta, dst_burst_shape=(h, w),
    matrix=A)

# resample
resampled_secondary_array = resampler.resample(secondary_burst_array)

# Do the interferogram (optional)
# read
primary_burst_array = eos.sar.io.read_window(image_readers[0],
                                             primary_burst_model.burst_roi)

interf = primary_burst_array * np.conj(resampled_secondary_array)

# Only resample the amplitude if you want
resampled_secondary_amplitude = resampler.resample(np.abs(secondary_burst_array))

#%% plots 
import matplotlib.pyplot as plt 
fig_size = (20,15)
plt.figure(figsize=fig_size) 
plt.imshow(np.abs(primary_burst_array), cmap='gray', 
            vmin=np.percentile(np.abs(primary_burst_array),10), 
            vmax=np.percentile(np.abs(primary_burst_array),90))
plt.show()

plt.figure(figsize=fig_size) 
plt.imshow(np.abs(resampled_secondary_array), cmap='gray', 
           vmin=np.nanpercentile(np.abs(resampled_secondary_array),10), 
            vmax=np.nanpercentile(np.abs(resampled_secondary_array),90)) 
plt.show()

plt.figure(figsize=fig_size) 
plt.imshow(resampled_secondary_amplitude, cmap='gray', 
           vmin=np.nanpercentile(resampled_secondary_amplitude,10), 
            vmax=np.nanpercentile(resampled_secondary_amplitude,90)) 
plt.show()

plt.figure(figsize=fig_size)
plt.imshow(np.angle(interf), cmap='jet')
plt.show() 
