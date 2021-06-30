import numpy as np
import os
import eos.products.sentinel1

xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'
output_folder = '/home/rakiki/CMLA/experiments/EACOP/registration_within/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# list of our xmls
xml_paths = [os.path.join(xml_folder, p) for p in
             ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
              's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']
             ]
tiff_paths = [os.path.join(tiff_folder, p) for p in
              ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                  's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']
              ]

# read the xmls as strings
xml_content = []
for i in range(2):
    with open(xml_paths[i]) as f:
        xml_content.append(f.read())

# burst id in subswath
# here, by "chance", the 3rd burst is the same geographical location in both products
burst_id = 3
# Region of interest inside the burst in the primary (col, row, w, h)
dst_roi_in_burst = (90, 90, 1000, 1000)


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

# get dem points
x, y, raster, transform, crs = eos.sar.regist.dem_points(primary_burst_model,
                                                         source='SRTM30',
                                                         datum='ellipsoidal',
                                                         outfile=os.path.join(
                                                             output_folder, 'dem.tif')
                                                         )

# you can mask some pixels to speed up the projection
mask = np.random.binomial(n=1, p=0.1, size=x.shape).astype(bool)
x = x[mask]
y = y[mask]
raster = raster[mask]

# project in primary
row_primary, col_primary, _ = primary_burst_model.projection(
    x.ravel(), y.ravel(), raster.ravel(), crs=crs)

# project in secondary and estimate registration
A = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                        secondary_burst_model, x, y, raster, crs)


# resampler on the complex secondary burst
col_dst, row_dst, w_dst, h_dst = primary_burst_meta['burst_roi']
col_src, row_src, w_src, h_src = secondary_burst_meta['burst_roi']
resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(secondary_burst_meta,
                                                                         dst_burst_shape=(
                                                                             h_dst, w_dst),
                                                                         matrix=A, degree=11)

# warp the roi to the secondary, and add a margin of 5 pixels on each side
src_roi_in_burst = eos.sar.roi.warp_valid_rois(dst_roi_in_burst, (h_dst, w_dst), (h_src, w_src),
                                               A, margin=5)

# set the resampler to work on rois inside the burst
# this will adapt the resampling matrix to the roi origins
# and will adapt the deramping origin (since deramping depends on pixel position)
resampler.set_inside_burst(dst_roi_in_burst, src_roi_in_burst)

# translate the roi origin from the burst to the tiff coordinates
secondary_tiff_roi = eos.sar.roi.translate_roi(
    src_roi_in_burst, col_src, row_src)

# read the roi inside the secondary burst
secondary_burst_array = eos.sar.io.read_window(
    tiff_paths[1], secondary_tiff_roi)

# resample
resampled_secondary_array = resampler.resample(secondary_burst_array)

# now reset the resampler in case burst resampling is need
resampler.set_to_default_roi()

################## Do the interferogram (optional)

# translate roi origin from burst to tiff coordinates
primary_tiff_roi = eos.sar.roi.translate_roi(
    dst_roi_in_burst, col_dst, row_dst)

# read the roi inside the primary burst
primary_burst_array = eos.sar.io.read_window(tiff_paths[0],
                                             primary_tiff_roi)

# Do the interferogram
interf = primary_burst_array * np.conj(resampled_secondary_array)



################### If you wish to resample the amplitude 

# resample amplitude if you want to do this only (without phase )
# it is important to use resampler.matrix after the resampler has been set 
# on the two rois. The matrix will be adapted to the rois location. 

_, _, w, h = dst_roi_in_burst

resampled_secondary_amplitude = eos.sar.regist.apply_affine(matrix=resampler.matrix,
                                                            src_array=np.abs(
                                                                secondary_burst_array),
                                                            destination_array_shape=(h, w))
