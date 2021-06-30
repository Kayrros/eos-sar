import numpy as np
import os
import eos.products.sentinel1
import eos.sar

xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
tiff_folder = '/home/rakiki/CMLA/Time_series/data/measurement'
output_folder = '/home/rakiki/CMLA/experiments/EACOP/registration/'
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
for xml_p in xml_paths:
    with open(xml_p) as f:
        xml_content.append(f.read())

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

# get dem points
x, y, raster, transform, crs = eos.sar.regist.dem_points(
    primary_burst_model,
    source='SRTM30',
    datum='ellipsoidal',
    outfile=os.path.join(output_folder, 'dem.tif')
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
                                        secondary_burst_model, x, y, raster, crs
                                        )
# Now read the secondary array
secondary_burst_array = eos.sar.io.read_window(
    tiff_paths[1], secondary_burst_meta['burst_roi'])

# resample the complex secondary burst
_, _, w, h = primary_burst_meta['burst_roi']

# create a resampler instance
resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(
    secondary_burst_meta, dst_burst_shape=(h, w),
    matrix=A, degree=11)

# resample
resampled_secondary_array = resampler.resample(secondary_burst_array)

# Do the interferogram (optional)
# read
primary_burst_array = eos.sar.io.read_window(tiff_paths[0],
                                             primary_burst_meta['burst_roi'])

interf = primary_burst_array * np.conj(resampled_secondary_array)

# Only resample the amplitude if you want
resampled_secondary_amplitude = eos.sar.regist.apply_affine(
    matrix=A,
    src_array=np.abs(secondary_burst_array),
    destination_array_shape=(h, w))
