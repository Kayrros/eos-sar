#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import rasterio
import warnings

import eos.products.sentinel1 as s1
import eos.sar 
from eos.sar.roi import Roi


# ## The DATA
# The data that will be used for this experiment is a couple of S1A acquisitions
# 
# Reference: S1A_IW_SLC__1SDV_20211229T231926_20211229T231953_041230_04E66A_3DBE
# 
# Secondary: S1A_IW_SLC__1SDV_20220110T231926_20220110T231953_041405_04EC57_103E
# 
# They span an earthquake taking place at January 7 2022: M 6.6 - 113 km SW of Jinchang, China
# https://sarviews-hazards.alaska.edu/Event/e2dfcb22-e1a4-43d8-a17e-c6b175849463
# 
# To download the data, we use the script provided by ASF (after slight modifications)
# in a shell, run: 
#     
#     mkdir tutorial 
#     cd tutorial
#     python ../download_pair.py  # you will be asked for your ASF credentials
#     
# The two products will be downloaded and unzipped in the directory. The two corresponding orbits will also be downloaded.
# 
# The links to the files have been included in the download script manually. An open source library for finding the links will probably be made available.

# In[ ]:


# Input/Output helper functions


def pid_from_safe_dir(safe_dir):
    return os.path.splitext(os.path.basename(safe_dir))[0]


def pid_to_date(product_id):
    return product_id.split('_')[5][:8]


def glob_single_file(pattern):
    list_results = glob.glob(pattern)
    if len(list_results):
        return list_results[0]


def write_array(array, path):
    height, width = array.shape
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning)
        profile = dict(count=1,
                       width=width,
                       height=height,
                       dtype=array.dtype)
        with rasterio.open(path, "w", **profile) as f:
            f.write(array, 1)


# In[ ]:


workdir = "./tutorial"
swath = "iw2"
polarization = "vv"
calibration_method = "sigma"

# out path
out_path = os.path.join(workdir, "out_swath")
if not os.path.exists(out_path):
    os.makedirs(out_path)


# In[ ]:


safe_dirs = glob.glob(os.path.join(workdir, "data", "safes", "*.SAFE"))
safe_dirs = sorted(safe_dirs, key=lambda x: pid_to_date(pid_from_safe_dir(x)))

tiff_readers = []
bursts_metas = []
ref_metas = []
calibration_readers = []  # these readers will calibrate the data after reading
for safe_dir in safe_dirs:
    # path to the tiff
    tiff_path = glob_single_file(
        os.path.join(safe_dir, "measurement", f"*{swath}*{polarization}*tiff")
    )
    # then instantiate a reader (objects having function .read())
    reader = eos.sar.io.open_image(tiff_path)
    # get the path to the xml annotation
    xml_path = glob_single_file(os.path.join(
        safe_dir, "annotation", f"*{swath}*{polarization}*xml"))
    # read the file into a string xml_content
    xml_content = eos.sar.io.read_xml_file(xml_path)
    # parse the string and extract necessary burst metadata (list of dicts, one per burst)
    bursts_meta = s1.metadata.extract_bursts_metadata(
        xml_content)
    # We also need some metadata from the second swath to perform bistatic correction
    if swath == "iw2":
        first_iw2_burst_meta = bursts_meta[0]
    else:
        xml_path = glob_single_file(os.path.join(
            safe_dir, "annotation", f"*iw2*{polarization}*xml"))
        xml_content = eos.sar.io.read_xml_file(xml_path)
        first_iw2_burst_meta = s1.metadata.extract_burst_metadata(
            xml_content, 0)
    # extract interesting iw2 metadata
    keys = ['slant_range_time', 'samples_per_burst', 'range_frequency']
    ref_meta = {key: first_iw2_burst_meta[key] for key in keys}

    # For Calibration
    # get the path to the xml annotation
    calibration_xml_path = glob_single_file(os.path.join(
        safe_dir, "annotation", "calibration", f"calibration*{swath}*{polarization}*xml"))
    calibration_xml_content = eos.sar.io.read_xml_file(calibration_xml_path)

    noise_xml_path = glob_single_file(os.path.join(
        safe_dir, "annotation", "calibration", f"noise*{swath}*{polarization}*xml"))
    # read the file into a string xml_content
    noise_xml_content = eos.sar.io.read_xml_file(noise_xml_path)

    calibrator = s1.calibration.Sentinel1Calibrator(calibration_xml_content,
                                                    noise_xml_content)

    calibration_reader = s1.calibration.CalibrationReader(reader, calibrator,
                                                          calibration_method)

    # store all values in a list
    tiff_readers.append(reader)
    bursts_metas.append(bursts_meta)
    ref_metas.append(ref_meta)
    calibration_readers.append(calibration_reader)


# In[ ]:


# get the indices of the common bursts
num_bursts = [len(bursts_meta) for bursts_meta in bursts_metas]
burst_rel_ids = [bursts_meta[0]['relative_burst_id']
                 for bursts_meta in bursts_metas]
common_burst_ids = s1.deburst.get_bursts_intersection(
    num_bursts, burst_rel_ids)

local_dir = os.path.join(workdir, "data", "orb")
for safe_dir, bursts_meta, burst_ids in zip(safe_dirs, bursts_metas, common_burst_ids):
    # keep only the common bursts
    bursts_meta = eos.sar.utils.filter_list(bursts_meta, burst_ids)
    # After this step, i^th burst of primary will correspond to i^th burst of secondary
    product_id = pid_from_safe_dir(safe_dir)
    # Apply restituted orbits to the metadata dictionnary from a local folder containing orbits
    print("Applying orbits to ", product_id)
    orb_type = s1.orbits.update_statevectors_using_local_folder(
        local_dir, product_id, bursts_meta)
    print(orb_type, "applied to ", product_id)
    print("######################")


# ## Digital Elevation Model (DEM)
# 
# Many processings require access to a DEM. An interface to define how dem data should be accessed has been implemented in the form of a DEMSource object. Any object of this class should be able to query a dem at a certain (lon, lat) location, or to crop a dem from the given bounds.
# 
# Before running the cell below, ensure you have installed [srtm4](https://github.com/centreborelli/srtm4) (open source) or multidem (Kayrros package). Otherwise, you would need to implement you own source by inheriting from eos.dem.DEMSource.

# In[ ]:


# If you wish to localize a point without giving the altitude
# you can use a eos.dem.DEMSource to get a query function to a dem
dem_source = eos.dem.get_any_source()

alt = dem_source.elevation(20,40)
print(alt, "m")

raster, transform ,crs = dem_source.crop((20,10,20.03,10.03))
print(raster.shape)


# ## Model tutorial
# 
# The metadata has been read previously and stored in dictionnaries. From the metadata, it is possible to instantiate a eos.sar.model.SensorModel (this is the base class) object.
# 
# A "SensorModel" is an object that mainly has the responsability to perform geolocation operations.

# In[ ]:


# Let us  test a "swath" model
# A swath is defined as the minimal mosaic (minimal in height, width)
# containing the given bursts
test_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    bursts_metas[0]) # all bursts in the first product are used to get a swath

# You can also create a burst model
# the first burst of the first product
test_burst_model = s1.proj_model.burst_model_from_burst_meta(bursts_metas[0][0]) 


# In[ ]:


# Here we do a plot to show what a swath is 

import matplotlib.patches as patches
# Create figure and axes
fig, ax = plt.subplots(figsize=(10,10))

def add_rect(col, row, w, h, color='b'): 
    rect = patches.Rectangle((col, row), w, h, 
                         linewidth=1, edgecolor=color, facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
w, h = test_swath_model.w, test_swath_model.h
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

for bid in range(len(bursts_metas[0])):
    col, row = test_swath_model.burst_orig_in_swath(bid)
    h, w = test_swath_model.bursts_rois[bid].get_shape()
    if bid%2: 
        color = 'r'
    else: 
        color = 'b'
     # Create a Rectangle patch on the swath
    add_rect(col, row, w, h, color)
plt.show()


# In the plot, you can see that here, the swath is composed of 9 bursts. The bursts' color alternates between red and blue. You can see that the bursts overlap (at the end of a burst and the start of the next one). The bursts may also be slightly shifted in the column direction. The swath limits are defined as the minimal limits containing all the bursts.

# In[ ]:


# Let us test some geolocation functions
# a set of points in the image, in this case in the swath, can be geolocated
rows = np.round(np.random.rand(5) * (test_swath_model.h - 1))
cols = np.round(np.random.rand(5) * (test_swath_model.w - 1))
# for localization, you need to give the altitude to get the 3D point
# because the 3D point falling in a pixel is ambiguous without the altitude

# for ex., finding the 3D points on the ellipsoid for a set of image pixels
alts = np.zeros(5)
lons, lats, alts = test_swath_model.localization(rows, cols, alts)

# Alternatively, the location of a set of 3D points in the image can be found
rows_pred, cols_pred, incidence_pred = test_swath_model.projection(
    lons, lats, alts)

np.testing.assert_allclose(rows_pred, rows, atol=1e-2)
np.testing.assert_allclose(cols_pred, cols, atol=1e-2)

# If you wish to localize a point without giving the altitude
# you can use a eos.dem.DEMSource to get a query function to a dem
dem_source = eos.dem.get_any_source()

# Then, the first 3D point lying on the dem that corresponds to the pixel is returned
# by this function call
lons, lats, alts, masks = test_swath_model.localize_without_alt(
    rows, cols, elev=dem_source.elevation)

# In the same logic, the four image corners can be localized to get an estimation of the
# footprint of the image
approx_geom, alts, masks = test_swath_model.get_approx_geom()


# In[ ]:


# Setup models for the primary and secondary image

primary_bursts_meta = bursts_metas[0]
primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

secondary_bursts_meta = bursts_metas[1]
secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    secondary_bursts_meta)


# ## Region of interest
# 
# eos was designed to be able to work on restricted regions within an image. The region can be defined in geographic coordinates as well as coordinates within the image.

# In[ ]:


# We can start defining the area of interest

# you can define it as a list of coordinates (lon, lat)
input_geometry = [
    (101.63079789214869, 38.119580588719934),  # (lon, lat)
    (100.95483833640877, 38.21396403905346),
    (100.7799350202732, 37.47156838966397),
    (101.45975167385876, 37.37555836504315)
]

# and then find the region in the swath to be studied
# To do so, we use the model to geolocate (project) the 3D points into the image

x = [pt[0] for pt in input_geometry]
y = [pt[1] for pt in input_geometry]

# we need to find the alt to have the 3D coordinates
# we use an interface to access and interpolate dem data
# This is a SRTM90 interface
dem_source = eos.dem.SRTM4Source()
alt = dem_source.elevation(x, y)

# projection
rows, cols, incidences = primary_swath_model.projection(x, y, alt)

primary_swath_roi = Roi.from_bounds_tuple(Roi.points_to_bbox(rows, cols))

# verifiy that the region lies inside the swath boundaries
swath_shape = (primary_swath_model.h, primary_swath_model.w)
primary_swath_roi.make_valid(swath_shape, inplace=True)
print(primary_swath_roi)
# In this case, the region of interest was defined in (lon, lat) coordinates
# Then it was transformed to image coordinates

# Another way is for example to want to get the whole swath
# if you set primary_swath_roi to None, the whole swath will be considered
# Change the comment here to see what happens!
# primary_swath_roi = None
# Also, you can set the roi yourself, but it is hard to know
# how to set it correctly if you don't have a debursted swath tif image
# for ex.
# primary_swath_roi = Roi(col=48, row=569, w=16221, h=6022)


# ## Processing flags

# In[ ]:


get_complex = True  # whether to work with complex images, or just amplitudes

# whether to fit a resampling matrix globaly on rows (similarly on all bursts)
# (might be useful for esd method for example)
global_rows_fit = False

# Geolocation corrections
# If any correction is activated, then the bursts in the primary image
# will be resampled to correct these effects before stitching
# The estimation of the registration will also take these effects into account
# If all corrections are deactivated, the bursts are simply read and stitched in
# the primary image, and the registration will ignore the effects.
bistatic_correction = True
apd_correction = True
intra_pulse_correction = True

# Flag set for the tutorial, to test coherence
compute_coherence = True  

# Flag set for tutorial, to test calibration
calibrate = True 

if calibrate:
    image_readers = calibration_readers
else:
    image_readers = tiff_readers


# ## Burst Resampling matrices
# 
# To get the mosaic image, all data coming the from different bursts intersecting the region of interest must be read and resampled. Resampling occurs for the primary image in case any correction is enabled. Resampling always occurs for the secondary image to perform the registration.
# 
# In summary, bursts ( or data within a burst) are resampled one by one, and we need a resampling matrix per burst. 
# 
# These burst resampling matrices are estimated using the geolocation of several 3D points.

# In[ ]:


# get dem points
dem_path = os.path.join(out_path, "dem.tif")

# A dem covering the swath is downloaded and saved
# It is randomly sampled to get some points to use in the geometric registration

# buffer the region of interest by this margin in px
# because the (lon, lat) geometry is re-estimated
# the margin ensures a correct estimation of the geometry
margin = 500

# we can download the dem only on the bounds of the region of interest
x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
    primary_swath_model, roi=primary_swath_roi, margin=margin,
    dem=dem_source, outfile=dem_path)


# In[ ]:


#%% Burst resampling matrices estimation for the primary image


# Determine the burst ids (0 based, from the common burst list) intersecting the region of interest
# Also determine the regions where we need to read from the tiff
# and the regions where we need to write in the output image of shape out_shape
burst_ids, read_rois_no_correc, write_rois_no_correc, out_shape = primary_swath_model.get_read_write_rois(
    primary_swath_roi)

# For each burst id intersected by the roi, we need to get a resampling matrix
# The burst model is constructed, it is responsible for the geolocation of the
# points in the burst, and more specifically, the application of the corrections
# some corrections, like intra-pulse correction, only make sense at the burst level
# So a burst model is responsible for applying the corrections
primary_burst_models = [s1.proj_model.burst_model_from_burst_meta(
    primary_bursts_meta[bid], bistatic_correction=bistatic_correction,
    full_bistatic_correction_reference=ref_metas[0],
    apd_correction=apd_correction,
    intra_pulse_correction=intra_pulse_correction) for bid in burst_ids]

# The DEM points are projected, with and without corrections, and masks to
# indicate which pts are in a burst are also returned
# A resampling matrix per burst is fitted to correct the bursts geometrically
# burst_resampling_matrices is a dict containing a matrix per burst
rows_no_correc_global, cols_no_correc_global,    rows_correc_global, cols_correc_global, pts_in_burst_mask,    burst_resampling_matrices_prim =     s1.regist.primary_registration_estimation(
        primary_swath_model, primary_burst_models, x, y, alt, crs, burst_ids)


# In[ ]:


# The secondary models are constructed with the corrections
secondary_burst_models = [s1.proj_model.burst_model_from_burst_meta(
    secondary_bursts_meta[bid], bistatic_correction=bistatic_correction,
    full_bistatic_correction_reference=ref_metas[1],
    apd_correction=apd_correction,
    intra_pulse_correction=intra_pulse_correction) for bid in burst_ids]

# The dem points that were projected in ith primary burst are now projected
# in the ith secondary burst
# The burst resampling matrix is estimated
burst_resampling_matrices_sec =     s1.regist.secondary_registration_estimation(
        secondary_swath_model, secondary_burst_models,  x, y, alt, crs,
        burst_ids, pts_in_burst_mask, primary_swath_model,  rows_no_correc_global,
        cols_no_correc_global, global_rows_fit=global_rows_fit)


# ## Debursting
# ### Alias for:  Reading, Calibrating(optional), Resampling, Stitching
# The burst resampling matrices, as well as the regions defined by 
#     
#     burst_ids, read_rois_no_correc, write_rois_no_correc, out_shape
#   
# are used to read, calibrate(optional: if `calibrate` flag was true) resample and stitch the data. 

# In[ ]:


# The primary image crop is constructed, by reading from each burst
# resampling, and then stitching
primary_debursted_crop, read_rois_correc, resamplers =      s1.deburst.warp_rois_read_resample_deburst(
        read_rois_no_correc, burst_ids, primary_swath_model,
        primary_swath_model, burst_resampling_matrices_prim,
        primary_bursts_meta, image_readers[0],
        write_rois_no_correc, out_shape,
        get_complex)


# In[ ]:


# The secondary image crop is constructed, by reading from each burst
# resampling, and then stitching
secondary_debursted_crop, read_rois_correc, resamplers =     s1.deburst.warp_rois_read_resample_deburst(
        read_rois_no_correc, burst_ids, primary_swath_model,
        secondary_swath_model, burst_resampling_matrices_sec,
        secondary_bursts_meta, image_readers[1],
        write_rois_no_correc, out_shape,
        get_complex=get_complex)

# %% Write results
write_array(primary_debursted_crop, os.path.join(out_path, "primary_crop.tif"))
write_array(secondary_debursted_crop, os.path.join(
    out_path, "secondary_crop.tif"))


# ## Interferometry
# 
# The cell below will only run if `get_complex` flag was set to True. In this case, The debursted crops will be used to create an interferogram.
# 
# Also, the orbital (flat earth) phase component as well as the topographic phase component will be simulated and compensated.
# 
# It is also possible to compute the coherence. (`compute_coherence` flag)
# 

# In[ ]:


if get_complex:
    # if you wish to do the interferogram
    interf = primary_debursted_crop * np.conj(secondary_debursted_crop)
    write_array(interf, os.path.join(out_path, "interf.tif"))

    #  create a TopoCorrection instance
    # This object computes the orbital and topographic phase
    # geometric quantities (baseline, incidence..)
    # are evaluated at a uniform grid (50x50 pixels in this case)
    # and a 2D polynomial of degree (7 here) is fitted for those quantities
    topo = eos.sar.geom_phase.TopoCorrection(primary_swath_model,
                                             [secondary_swath_model],
                                             grid_size=50, degree=7,
                                             )

    # predict flat earth (orbital phase) and correction
    flat_earth = topo.flat_earth_image(primary_swath_roi)
    flat_earth_interf_correction = np.exp(- 1j *
                                          flat_earth[0]).astype(np.complex64)
    del flat_earth
    flattened = interf * flat_earth_interf_correction
    write_array(flat_earth_interf_correction,
                os.path.join(out_path, "flat_earth.tif"))
    write_array(flattened, os.path.join(out_path, "flattened.tif"))
    del flat_earth_interf_correction

    # re-read the saved dem (downloaded before at registration step)
    with rasterio.open(dem_path, 'r') as dem_db:
        raster = dem_db.read(1)
        transform = dem_db.transform
        left, bottom, right, top = dem_db.bounds
    geometry = [(left, top), (right, top), (right, bottom), (left, bottom)]
    # Dem projection in radar coordinates
    heights = eos.sar.dem_to_radar.dem_radarcoding(raster, transform,
                                                   primary_swath_model,
                                                   roi=primary_swath_roi,
                                                   approx_geometry=geometry,
                                                   margin=margin)

    # predict topographic phase and correction
    topo_phase = topo.topo_phase_image(heights,
                                       primary_roi=primary_swath_roi)

    topo_interf_correction = np.exp(- 1j * topo_phase[0]).astype(np.complex64)
    write_array(topo_interf_correction, os.path.join(out_path, "topo.tif"))
    del topo_phase

    corrected_interf = flattened * topo_interf_correction

    write_array(corrected_interf, os.path.join(out_path, "dinterf.tif"))

    if compute_coherence:
        # It is also possible to compute the coherence
        coher = eos.sar.coherence.on_pair(
            primary_debursted_crop, secondary_debursted_crop,
            # here, you can modify the filter size if you wish
            filter_size=(5, 21),
            might_contain_nans=True)
        write_array(coher, os.path.join(out_path, "coherence.tif"))


# ## Plot Results

# In[ ]:


def multilook(u, filter_size=(5,21)): 
    import scipy.ndimage as ndimage
    mask = np.isnan(u)
    mlooked = u.copy()
    mlooked[mask] = 0
    mlooked = ndimage.uniform_filter(mlooked, size=filter_size, mode="nearest")[::filter_size[0], ::filter_size[1]]
    return mlooked 

def display_amp(cmplx_img): 
    intensity = np.abs(cmplx_img)**2
    amp = np.sqrt(multilook(intensity))
    plt.figure()
    plt.imshow(amp, cmap='gray',
               vmin=np.nanpercentile(amp, 10),
               vmax=np.nanpercentile(amp, 90)
               )
    plt.colorbar()
    plt.show()

    
display_amp(primary_debursted_crop)

display_amp(secondary_debursted_crop)


# In[ ]:


if get_complex:
    def display_interf(interf):
        interf_disp = multilook(interf)
        plt.figure()
        plt.imshow(np.angle(interf_disp), cmap='jet', resample=False)
        plt.colorbar()
        plt.show()
        
    display_interf(interf)
    display_interf(flattened)
    display_interf(corrected_interf)

