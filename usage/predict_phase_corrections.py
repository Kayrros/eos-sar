import numpy as np
import os
import eos.products.sentinel1 as s1
import eos.sar
from eos.sar.roi import Roi 
#%% init swath models 
remote_test = False

if remote_test: 
    
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
else: 
    
    xml_folder = '/home/rakiki/CMLA/Time_series/data/annotation'
xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                     's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']

# list of our xmls
xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames ]
# read the xmls as strings
xml_content = []
for xml_path in xml_paths: 
        xml_content.append( eos.sar.io.read_xml_file(xml_path))


# Now extract the needed metadata
primary_bursts_meta = s1.metadata.extract_bursts_metadata(
    xml_content[0])
secondary_bursts_meta = s1.metadata.extract_bursts_metadata(
    xml_content[1])

# get the indices of the common bursts
prim_burst_ids, sec_burst_ids = s1.deburst.get_bursts_intersection(
    [len(primary_bursts_meta), len(secondary_bursts_meta)], 
    [primary_bursts_meta[0]['relative_burst_id'], secondary_bursts_meta[0]['relative_burst_id']]
)

# keep only the bursts intersecting
primary_bursts_meta = eos.sar.utils.filter_list(primary_bursts_meta, prim_burst_ids)
secondary_bursts_meta = eos.sar.utils.filter_list(secondary_bursts_meta, sec_burst_ids)

primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    primary_bursts_meta)

secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
    secondary_bursts_meta)


#%% Now estimate the registration matrix

# get a good approximation of the geometry of the swath
# with a margin of 10 pixels
margin=10
refined_geom, alts, mask = primary_swath_model.get_approx_geom(margin=margin)

# get dem points
x, y, raster, transform, crs = eos.sar.regist.dem_points(refined_geom)
#%% define the roi in the swath 
# define the roi in the primary swath
# Here, if you set a region of interest within the swath
# in the primary burst, only this region will be considered

primary_swath_roi = Roi(10000, 785, 3000, 3000)
# primary_swath_roi = None

#%% create a TopoCorrection instance
topo = eos.sar.geom_phase.TopoCorrection(primary_swath_model,
                                         [secondary_swath_model],
                                         grid_size=50, degree=7,
                                         )

#%% predict flat earth and correct it 
flat_earth = topo.flat_earth_image(primary_swath_roi)

flat_earth_interf_correction = np.exp(- 1j * flat_earth[0]).astype(np.complex64)
#%% Dem projection in radar coordinates
heights = eos.sar.dem_to_radar.dem_radarcoding(raster, transform,
                                               primary_swath_model,
                                               roi=primary_swath_roi,
                                               margin=margin)

#%% predict topographic phase
topo_phase = topo.topo_phase_image(heights, 
                                   primary_roi=primary_swath_roi)

topo_interf_correction = np.exp(- 1j * topo_phase[0]).astype(np.complex64)
