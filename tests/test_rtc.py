import numpy as np
from eos.sar.simulator import SARSimulator
from rasterio.warp import Resampling

import eos.dem
from eos.dem import DEMStitcherSource
from eos.products import sentinel1
from eos.products.nisar.metadata import (
    Frequency,
    NisarRSLCMetadata,
)
from eos.products.nisar.proj_model import NisarModel
from eos.sar.io import RemoteH5Loader
from eos.sar.orbit import Orbit
from eos.sar.roi import Roi
from eos.sar.rtc import RadiometricTerrainCorrector


def test_rtc_simulation(tmp_path):
    xml_path = "./tests/data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml"
    with open(xml_path) as f:
        xml_content = f.read()
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id=1)
    orbit = Orbit(burst_meta.state_vectors)
    model = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta, orbit)
    roi = Roi(3000, 100, 1000, 1000)
    dem_source = eos.dem.get_any_source()
    dem = model.fetch_dem(dem_source, roi)

    rtc = RadiometricTerrainCorrector(model, dem, roi)
    sim = rtc.get_simulation()
    np.save(tmp_path / "sim.npy", sim)


def test_nisar_simulation(tmp_path):
    rslc_path = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5"

    frequency: Frequency = "A"

    # replace with GeometryRoiProvider or other if you want
    roi = Roi(0, 0, 500, 500)
    dem_source = DEMStitcherSource()

    loader = RemoteH5Loader(rslc_path)

    with loader as h5py_file:
        metadata = NisarRSLCMetadata.parse_metadata(h5py_file)
        orbit = Orbit(sv=metadata.state_vectors, degree=11)
        model = NisarModel.from_metadata(metadata, frequency, orbit)

    # Then simulate
    dem = model.fetch_dem(dem_source, roi)
    simulator = SARSimulator(model, dem, dem_resampling=Resampling.cubic_spline)
    simulation = simulator.simulate(roi)

    np.save(tmp_path / "sim_nisar.npy", simulation)
