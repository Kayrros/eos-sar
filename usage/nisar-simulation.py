import logging

import numpy as np
from rasterio.warp import Resampling

from eos.dem import DEMStitcherSource
from eos.products.nisar.cropper import get_primary_crop
from eos.products.nisar.metadata import Frequency, Polarization
from eos.sar.io import RemoteH5Loader
from eos.sar.roi import Roi
from eos.sar.roi_provider import PrescribedRoiProvider
from eos.sar.simulator import SARSimulator

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # input
    rslc_path = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5"

    frequency: Frequency = "A"
    polarization: Polarization = "HH"

    # replace with GeometryRoiProvider or other if you want
    roi_provider = PrescribedRoiProvider(Roi(0, 0, 500, 500))
    dem_source = DEMStitcherSource()
    get_complex = False
    use_apd = False

    # code

    # first crop
    # replace LocalH5Loader if reading locally
    loader = RemoteH5Loader(rslc_path)

    with loader as h5py_file:
        crop = get_primary_crop(
            h5py_file,
            frequency,
            polarization,
            roi_provider,
            dem_source,
            get_complex=get_complex,
            use_apd=use_apd,
        )

    np.save("array.npy", crop.amplitude)

    # Then simulate
    # You can simulate without cropping the array, you just need to copy paste the
    # content of get_primary_crop and remove the array reading step (get proj_model and roi)
    dem = crop.model.fetch_dem(dem_source, crop.roi)
    simulator = SARSimulator(crop.model, dem, dem_resampling=Resampling.cubic_spline)
    simulation = simulator.simulate(crop.roi)

    np.save("simulation.npy", simulation)
