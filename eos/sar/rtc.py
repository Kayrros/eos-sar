import numpy as np

import eos.dem
from eos.sar import model, roi
from eos.sar import simulator  # type: ignore


def normalize(raster, simulation, shadow_threshold=0.05, shadow_value=0.0):
    normalized = np.sqrt(np.abs(raster)**2 / (simulation + 1e-30))

    if shadow_value is not None:
        normalized[simulation < shadow_threshold] = shadow_value

    # TODO: check if it requires normalization with incidence angle as well
    return normalized


class RadiometricTerrainCorrector:

    def __init__(self,
                 proj_model: model.SensorModel,
                 dem_source: eos.dem.DEMSource,
                 roi: roi.Roi,
                 simulator_kwargs={},
                 ):
        self.simulator = simulator.SARSimulator(proj_model, dem_source, **simulator_kwargs)
        self.roi = roi
        self._simulation = None

    def apply(self, raster: np.ndarray):
        sim = self.get_simulation()
        assert raster.shape == sim.shape
        return normalize(raster, sim)

    def get_simulation(self):
        if self._simulation is None:
            self._simulation = self.simulator.simulate(self.roi).astype(np.float32)
        return self._simulation
