import numpy as np

from eos.sar import model, roi, simulator


def normalize(raster, simulation):
    normalized = np.sqrt(np.abs(raster)**2 / (simulation + 1e-30))
    # TODO: check if it requires normalization with incidence angle as well
    normalized[simulation < 0.05] = 0
    return normalized


class RadiometricTerrainCorrector:

    def __init__(self,
                 proj_model: model.SensorModel,
                 roi: roi.Roi,
                 simulator_kwargs={},
                 ):
        self.simulator = simulator.SARSimulator(proj_model, **simulator_kwargs)
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
