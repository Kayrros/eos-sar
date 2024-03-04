from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

import eos.dem
from eos.sar import (  # type: ignore
    model,
    roi,
    simulator,
)


def normalize(
    raster: NDArray[Any],
    simulation: NDArray[np.float32],
    shadow_threshold: float = 0.05,
    shadow_value: Optional[float] = 0.0,
) -> NDArray[Any]:
    normalized = np.sqrt(np.abs(raster) ** 2 / (simulation + 1e-30))

    if shadow_value is not None:
        normalized[simulation < shadow_threshold] = shadow_value

    # TODO: check if it requires normalization with incidence angle as well
    return normalized


class RadiometricTerrainCorrector:
    def __init__(
        self,
        proj_model: model.SensorModel,
        dem: eos.dem.DEM,
        roi: roi.Roi,
        simulator_kwargs={},
    ):
        self.simulator = simulator.SARSimulator(proj_model, dem, **simulator_kwargs)
        self.roi = roi
        self._simulation = None

    def apply(self, raster: np.ndarray):
        sim = self.get_simulation()
        assert raster.shape == sim.shape
        return normalize(raster, sim)

    def get_simulation(self) -> NDArray[np.float32]:
        if self._simulation is None:
            self._simulation = self.simulator.simulate(self.roi).astype(np.float32)
        assert self._simulation is not None
        return self._simulation
