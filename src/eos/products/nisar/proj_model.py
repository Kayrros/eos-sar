from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import h5py
import numpy as np
import pyproj
from numpy.typing import ArrayLike
from typing_extensions import override

from eos.products.nisar.metadata import (
    Frequency,
    NisarRSLCMetadata,
)
from eos.sar import coordinates
from eos.sar.model import Arrayf64, CoordArrayLike, SensorModel
from eos.sar.model_helper import GenericSensorModelHelper
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector


@dataclass(frozen=True)
class NisarModel(SensorModel):
    generic_model: GenericSensorModelHelper
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float

    @staticmethod
    def from_metadata(
        meta: NisarRSLCMetadata,
        frequency: Frequency,
        orbit: Orbit,
        corrector: Corrector = Corrector(),
        max_iterations: int = 20,
        tolerance: float = 0.0001,
    ) -> NisarModel:
        frequency_meta = meta.frequency_a if frequency == "A" else meta.frequency_b
        assert frequency_meta is not None, f"Frequency {frequency} metadata is None"

        coordinate = coordinates.SLCCoordinate(
            first_row_time=meta.azimuth_time_first,
            first_col_time=frequency_meta.first_col_time,
            azimuth_frequency=meta.azimuth_frequency,
            range_frequency=frequency_meta.range_frequency,
        )

        projection_tolerance = float(tolerance / np.linalg.norm(orbit.sv[0].velocity))
        approx_centroid_lon, approx_centroid_lat = np.mean(meta.approx_geom, axis=0)

        generic_model = GenericSensorModelHelper(
            orbit=orbit,
            coordinate=coordinate,
            azt_init=float(coordinate.to_azt(meta.height / 2)),
            projection_tolerance=projection_tolerance,
            localization_tolerance=tolerance,
            max_iterations=max_iterations,
            coord_corrector=corrector,
            approx_centroid_lon=approx_centroid_lon,
            approx_centroid_lat=approx_centroid_lat,
        )

        return NisarModel(
            generic_model=generic_model,
            w=frequency_meta.width,
            h=meta.height,
            orbit=orbit,
            wavelength=frequency_meta.wavelength,
        )

    @override
    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        return self.generic_model.to_azt_rng(row, col)

    @override
    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        return self.generic_model.to_row_col(azt, rng)

    @override
    def projection(
        self,
        x: CoordArrayLike,
        y: CoordArrayLike,
        alt: CoordArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        azt_init: Optional[ArrayLike] = None,
        as_azt_rng: bool = False,
    ) -> tuple[CoordArrayLike, CoordArrayLike, CoordArrayLike]:
        return self.generic_model.projection(
            x, y, alt, crs, vert_crs, azt_init, as_azt_rng
        )

    @override
    def localization(
        self,
        row: CoordArrayLike,
        col: CoordArrayLike,
        alt: CoordArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        x_init: Optional[ArrayLike] = None,
        y_init: Optional[ArrayLike] = None,
        z_init: Optional[ArrayLike] = None,
    ) -> tuple[CoordArrayLike, CoordArrayLike, CoordArrayLike]:
        return self.generic_model.localization(
            row, col, alt, crs, vert_crs, x_init, y_init, z_init
        )


def main(h5_file_path, frequency: Frequency = "A") -> NisarModel:
    """
    Example usage
    """
    with h5py.File(h5_file_path, "r") as ds:
        metadata = NisarRSLCMetadata.parse_metadata(ds)
    orbit = Orbit(sv=metadata.state_vectors, degree=11)
    model = NisarModel.from_metadata(metadata, frequency, orbit)
    return model


if __name__ == "__main__":
    import fire

    fire.Fire(main)
