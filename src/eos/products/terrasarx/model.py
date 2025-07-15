from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyproj
from numpy.typing import ArrayLike
from typing_extensions import override

from eos.products.terrasarx.metadata import TSXMetadata, parse_tsx_metadata
from eos.sar import coordinates
from eos.sar.model import Arrayf64, CoordArrayLike, SensorModel
from eos.sar.model_helper import GenericSensorModelHelper
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector


@dataclass(frozen=True)
class TSXModel(SensorModel):
    generic_model: GenericSensorModelHelper
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float
    coordinate: coordinates.SLCCoordinate

    @staticmethod
    def from_metadata(
        meta: TSXMetadata, orbit: Orbit, corrector: Corrector = Corrector()
    ) -> TSXModel:
        coordinate = coordinates.SLCCoordinate(
            first_row_time=meta.image_start,
            first_col_time=meta.slant_range_time,
            azimuth_frequency=meta.azimuth_frequency,
            range_frequency=meta.range_frequency,
        )

        tolerance = 0.001
        projection_tolerance = float(tolerance / np.linalg.norm(orbit.sv[0].velocity))
        approx_centroid_lon, approx_centroid_lat = np.mean(meta.approx_geom, axis=0)

        generic_model = GenericSensorModelHelper(
            orbit=orbit,
            coordinate=coordinate,
            azt_init=meta.image_start,
            projection_tolerance=projection_tolerance,
            localization_tolerance=tolerance,
            max_iterations=20,
            coord_corrector=corrector,
            approx_centroid_lon=approx_centroid_lon,
            approx_centroid_lat=approx_centroid_lat,
        )

        return TSXModel(
            generic_model=generic_model,
            w=meta.width,
            h=meta.height,
            orbit=orbit,
            wavelength=meta.wavelength,
            coordinate=coordinate
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


def main(xml_annotation_file_path):
    """
    Example usage
    """
    metadata = parse_tsx_metadata(xml_annotation_file_path)
    orbit = Orbit(sv=metadata.state_vectors, degree=11)
    model = TSXModel.from_metadata(metadata, orbit)
    return model


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(main)
