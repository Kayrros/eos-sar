from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyproj
from numpy.typing import ArrayLike
from typing_extensions import override

from eos.products.capella.metadata import CapellaSLCMetadata
from eos.sar import coordinates
from eos.sar.const import LIGHT_SPEED_M_PER_SEC as C
from eos.sar.model import Arrayf64, CoordArrayLike, SensorModel
from eos.sar.model_helper import GenericSensorModelHelper
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector
from eos.sar.roi import Roi


@dataclass(frozen=True)
class CapellaSLCModel(SensorModel):
    generic_model: GenericSensorModelHelper
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float

    @staticmethod
    def from_metadata(
        meta: CapellaSLCMetadata,
        orbit: Orbit,
        corrector: Corrector = Corrector(),
        max_iterations: int = 20,
        tolerance: float = 0.0001,
    ) -> CapellaSLCModel:
        coordinate = coordinates.SLCCoordinate(
            first_row_time=meta.first_line_time_since_ref,
            first_col_time=(2.0 * meta.starting_range) / C,
            azimuth_frequency=1.0 / meta.delta_line_time,
            range_frequency=C / (2 * meta.range_pixel_size),
        )

        projection_tolerance = float(tolerance / np.linalg.norm(orbit.sv[0].velocity))

        # Get the approximate location of the center of the sensor model
        centroid = meta.center_pixel_target_position
        transformer = pyproj.Transformer.from_crs(
            "epsg:4978", "epsg:4326", always_xy=True
        )
        centroid_lon, centroid_lat, _ = transformer.transform(
            centroid[0], centroid[1], centroid[2]
        )

        generic_model = GenericSensorModelHelper(
            orbit=orbit,
            coordinate=coordinate,
            azt_init=float(coordinate.to_azt(meta.height / 2)),
            projection_tolerance=projection_tolerance,
            localization_tolerance=tolerance,
            max_iterations=max_iterations,
            coord_corrector=corrector,
            approx_centroid_lon=centroid_lon,
            approx_centroid_lat=centroid_lat,
        )

        return CapellaSLCModel(
            generic_model=generic_model,
            w=meta.width,
            h=meta.height,
            orbit=orbit,
            wavelength=meta.wavelength,
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

    def to_cropped_model(self, roi: Roi):
        coordinate_prev = self.generic_model.coordinate
        assert isinstance(coordinate_prev, coordinates.SLCCoordinate)  # for mypy

        first_col_time = (
            coordinate_prev.first_col_time + roi.col / coordinate_prev.range_frequency
        )
        first_row_time = (
            coordinate_prev.first_row_time + roi.row / coordinate_prev.azimuth_frequency
        )

        coordinate = coordinates.SLCCoordinate(
            first_row_time=first_row_time,
            first_col_time=first_col_time,
            azimuth_frequency=coordinate_prev.azimuth_frequency,
            range_frequency=coordinate_prev.range_frequency,
        )

        # estimate the lon/lat center of the crop
        # it is only an approximation, so we can use alt=0.0
        center_x = roi.col + roi.w // 2
        center_y = roi.row + roi.h // 2
        approx_centroid_lon, approx_centroid_lat, _ = self.localization(
            center_y, center_x, 0.0
        )

        generic_model = GenericSensorModelHelper(
            orbit=self.generic_model.orbit,
            coordinate=coordinate,
            azt_init=float(coordinate.to_azt(roi.h / 2)),
            projection_tolerance=self.generic_model.projection_tolerance,
            localization_tolerance=self.generic_model.localization_tolerance,
            max_iterations=self.generic_model.max_iterations,
            coord_corrector=self.generic_model.coord_corrector,
            approx_centroid_lon=approx_centroid_lon,
            approx_centroid_lat=approx_centroid_lat,
        )

        return CapellaSLCModel(
            generic_model=generic_model,
            w=roi.w,
            h=roi.h,
            orbit=self.generic_model.orbit,
            wavelength=self.wavelength,
        )
