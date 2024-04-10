from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pyproj
import xmltodict
from numpy.typing import ArrayLike
from typing_extensions import override

from eos.sar import coordinates
from eos.sar.const import LIGHT_SPEED_M_PER_SEC
from eos.sar.model import Arrayf32, SensorModel
from eos.sar.model_helper import GenericSensorModelHelper
from eos.sar.orbit import Orbit, StateVector
from eos.sar.projection_correction import Corrector


@dataclass(frozen=True)
class TSXMetadata:
    mission_id: Literal["TSX-1", "TDX-1", "PAZ-1"]
    state_vectors: list[StateVector]
    orbit_direction: Literal["ascending", "descending"]
    look_side: Literal["left", "right"]
    width: int
    height: int
    approx_geom: list[tuple[float, float]]
    image_start: float
    azimuth_frequency: float
    slant_range_time: float
    range_pixel_spacing: float
    azimuth_pixel_spacing: float
    range_sampling_rate: float
    wavelength: float

    @property
    def range_frequency(self) -> float:
        return LIGHT_SPEED_M_PER_SEC / (2.0 * self.range_pixel_spacing)

    @property
    def azimuth_time_interval(self) -> float:
        return 1.0 / self.azimuth_frequency

    @property
    def range_time_interval(self):
        return 1.0 / self.range_sampling_rate


def string_to_timestamp(s: str) -> float:
    """
    Convert a string representing a date and time to a float number.
    """
    # remove "Z" suffix
    if s.endswith("Z"):
        s = s[:-1]

    t = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    t = t.replace(tzinfo=datetime.timezone.utc)
    return t.timestamp()


def parse_tsx_metadata(xml_path: str) -> TSXMetadata:
    """
    Extract relevant metadata fields from TerraSAR-X XML metadata file.

    Args:
        xml_path (str): Path to a TerraSAR-X XML metadata file.

    Returns:
        TSXMetadata: TerraSAR-X metadata object
    """
    # Parse the XML metadata file into a dictionary
    with open(xml_path, "r") as src:
        metadata = xmltodict.parse(src.read())["level1Product"]

    # Extract relevant metadata fields
    raster_info = metadata["productInfo"]["imageDataInfo"]["imageRaster"]
    height = int(raster_info["numberOfRows"])
    width = int(raster_info["numberOfColumns"])

    mission_id = metadata["generalHeader"]["mission"]
    assert mission_id in ("TSX-1", "TDX-1", "PAZ-1")

    scene_info = metadata["productInfo"]["sceneInfo"]
    image_start = scene_info["start"]["timeUTC"]
    image_start = string_to_timestamp(image_start)
    slant_range_time = float(scene_info["rangeTime"]["firstPixel"])

    azimuth_period = float(raster_info["columnSpacing"]["#text"])
    azimuth_frequency = 1 / azimuth_period
    azimuth_pixel_spacing = float(
        metadata["productSpecific"]["complexImageInfo"]["projectedSpacingAzimuth"]
    )

    range_period = float(raster_info["rowSpacing"]["#text"])
    range_sampling_rate = 1 / range_period
    range_pixel_spacing = float(
        metadata["productSpecific"]["complexImageInfo"]["projectedSpacingRange"][
            "slantRange"
        ]
    )

    frequency = float(metadata["instrument"]["radarParameters"]["centerFrequency"])
    wavelength = LIGHT_SPEED_M_PER_SEC / frequency
    orbit_direction = metadata["productInfo"]["missionInfo"]["orbitDirection"].lower()
    look_side = metadata["productInfo"]["acquisitionInfo"]["lookDirection"].lower()

    assert orbit_direction in ("ascending", "descending")
    assert look_side in ("left", "right")

    # state vectors
    state_vectors: list[StateVector] = []

    for v in metadata["platform"]["orbit"]["stateVec"]:
        t = string_to_timestamp(v["timeUTC"])
        x = float(v["posX"])
        y = float(v["posY"])
        z = float(v["posZ"])
        vx = float(v["velX"])
        vy = float(v["velY"])
        vz = float(v["velZ"])

        sv = StateVector(time=t, position=(x, y, z), velocity=(vx, vy, vz))
        state_vectors.append(sv)

    # longitude, latitude bounding box
    approx_geom = [
        (float(c["lon"]), float(c["lat"])) for c in scene_info["sceneCornerCoord"]
    ]

    return TSXMetadata(
        width=width,
        height=height,
        mission_id=mission_id,
        state_vectors=state_vectors,
        orbit_direction=orbit_direction,
        look_side=look_side,
        approx_geom=approx_geom,
        image_start=image_start,
        azimuth_frequency=azimuth_frequency,
        slant_range_time=slant_range_time,
        range_pixel_spacing=range_pixel_spacing,
        azimuth_pixel_spacing=azimuth_pixel_spacing,
        range_sampling_rate=range_sampling_rate,
        wavelength=wavelength,
    )


@dataclass(frozen=True)
class TSXModel(SensorModel):
    generic_model: GenericSensorModelHelper
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float

    @staticmethod
    def from_metadata(
        meta: TSXMetadata, orbit_degree: int, corrector: Corrector = Corrector()
    ) -> TSXModel:
        coordinate = coordinates.SLCCoordinate(
            first_row_time=meta.image_start,
            first_col_time=meta.slant_range_time,
            azimuth_frequency=meta.azimuth_frequency,
            range_frequency=meta.range_frequency,
        )

        orbit = Orbit(sv=meta.state_vectors, degree=orbit_degree)
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
        )

    @override
    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        return self.generic_model.to_azt_rng(row, col)

    @override
    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        return self.generic_model.to_row_col(azt, rng)

    @override
    def projection(
        self,
        x: ArrayLike,
        y: ArrayLike,
        alt: ArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        azt_init: Optional[ArrayLike] = None,
        as_azt_rng: bool = False,
    ) -> tuple[Arrayf32, Arrayf32, Arrayf32]:
        return self.generic_model.projection(
            x, y, alt, crs, vert_crs, azt_init, as_azt_rng
        )

    @override
    def localization(
        self,
        row: ArrayLike,
        col: ArrayLike,
        alt: ArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        x_init: Optional[ArrayLike] = None,
        y_init: Optional[ArrayLike] = None,
        z_init: Optional[ArrayLike] = None,
    ) -> tuple[Arrayf32, Arrayf32, Arrayf32]:
        return self.generic_model.localization(
            row, col, alt, crs, vert_crs, x_init, y_init, z_init
        )


def main(xml_annotation_file_path):
    """
    Example usage
    """
    metadata = parse_tsx_metadata(xml_annotation_file_path)

    model = TSXModel.from_metadata(metadata, orbit_degree=11)
    return model


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(main)
