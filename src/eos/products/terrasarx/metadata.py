from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from typing import Any, Literal

import xmltodict

from eos.sar.const import LIGHT_SPEED_M_PER_SEC
from eos.sar.orbit import StateVector


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
    azimuth_time_interval: float
    azimuth_pixel_spacing: float
    slant_range_time: float
    range_time_interval: float
    range_pixel_spacing: float
    wavelength: float
    center_pixel_incidence_angle: float

    @property
    def azimuth_frequency(self) -> float:
        return 1.0 / self.azimuth_time_interval

    @property
    def range_frequency(self):
        return 1.0 / self.range_time_interval

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TSXMetadata:
        return TSXMetadata(
            mission_id=d["mission_id"],
            state_vectors=[
                StateVector.from_dict(sv_dict) for sv_dict in d["state_vectors"]
            ],
            orbit_direction=d["orbit_direction"],
            look_side=d["look_side"],
            width=d["width"],
            height=d["height"],
            approx_geom=d["approx_geom"],
            image_start=d["image_start"],
            azimuth_time_interval=d["azimuth_time_interval"],
            azimuth_pixel_spacing=d["azimuth_pixel_spacing"],
            slant_range_time=d["slant_range_time"],
            range_time_interval=d["range_time_interval"],
            range_pixel_spacing=d["range_pixel_spacing"],
            wavelength=d["wavelength"],
            center_pixel_incidence_angle=d["center_pixel_incidence_angle"],
        )


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

    azimuth_time_interval = float(raster_info["columnSpacing"]["#text"])
    azimuth_pixel_spacing = float(
        metadata["productSpecific"]["complexImageInfo"]["projectedSpacingAzimuth"]
    )

    range_time_interval = float(raster_info["rowSpacing"]["#text"])
    range_pixel_spacing = float(
        metadata["productSpecific"]["complexImageInfo"]["projectedSpacingRange"][
            "slantRange"
        ]
    )

    frequency = float(metadata["instrument"]["radarParameters"]["centerFrequency"])
    wavelength = LIGHT_SPEED_M_PER_SEC / frequency
    orbit_direction = metadata["productInfo"]["missionInfo"]["orbitDirection"].lower()
    look_side = metadata["productInfo"]["acquisitionInfo"]["lookDirection"].lower()
    center_pixel_incidence_angle = float(metadata["productInfo"]["sceneInfo"]["sceneCenterCoord"]["incidenceAngle"])

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
        azimuth_time_interval=azimuth_time_interval,
        azimuth_pixel_spacing=azimuth_pixel_spacing,
        slant_range_time=slant_range_time,
        range_time_interval=range_time_interval,
        range_pixel_spacing=range_pixel_spacing,
        wavelength=wavelength,
        center_pixel_incidence_angle=center_pixel_incidence_angle,
    )


def main(xml_annotation_file_path):
    """
    Example usage
    """
    metadata = parse_tsx_metadata(xml_annotation_file_path)
    return metadata


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(main)
