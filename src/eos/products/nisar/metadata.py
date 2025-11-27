from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import h5py
import numpy as np
from shapely import from_wkt

from eos.sar.const import LIGHT_SPEED_M_PER_SEC
from eos.sar.orbit import StateVector


def parse_date_as_numpy_datetime64(date_str: str) -> np.datetime64:
    return np.datetime64(date_str.removesuffix("Z"), "ns")


@dataclass(frozen=True)
class NisarMetadata:
    radar_band: Literal["L", "S"]
    mission_id: str
    orbit_direction: Literal["ascending", "descending"]
    look_side: Literal["left", "right"]
    absolute_orbit_number: int
    relative_orbit_number: int
    frame_number: int
    approx_geom: list[tuple[float, float]]

    def __post_init__(self):
        assert self.radar_band in ["L", "S"], "Unrecognized radar band"
        assert self.orbit_direction in ["ascending", "descending"], (
            "Unrecognized orbit direction"
        )
        assert self.look_side in ["left", "right"], "Unrecognized look side"
        assert 0 < self.relative_orbit_number <= 173, (
            "Unrecognized relative orbit number"
        )
        assert 0 < self.frame_number <= 176, "Unrecognized frame number"


@dataclass(frozen=True)
class NisarFrequencyMetadata:
    slant_range_first: float
    slant_range_spacing: float
    ground_range_spacing: float
    acquired_center_frequency: float
    processed_center_frequency: float
    wavelength: float
    width: int
    polarizations: list[Literal["HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"]]
    ne_backscatter_dataset: Literal["nes0", "noiseEquivalentBackscatter"]
    ne_backscatter_azimuth_time: list[float]
    ne_backscatter_slant_range: list[float]
    ne_backscatter: dict[str, list[list[float]]]
    ref_timestamp: np.datetime64

    def __post_init__(self):
        assert all(
            polarization in ["HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"]
            for polarization in self.polarizations
        ), "Unrecognized polarizations"
        assert self.ne_backscatter_dataset in [
            "nes0",
            "noiseEquivalentBackscatter",
        ], "Unrecognized noise equivalent backscatter dataset"
        assert all(
            [
                polarization in self.polarizations
                for polarization in self.ne_backscatter.keys()
            ]
        )

    @staticmethod
    def parse_metadata(
        ds: h5py.File, frequency: Literal["A", "B"], radar_band: Literal["L", "S"]
    ) -> NisarFrequencyMetadata:
        # Swath
        frequency_group = f"science/{radar_band}SAR/RSLC/swaths/frequency{frequency}"
        slant_range_first = float(ds[f"{frequency_group}/slantRange"][0])
        slant_range_spacing = float(ds[f"{frequency_group}/slantRangeSpacing"][()])
        ground_range_spacing = float(
            ds[f"{frequency_group}/sceneCenterGroundRangeSpacing"][()]
        )
        acquired_center_frequency = float(
            ds[f"{frequency_group}/acquiredCenterFrequency"][()]
        )
        processed_center_frequency = float(
            ds[f"{frequency_group}/processedCenterFrequency"][()]
        )
        wavelength = LIGHT_SPEED_M_PER_SEC / processed_center_frequency
        width = ds[f"{frequency_group}/slantRange"].size
        polarizations = [
            polarization.decode("utf-8")
            for polarization in ds[f"{frequency_group}/listOfPolarizations"][:]
        ]

        # Calibration
        calibr_freq_group = f"science/{radar_band}SAR/RSLC/metadata/calibrationInformation/frequency{frequency}"
        ne_backscatter_dataset: Literal["nes0", "noiseEquivalentBackscatter"] = (
            "nes0"
            if "nes0" in ds[calibr_freq_group].keys()
            else "noiseEquivalentBackscatter"
        )
        # in the product specs, nes0 was changed to noiseEquivalentBackscatter in November 2024
        calibration_group = f"{calibr_freq_group}/{ne_backscatter_dataset}"
        ne_backscatter_azimuth_time = ds[f"{calibration_group}/zeroDopplerTime"][
            :
        ].tolist()
        ref_timestamp = parse_date_as_numpy_datetime64(
            ds[f"{calibration_group}/zeroDopplerTime"]
            .attrs["units"]
            .decode("utf-8")
            .split(" ")[2]
        )
        ne_backscatter_slant_range = ds[f"{calibration_group}/slantRange"][:].tolist()
        ne_backscatter = {
            polarization: ds[f"{calibration_group}/{polarization}"][:].tolist()
            for polarization in ds[f"{calibration_group}"].keys()
            if polarization not in ["zeroDopplerTime", "slantRange"]
        }

        return NisarFrequencyMetadata(
            slant_range_first=slant_range_first,
            slant_range_spacing=slant_range_spacing,
            ground_range_spacing=ground_range_spacing,
            acquired_center_frequency=acquired_center_frequency,
            processed_center_frequency=processed_center_frequency,
            wavelength=wavelength,
            width=width,
            polarizations=polarizations,
            ne_backscatter_dataset=ne_backscatter_dataset,
            ne_backscatter_azimuth_time=ne_backscatter_azimuth_time,
            ne_backscatter_slant_range=ne_backscatter_slant_range,
            ne_backscatter=ne_backscatter,
            ref_timestamp=ref_timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ref_timestamp"] = str(self.ref_timestamp)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NisarFrequencyMetadata:
        d = d.copy()
        d["ref_timestamp"] = np.datetime64(d["ref_timestamp"], "ns")
        return NisarFrequencyMetadata(**d)


@dataclass(frozen=True)
class NisarRSLCMetadata(NisarMetadata):
    height: int
    azimuth_time_first: float
    azimuth_time_interval: float
    ref_timestamp: np.datetime64
    frequency_a: NisarFrequencyMetadata
    frequency_b: Optional[NisarFrequencyMetadata]
    state_vectors: list[StateVector]
    gcps_x: list[list[list[float]]]
    gcps_y: list[list[list[float]]]
    gcps_height: list[float]
    gcps_incidence_angle: list[list[list[float]]]
    gcps_slant_range: list[float]
    gcps_azimuth_time: list[float]
    gcps_epsg: int
    lut_beta0: list[list[float]]
    lut_gamma0: list[list[float]]
    lut_sigma0: list[list[float]]
    lut_slant_range: list[float]
    lut_azimuth_time: list[float]

    def __post_init__(self):
        super().__post_init__()
        assert self.ref_timestamp == self.frequency_a.ref_timestamp
        if self.frequency_b is not None:
            assert self.ref_timestamp == self.frequency_b.ref_timestamp

    @staticmethod
    def parse_metadata(ds: h5py.File) -> NisarRSLCMetadata:
        radar_band = next(iter(ds["science"].keys()))[0]

        # Identification
        identification_group = f"science/{radar_band}SAR/identification"
        mission_id = ds[f"{identification_group}/missionId"][()].decode("utf-8")
        orbit_direction = (
            ds[f"{identification_group}/orbitPassDirection"][()].decode("utf-8").lower()
        )

        look_side = (
            ds[f"{identification_group}/lookDirection"][()].decode("utf-8").lower()
        )

        absolute_orbit_number = int(
            ds[f"{identification_group}/absoluteOrbitNumber"][()]
        )
        relative_orbit_number = int(ds[f"{identification_group}/trackNumber"][()])

        frame_number = int(ds[f"{identification_group}/frameNumber"][()])

        image_start = parse_date_as_numpy_datetime64(
            ds[f"{identification_group}/zeroDopplerStartTime"][()].decode("utf-8")
        )
        assert int(ds[f"{identification_group}/boundingPolygon"].attrs["epsg"]) == 4326
        approx_geom = [
            (float(c[0]), float(c[1]))
            for c in from_wkt(
                ds[f"{identification_group}/boundingPolygon"][()].decode("utf-8")
            ).exterior.coords
        ]
        available_frequencies = [
            frequency.decode("utf-8")
            for frequency in ds[
                f"science/{radar_band}SAR/identification/listOfFrequencies"
            ][:]
        ]

        # Swaths
        swaths_group = f"science/{radar_band}SAR/RSLC/swaths"
        azimuth_time_first = float(ds[f"{swaths_group}/zeroDopplerTime"][0])
        ref_timestamp = parse_date_as_numpy_datetime64(
            ds[f"{swaths_group}/zeroDopplerTime"]
            .attrs["units"]
            .decode("utf-8")
            .split(" ")[2]
        )
        assert image_start - ref_timestamp == np.timedelta64(
            round(azimuth_time_first * 1e9), "ns"
        ), "Inconsistent azimuth time first value"
        azimuth_time_interval = float(ds[f"{swaths_group}/zeroDopplerTimeSpacing"][()])
        height = ds[f"{swaths_group}/zeroDopplerTime"].size

        # Orbit (position and velocity with respect to WGS84 G1762 reference frame)
        orbit_group = f"science/{radar_band}SAR/RSLC/metadata/orbit"
        state_vectors: list[StateVector] = []
        timesteps = ds[f"{orbit_group}/time"].shape[0]
        orbit_ref_timestamp = parse_date_as_numpy_datetime64(
            ds[f"{orbit_group}/time"].attrs["units"].decode("utf-8").split(" ")[2]
        )
        for timestep in range(timesteps):
            t = float(ds[f"{orbit_group}/time"][timestep])
            x, y, z = (
                float(ds[f"{orbit_group}/position"][timestep, 0]),
                float(ds[f"{orbit_group}/position"][timestep, 1]),
                float(ds[f"{orbit_group}/position"][timestep, 2]),
            )
            vx, vy, vz = (
                float(ds[f"{orbit_group}/velocity"][timestep, 0]),
                float(ds[f"{orbit_group}/velocity"][timestep, 1]),
                float(ds[f"{orbit_group}/velocity"][timestep, 2]),
            )
            sv = StateVector(time=t, position=(x, y, z), velocity=(vx, vy, vz))
            state_vectors.append(sv)

        # Geolocation
        geolocation_group = f"science/{radar_band}SAR/RSLC/metadata/geolocationGrid"
        gcps_x = ds[f"{geolocation_group}/coordinateX"][:].tolist()
        gcps_y = ds[f"{geolocation_group}/coordinateY"][:].tolist()
        gcps_height = ds[f"{geolocation_group}/heightAboveEllipsoid"][:].tolist()
        gcps_incidence_angle = ds[f"{geolocation_group}/incidenceAngle"][:].tolist()
        gcps_slant_range = ds[f"{geolocation_group}/slantRange"][:].tolist()
        gcps_azimuth_time = ds[f"{geolocation_group}/zeroDopplerTime"][:].tolist()
        gcps_ref_timestamp = parse_date_as_numpy_datetime64(
            ds[f"{geolocation_group}/zeroDopplerTime"]
            .attrs["units"]
            .decode("utf-8")
            .split(" ")[2]
        )
        gcps_epsg = int(ds[f"{geolocation_group}/epsg"][()])

        # Lookup tables (LUT)
        calibration_group = (
            f"science/{radar_band}SAR/RSLC/metadata/calibrationInformation/geometry"
        )
        lut_beta0 = ds[f"{calibration_group}/beta0"][:].tolist()
        lut_gamma0 = ds[f"{calibration_group}/gamma0"][:].tolist()
        lut_sigma0 = ds[f"{calibration_group}/sigma0"][:].tolist()
        lut_slant_range = ds[f"{calibration_group}/slantRange"][:].tolist()
        lut_azimuth_time = ds[f"{calibration_group}/zeroDopplerTime"][:].tolist()
        lut_ref_timestamp = parse_date_as_numpy_datetime64(
            ds[f"{calibration_group}/zeroDopplerTime"]
            .attrs["units"]
            .decode("utf-8")
            .split(" ")[2]
        )

        # Frequency
        assert "frequencyA" in ds[f"science/{radar_band}SAR/RSLC/swaths"].keys()
        frequency_a = NisarFrequencyMetadata.parse_metadata(
            ds=ds, frequency="A", radar_band=radar_band
        )
        frequency_b = (
            NisarFrequencyMetadata.parse_metadata(
                ds=ds, frequency="B", radar_band=radar_band
            )
            if "frequencyB" in ds[f"science/{radar_band}SAR/RSLC/swaths"].keys()
            else None
        )
        if frequency_b is not None:
            assert "B" in available_frequencies
        else:
            assert available_frequencies == ["A"]

        ref_timestamps = [
            ref_timestamp,
            orbit_ref_timestamp,
            gcps_ref_timestamp,
            lut_ref_timestamp,
        ]
        assert all([r == ref_timestamps[0] for r in ref_timestamps[1:]])

        return NisarRSLCMetadata(
            radar_band=radar_band,
            mission_id=mission_id,
            orbit_direction=orbit_direction,
            look_side=look_side,
            absolute_orbit_number=absolute_orbit_number,
            relative_orbit_number=relative_orbit_number,
            frame_number=frame_number,
            approx_geom=approx_geom,
            ref_timestamp=ref_timestamp,
            height=height,
            azimuth_time_first=azimuth_time_first,
            azimuth_time_interval=azimuth_time_interval,
            frequency_a=frequency_a,
            frequency_b=frequency_b,
            state_vectors=state_vectors,
            gcps_x=gcps_x,
            gcps_y=gcps_y,
            gcps_height=gcps_height,
            gcps_incidence_angle=gcps_incidence_angle,
            gcps_slant_range=gcps_slant_range,
            gcps_azimuth_time=gcps_azimuth_time,
            gcps_epsg=gcps_epsg,
            lut_beta0=lut_beta0,
            lut_gamma0=lut_gamma0,
            lut_sigma0=lut_sigma0,
            lut_slant_range=lut_slant_range,
            lut_azimuth_time=lut_azimuth_time,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state_vectors"] = [
            state_vector.to_dict() for state_vector in self.state_vectors
        ]
        d["frequency_a"] = self.frequency_a.to_dict()
        d["frequency_b"] = self.frequency_b.to_dict() if self.frequency_b else None
        d["ref_timestamp"] = str(self.ref_timestamp)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NisarRSLCMetadata:
        d = d.copy()
        d["state_vectors"] = [StateVector.from_dict(s) for s in d["state_vectors"]]
        d["frequency_a"] = NisarFrequencyMetadata.from_dict(d["frequency_a"])
        d["frequency_b"] = (
            NisarFrequencyMetadata.from_dict(d["frequency_b"])
            if d["frequency_b"]
            else None
        )
        d["ref_timestamp"] = np.datetime64(d["ref_timestamp"], "ns")
        d["approx_geom"] = [tuple(pt) for pt in d["approx_geom"]]
        return NisarRSLCMetadata(**d)


def main(h5_file_path: str) -> NisarRSLCMetadata:
    """
    Example usage
    """
    with h5py.File(h5_file_path, "r") as ds:
        metadata = NisarRSLCMetadata.parse_metadata(ds)
    return metadata


if __name__ == "__main__":
    import fire

    fire.Fire(main)
