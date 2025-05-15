from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pyproj
import rasterio
from numpy.typing import ArrayLike, NDArray
from typing_extensions import override

from eos.sar import coordinates
from eos.sar.const import LIGHT_SPEED_M_PER_SEC
from eos.sar.model import Arrayf64, CoordArrayLike, SensorModel
from eos.sar.model_helper import GenericSensorModelHelper
from eos.sar.orbit import Orbit, StateVector
from eos.sar.projection_correction import Corrector
from eos.sar.roi import Roi


@dataclass(frozen=True)
class CosmoSkyMedMetadata:
    mission_id: Literal["CSK", "CSG"]
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
    doppler_coefficients: list[float]

    @property
    def range_frequency(self) -> float:
        return LIGHT_SPEED_M_PER_SEC / (2.0 * self.range_pixel_spacing)

    @property
    def azimuth_time_interval(self) -> float:
        return 1.0 / self.azimuth_frequency

    @property
    def range_time_interval(self):
        return 1.0 / self.range_sampling_rate

    def get_gdal_image_path(self, hdf5_path) -> str:
        ipt = "IMG" if self.mission_id == "CSG" else "SBI"
        return f'HDF5:"{hdf5_path}"://S01/{ipt}'

    def deramping_phases(self, roi: Roi) -> NDArray[np.float32]:
        # from SNAP microwave-toolbox
        #   sar-io/src/main/java/eu/esa/sar/io/cosmo/CosmoSkymedNetCDFReader.java
        #   and sar-op-utilities/src/main/java/eu/esa/sar/utilities/gpf/DemodulateOp.java
        # except:
        #   I considered that "offsets" were 0 (TODO)
        #   mid_range_time is the middle of the Roi and not the middle of the product, it seems to improve the deramping
        # TODO: there is a small residual shift that depends on the range location, can we fix it?

        mid_range_time = (roi.col + roi.w // 2) * self.range_time_interval

        doppler_centroid = sum(
            coeff * mid_range_time**p
            for p, coeff in enumerate(self.doppler_coefficients)
        )

        h, w = roi.get_shape()
        rows = np.arange(roi.row, roi.row + h, dtype=np.float32)
        ta = rows * self.azimuth_time_interval
        phi = -np.pi * 2 * doppler_centroid * ta
        phi = np.broadcast_to(phi.reshape((h, 1)), (h, w))

        assert phi.shape == (h, w)
        assert phi.dtype == np.float32
        return phi


def string_to_timestamp(s: str) -> float:
    """Convert a string representing a date and time to a float number."""
    # remove nanoseconds
    s = s.replace(".000000000", ".000000")
    return (
        datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def parse_cosmoskymed_metadata(hdf5_path: str) -> CosmoSkyMedMetadata:
    with rasterio.open(hdf5_path, driver="HDF5") as f:
        d = f.tags()

    csg = d["Mission_ID"] == "CSG"
    ipt = "IMG" if csg else "SBI"

    with rasterio.open(f'HDF5:"{hdf5_path}"://S01/{ipt}') as f:
        height, width = f.shape

    mission_id = d["Mission_ID"]
    assert mission_id in ("CSK", "CSG")

    reference_time = string_to_timestamp(d["Reference_UTC"])

    image_start = reference_time + float(
        d[f"S01_{ipt}_Zero_Doppler_Azimuth_First_Time"]
    )
    azimuth_frequency = 1 / float(d[f"S01_{ipt}_Line_Time_Interval"])
    slant_range_time = float(d[f"S01_{ipt}_Zero_Doppler_Range_First_Time"])
    range_pixel_spacing = float(d[f"S01_{ipt}_Column_Spacing"])
    azimuth_pixel_spacing = float(d[f"S01_{ipt}_Line_Spacing"])
    wavelength = float(d["Radar_Wavelength"])
    range_sampling_rate = float(d["S01_Sampling_Rate"])

    if "RANGE_PIXEL_SPACING" in d:
        range_pixel_spacing = float(d["RANGE_PIXEL_SPACING"])
    if "AZIMUTH_PIXEL_SPACING" in d:
        azimuth_pixel_spacing = float(d["AZIMUTH_PIXEL_SPACING"])

    light_speed = float(d["Light_Speed"])
    assert light_speed == LIGHT_SPEED_M_PER_SEC
    orbit_direction = d["Orbit_Direction"].lower()
    assert orbit_direction in ("ascending", "descending")
    look_side = d["Look_Side"].lower()
    assert look_side in ("left", "right")

    # state vectors (sv)
    state_vectors: list[StateVector] = []
    times = [float(x) for x in d["State_Vectors_Times"].split()]
    positions = np.asarray(
        [float(x) for x in d["ECEF_Satellite_Position"].split()]
    ).reshape(-1, 3)
    velocities = np.asarray(
        [float(x) for x in d["ECEF_Satellite_Velocity"].split()]
    ).reshape(-1, 3)
    for t, p, v in zip(times, positions, velocities):
        p = tuple(p)
        v = tuple(v)
        assert len(p) == 3
        assert len(v) == 3
        state_vectors.append(
            StateVector(time=reference_time + t, position=p, velocity=v)
        )

    # longitude, latitude bounding box
    corners = (
        [float(x) for x in d[f"S01_{ipt}_Top_Left_Geodetic_Coordinates"].split()],
        [float(x) for x in d[f"S01_{ipt}_Top_Right_Geodetic_Coordinates"].split()],
        [float(x) for x in d[f"S01_{ipt}_Bottom_Right_Geodetic_Coordinates"].split()],
        [float(x) for x in d[f"S01_{ipt}_Bottom_Left_Geodetic_Coordinates"].split()],
    )
    approx_geom = [(x[1], x[0]) for x in corners]

    doppler_coefficients = [
        float(x) for x in d["Centroid_vs_Range_Time_Polynomial"].split()
    ]

    return CosmoSkyMedMetadata(
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
        doppler_coefficients=doppler_coefficients,
    )


@dataclass(frozen=True)
class CosmoSkyMedModel(SensorModel):
    generic_model: GenericSensorModelHelper
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float

    @staticmethod
    def from_metadata(
        meta: CosmoSkyMedMetadata, orbit_degree: int, corrector: Corrector = Corrector()
    ) -> CosmoSkyMedModel:
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

        return CosmoSkyMedModel(
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
