import json
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from typing_extensions import assert_never

from eos.sar.const import LIGHT_SPEED_M_PER_SEC as C
from eos.sar.orbit import StateVector

"""
We take extra measures to preserve the max. precision of timestamps, i.e. nanoseconds. 
Indeed, an example azimuth time interval (collect/image/image_geometry/delta_line_time) 
of file CAPELLA_C02_SS_SLC_HH_20210204153042_20210204153058_extended.json (sliding spotlight) 
is 0.0001224950981868954 seconds. This means that 1 microsecond error (worst case for truncation)
can yield 1e-6  / delta_line_time = 0.008163591970629405 pixels error in azimuth. 
This can be worse for Spotlight (ultra) products.
"""


def time_since_ref(mydate: np.datetime64, ref: np.datetime64) -> float:
    """
    Get time since ref in seconds.
    """

    time_delta = mydate - ref

    diff_in_seconds = time_delta / np.timedelta64(1, "s")  # This is np.float64

    # convert to float
    return float(diff_in_seconds)


def parse_date_as_numpy_datetime64(date_str: str) -> np.datetime64:
    """
    Parse date string with nanosecond precision into numpy datetime64:
        ex date_str: '2021-02-04T15:30:42.421646560Z'
        As per the documentation https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
        the datetime is represented internally as an int64 w.r.t. to 1970-01-01T00:00.
        For ns precision, the time delta representable on a signed int64 is the interval [-2**63 + 1, 2**63]
        so converting the largest value to years: 2**63 * 1e-9 /3600/24/365.25 = 292.27 years
        This is in accordance with numpy doc. For a datetime64 with ns precision, we can represent dates in the interval [ 1678 AD, 2262 AD].
        So using ns precision shouldn't be an issue until 2262 ;).
    """
    return np.datetime64(date_str.removesuffix("Z"), "ns")


def _get_3D_tuple(vec_3D: list[float]) -> tuple[float, float, float]:
    assert len(vec_3D) == 3, f"Expected 3 coordinates, got {len(vec_3D)}"
    return (vec_3D[0], vec_3D[1], vec_3D[2])


@dataclass(frozen=True)
class CapellaMetadata:
    """
    Filled attributes can be interpreted from Capella_Space_SAR_Products_Format_Specification_v1.8.pdf
    In the documentation below, items represented as *item* refer to the name in the Capella metdata.
    """

    start_timestamp: np.datetime64
    """
    *start_timestamp*: Timestamp for the start of the collection. ex. "2021-02-04T15:30:42.421646560Z", converted to np.datetime64.
    """
    stop_timestamp: np.datetime64
    """
    *stop_timestamp*: Timestamp for the end of the collection. ex. "2021-02-04T15:30:58.029504459Z", converted to np.datetime64.
    """
    ref_timestamp: np.datetime64
    """
    reference timestamp against which floating time deltas will be computed ex. "2021-02-04T00:00:00.000000000", converted to np.datetime64.
    """
    height: int
    """
    *collect/image/rows*: The number of rows in the image.
    """
    width: int
    """ 
    *collect/image/columns*: The number of columns in the image.
    """
    center_pixel_incidence_angle: float
    """
    *collect/image/center_pixel/incidence_angle*: The incidence angle in degrees.
    """
    center_pixel_target_position: tuple[float, float, float]
    """
    *collect/image/center_pixel/target_position*: The ECEF coordinates of the center pixel
    """
    pixel_spacing_column: float
    """
    *collect/image/pixel_spacing_column*: The meters between samples in the
                                          column direction at the center of the
                                          image.
    """
    range_resolution: float
    """
    *collect/image/range_resolution*: The resolution in the slant range direction.
    """

    ground_range_resolution: float
    """
    *collect/image/ground_range_resolution*: The resolution in the ground range direction.
    """
    range_looks: float
    """
    *collect/image/range_looks*: The number of looks in the range direction.
    """
    azimuth_pixel_size: float
    """
    *collect/image/pixel_spacing_row*: The meters between samples in the
                                    row direction at the center of the
                                    image
    """
    azimuth_resolution: float
    """
    *collect/image/azimuth_resolution*: The number of looks in the azimuth direction.
    """
    azimuth_looks: float
    """
    *collect/image/azimuth_looks*: The number of looks in the azimuth direction.
    """

    center_frequency: float
    """
    *collect/radar/center_frequency*: The center frequency of the radar (Hz)
    """

    transmit_polarization: Literal["H", "V"]
    """
    *collect/radar/transmit_polarization*: The transmit polarization of the radar. Ex. H
    """
    receive_polarization: Literal["H", "V"]
    """
    *collect/radar/receive_polarization*: The receive polarization of the radar. Ex. H
    """

    look_direction: Literal["right", "left"]
    """
    *collect/radar/pointing*
    """

    orbit_direction: Literal["ascending", "descending", "null"]
    """ 
    *collect/state/direction*: null when not applicable
    """

    state_vectors: list[StateVector]
    """
    Deduced from collect/state/state_vectors.
    """

    @property
    def wavelength(self) -> float:
        return C / self.center_frequency

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class CapellaSLCMetadata(CapellaMetadata):
    starting_range: float
    """
    *collect/image/image_geometry/range_to_first_sample*: The slant range distance to the first sample in meters
    """
    range_pixel_size: float
    """
    *collect/image/image_geometry/delta_range_sample*: The slant range delta distance between each sample in
meters
    """
    delta_line_time: float
    """
    *collect/image/image_geometry/delta_line_time*: The time difference between successive lines in seconds
    """

    first_line_time: np.datetime64
    """
    *collect/image/image_geometry/first_line_time*: The timestamp of the first line, converted to np.datetime64.
    """

    @property
    def first_col_time(self) -> float:
        """
        Two way range time of first col
        """
        return 2 * self.starting_range / C

    @property
    def first_line_time_since_ref(self) -> float:
        """
        Two way range time of first col
        """
        return time_since_ref(self.first_line_time, self.ref_timestamp)


@dataclass(frozen=True)
class CapellaGECMetadata(CapellaMetadata):
    alt_inflated_wgs84: float
    """
    Deduced from collect/image/terrain_models/reprojection/name, for ex. ExplicitInflatedWGS84[243.3561248779297] would give alt_inflated_wgs84=243.3561248779297
    """


def parse_metadata(json_content: str) -> Union[CapellaSLCMetadata, CapellaGECMetadata]:
    data = json.loads(json_content)

    product_type = data["product_type"]

    assert product_type in ["SLC", "GEC"], (
        "Unsupported Product Type: Only supported product types are SLC and GEC"
    )

    collect = data["collect"]
    del data

    start_timestamp = parse_date_as_numpy_datetime64(collect["start_timestamp"])

    # midnight of start_timestamp
    # floor on day frequency then convert back to ns precision
    ref_timestamp = start_timestamp.astype("datetime64[D]").astype("datetime64[ns]")

    stop_timestamp = parse_date_as_numpy_datetime64(collect["stop_timestamp"])

    image = collect["image"]
    radar = collect["radar"]
    state = collect["state"]

    del collect

    height = image["rows"]
    width = image["columns"]

    center_pixel_incidence_angle = image["center_pixel"]["incidence_angle"]

    center_pixel_target_position = _get_3D_tuple(
        image["center_pixel"]["target_position"]
    )

    pixel_spacing_column = image["pixel_spacing_column"]
    range_resolution = image["range_resolution"]
    ground_range_resolution = image["ground_range_resolution"]
    range_looks = image["range_looks"]

    azimuth_pixel_size = image["pixel_spacing_row"]
    azimuth_resolution = image["azimuth_resolution"]
    azimuth_looks = image["azimuth_looks"]

    center_frequency = radar["center_frequency"]

    transmit_polarization = radar["transmit_polarization"]
    assert transmit_polarization in ["H", "V"], "Unrecognized transmit polarization"
    receive_polarization = radar["receive_polarization"]
    assert receive_polarization in ["H", "V"], "Unrecognized receive polarization"

    look_direction = radar["pointing"]
    assert look_direction in ["right", "left"], "Unrecognized look direction string"

    orbit_direction = state["direction"]
    assert orbit_direction in ["ascending", "descending", "null"], (
        "Unrecognized orbit direction string"
    )

    # State vectors
    state_vectors = []
    for p in state["state_vectors"]:
        sv_time = parse_date_as_numpy_datetime64(p["time"])
        sv_time_since_ref = time_since_ref(sv_time, ref_timestamp)
        sv_pos = _get_3D_tuple(p["position"])
        sv_velocity = _get_3D_tuple(p["velocity"])
        state_vectors.append(StateVector(sv_time_since_ref, sv_pos, sv_velocity))

    if product_type == "SLC":
        image_geometry = image["image_geometry"]

        im_geom_type = image_geometry["type"]
        assert im_geom_type == "slant_plane", (
            f"Unsupported image geometry type: {im_geom_type}"
        )

        # The next attributes are only valid when the image geometry type is slant plane
        starting_range = image_geometry["range_to_first_sample"]

        range_pixel_size = image_geometry["delta_range_sample"]

        delta_line_time = image_geometry["delta_line_time"]

        first_line_time = parse_date_as_numpy_datetime64(
            image_geometry["first_line_time"]
        )
        return CapellaSLCMetadata(
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            ref_timestamp=ref_timestamp,
            height=height,
            width=width,
            center_pixel_incidence_angle=center_pixel_incidence_angle,
            center_pixel_target_position=center_pixel_target_position,
            pixel_spacing_column=pixel_spacing_column,
            range_resolution=range_resolution,
            ground_range_resolution=ground_range_resolution,
            range_looks=range_looks,
            azimuth_pixel_size=azimuth_pixel_size,
            azimuth_resolution=azimuth_resolution,
            azimuth_looks=azimuth_looks,
            center_frequency=center_frequency,
            transmit_polarization=transmit_polarization,
            receive_polarization=receive_polarization,
            look_direction=look_direction,
            orbit_direction=orbit_direction,
            state_vectors=state_vectors,
            # SLC specific:
            starting_range=starting_range,
            range_pixel_size=range_pixel_size,
            delta_line_time=delta_line_time,
            first_line_time=first_line_time,
        )
    elif product_type == "GEC":
        reproj_name = image["terrain_models"]["reprojection"]["name"]
        # expect something like "ExplicitInflatedWGS84[-13.663856506347656]"
        assert reproj_name.startswith("ExplicitInflatedWGS84["), (
            f"unsupported terrain model reprojection name `{reproj_name}`"
        )
        alt_inflated_wgs84 = float(reproj_name.split("[")[1][:-1])

        return CapellaGECMetadata(
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            ref_timestamp=ref_timestamp,
            height=height,
            width=width,
            center_pixel_incidence_angle=center_pixel_incidence_angle,
            center_pixel_target_position=center_pixel_target_position,
            pixel_spacing_column=pixel_spacing_column,
            range_resolution=range_resolution,
            ground_range_resolution=ground_range_resolution,
            range_looks=range_looks,
            azimuth_pixel_size=azimuth_pixel_size,
            azimuth_resolution=azimuth_resolution,
            azimuth_looks=azimuth_looks,
            center_frequency=center_frequency,
            transmit_polarization=transmit_polarization,
            receive_polarization=receive_polarization,
            look_direction=look_direction,
            orbit_direction=orbit_direction,
            state_vectors=state_vectors,
            # GEC specific:
            alt_inflated_wgs84=alt_inflated_wgs84,
        )
    else:
        # unreachable code, because of assert at start of function
        assert_never(product_type)
