import datetime
import json
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
from typing_extensions import assert_never

from eos.sar.const import LIGHT_SPEED_M_PER_SEC as C
from eos.sar.orbit import StateVector


def UTC_time_since_midnight(mydate):
    """
    Function to convert UTC time to time in seconds since midnight.

    Parameters
    ----------
    mydate: pd._libs.tslibs.timestamps.Timestamp
        Date in UTC format: Timestamp('YYYY-MM-DD hh:mm:ss').

    Returns
    -------
    time_since_midnight: float
        Date converted in time in seconds since midnight.
    """

    date_midnight = pd.to_datetime(
        "%4d-%02d-%02dT00:00:00.000000000Z" % (mydate.year, mydate.month, mydate.day),
        format="%Y-%m-%dT%H:%M:%S.%fZ",
    )
    time_since_midnight = float(
        (mydate - date_midnight).seconds + (mydate - date_midnight).microseconds / 1e6
    )
    return time_since_midnight


@dataclass(frozen=True)
class CapellaMetadata:
    """
    Filled attributes can be interpreted from Capella_Space_SAR_Products_Format_Specification_v1.8.pdf
    In the documentation below, items represented as *item* refer to the name in the Capella metdata.
    """

    processing_version: str
    """
    *software_version*: The version of the processor that created this product
    """
    start_timestamp: pd.Timestamp
    """
    *start_timestamp*: Timestamp for the start of the collection. ex. "2021-02-04T15:30:42.421646560Z"
    """
    stop_timestamp: pd.Timestamp
    """
    *stop_timestamp*: Timestamp for the end of the collection. ex. "2021-02-04T15:30:58.029504459Z"
    """
    height: int
    """
    *collect/image/rows*: The number of rows in the image.
    """
    width: int
    """
    *collect/image/columns*: The number of columns in the image.
    """
    incidence_angle: float
    """
    *collect/image/center_pixel/incidence_angle*: The incidence angle in degrees.
    """
    look_angle: float
    """
    *collect/image/center_pixel/look_angle*: The look angle in degrees.
    """
    squint_angle: float
    """
    *collect/image/center_pixel/squint_angle*: The squint angle in degrees.
    """
    center_pixel_target_position: tuple[float, float, float]
    """
    *collect/image/center_pixel/target_position*: The ECEF coordinates of the center pixel
    """
    center_pixel_time: pd.Timestamp
    """
    *collect/image/center_pixel/center_time*: The timestamp of when the antenna center acquired the pixel
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
    radiometry: Literal["none", "limited", "partial", "full"]
    """
    *collect/image/calibration*:
                                    • none (no calibration applied)
                                    • limited (calibration applied, with no telemetry)
                                    • partial (calibration applied, with partial telemetry)
                                    • full (calibration applied, with all telemetry)
    """
    calibration_id: str
    """
    *collect/image/calibration_id*: Version of the calibration applied to the data products. ex. "calibration_bundle/2bfbc3b1-5d72-4ada-b36b-4479e0eb73fa"
    """
    range_sampling_frequency: float
    """
    *collect/radar/sampling_frequency*: The sampling frequency of the ADC in Hz. 
    """
    center_frequency: float
    """
    *collect/radar/center_frequency*: The center frequency of the radar (Hz)
    """
    wavelength: float
    """
    C / center_frequency
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

    antenna_side: Literal[-1, 1]
    """
    -1 if look_direction=="right" else 1
    """

    orbit_direction: Literal["ascending", "descending", "null"]
    """ 
    *collect/state/direction*: null when not applicable
    """

    platform: str
    """
    *collect/platform*: The platform of the acquisition. Ex. capella-2
    """

    prf: float
    """
    Deduced from collect/radar/time_varying_parameters, by averaging values corresponding to prf key. The PRF in Hz
    """
    pulse_length: float
    """
    Deduced from collect/radar/time_varying_parameters, by averaging values corresponding to pulse_duration key. The time duration of the transmitted pulse in seconds
    """

    chirp_slope: float
    """
    Deduced from collect/radar/time_varying_parameters, by averaging values corresponding to pulse_bandwidth key and dividing by the pulse_length attribute. Hz/sec.
    """

    state_vectors: list[StateVector]
    """
    Deduced from collect/state/state_vectors.
    """

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
    delta_line_utc: float
    """
    *collect/image/image_geometry/delta_line_time*: The time difference between successive lines in seconds
    """

    first_col_time: float
    """
    Two way range time of first col: 2 * self.starting_range / C
    """

    first_line_time: pd.Timestamp
    """
    *collect/image/image_geometry/first_line_time*: The timestamp of the first line
    """
    first_line_utc: float
    """
    self.first_line_time - midnight
    """

    date: str
    """
    %Y%m%d of first line
    """
    date_spaced: str
    """
    %Y-%m-%d of first line
    """

    last_line_time: pd.Timestamp
    """
    Time of last line (non inclusive)
    """
    last_line_utc: float
    """
    self.last_line_time - midnight
    """


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

    # General information
    processing_version = data["software_version"]

    start_timestamp = pd.to_datetime(
        data["collect"]["start_timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ"
    )
    stop_timestamp = pd.to_datetime(
        data["collect"]["stop_timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ"
    )

    height = int(data["collect"]["image"]["rows"])
    width = int(data["collect"]["image"]["columns"])

    incidence_angle = float(data["collect"]["image"]["center_pixel"]["incidence_angle"])
    look_angle = float(data["collect"]["image"]["center_pixel"]["look_angle"])
    squint_angle = float(data["collect"]["image"]["center_pixel"]["squint_angle"])
    center_pixel_target_position = data["collect"]["image"]["center_pixel"][
        "target_position"
    ]
    center_pixel_time = pd.to_datetime(
        data["collect"]["image"]["center_pixel"]["center_time"],
        format="%Y-%m-%dT%H:%M:%S.%fZ",
    )

    pixel_spacing_column = float(data["collect"]["image"]["pixel_spacing_column"])
    range_resolution = float(data["collect"]["image"]["range_resolution"])
    ground_range_resolution = float(data["collect"]["image"]["ground_range_resolution"])
    range_looks = float(data["collect"]["image"]["range_looks"])

    azimuth_pixel_size = float(data["collect"]["image"]["pixel_spacing_row"])
    azimuth_resolution = float(data["collect"]["image"]["azimuth_resolution"])
    azimuth_looks = float(data["collect"]["image"]["azimuth_looks"])

    radiometry = data["collect"]["image"]["calibration"]
    calibration_id = data["collect"]["image"]["calibration_id"]

    range_sampling_frequency = float(data["collect"]["radar"]["sampling_frequency"])
    center_frequency = float(data["collect"]["radar"]["center_frequency"])
    wavelength = C / center_frequency

    transmit_polarization = data["collect"]["radar"]["transmit_polarization"]
    receive_polarization = data["collect"]["radar"]["receive_polarization"]

    look_direction = data["collect"]["radar"]["pointing"]

    if look_direction == "right":
        antenna_side: Literal[-1, 1] = -1
    else:
        antenna_side = 1

    orbit_direction = data["collect"]["state"]["direction"]

    platform = data["collect"]["platform"]

    # PRF
    radar_info_prf = []
    radar_info_pulse_duration = []
    radar_info_pulse_bw = []
    for p in data["collect"]["radar"]["time_varying_parameters"]:
        for mytime in p["start_timestamps"]:
            radar_info_prf.append(float(p["prf"]))
            radar_info_pulse_duration.append(float(p["pulse_duration"]))
            radar_info_pulse_bw.append(float(p["pulse_bandwidth"]))
    radar_info_prf = np.array(radar_info_prf)
    radar_info_pulse_duration = np.array(radar_info_pulse_duration)
    radar_info_pulse_bw = np.array(radar_info_pulse_bw)

    prf = float(np.mean(radar_info_prf))
    pulse_length = float(np.mean(radar_info_pulse_duration))
    chirp_slope = float(np.mean(radar_info_pulse_bw) / pulse_length)

    # State vectors
    state_vectors_tim = []
    state_vectors_pos = []
    state_vectors_vel = []
    for p in data["collect"]["state"]["state_vectors"]:
        state_vectors_tim.append(
            UTC_time_since_midnight(
                pd.to_datetime(p["time"], format="%Y-%m-%dT%H:%M:%S.%fZ")
            )
        )
        state_vectors_pos.append(np.array(p["position"], dtype=float))
        state_vectors_vel.append(np.array(p["velocity"], dtype=float))
    state_vectors_tim = np.array(state_vectors_tim)
    state_vectors_pos = np.array(state_vectors_pos)
    state_vectors_vel = np.array(state_vectors_vel)

    n = len(state_vectors_tim)
    state_vectors = []
    for i in range(n):
        state_vectors.append(
            StateVector(
                state_vectors_tim[i], state_vectors_pos[i], state_vectors_vel[i]
            )
        )

    # Time information
    if product_type == "SLC":
        starting_range = float(
            data["collect"]["image"]["image_geometry"]["range_to_first_sample"]
        )
        range_pixel_size = float(
            data["collect"]["image"]["image_geometry"]["delta_range_sample"]
        )

        delta_line_utc = float(
            data["collect"]["image"]["image_geometry"]["delta_line_time"]
        )

        first_col_time = 2 * starting_range / C

        first_line_time = pd.to_datetime(
            data["collect"]["image"]["image_geometry"]["first_line_time"],
            format="%Y-%m-%dT%H:%M:%S.%fZ",
        )
        first_line_utc = UTC_time_since_midnight(first_line_time)

        date = first_line_time.strftime("%Y%m%d")
        date_spaced = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        # non inclusive
        last_line_time = first_line_time + datetime.timedelta(
            seconds=height * delta_line_utc
        )
        last_line_utc = UTC_time_since_midnight(last_line_time)

        return CapellaSLCMetadata(
            processing_version,
            start_timestamp,
            stop_timestamp,
            height,
            width,
            incidence_angle,
            look_angle,
            squint_angle,
            center_pixel_target_position,
            center_pixel_time,
            pixel_spacing_column,
            range_resolution,
            ground_range_resolution,
            range_looks,
            azimuth_pixel_size,
            azimuth_resolution,
            azimuth_looks,
            radiometry,
            calibration_id,
            range_sampling_frequency,
            center_frequency,
            wavelength,
            transmit_polarization,
            receive_polarization,
            look_direction,
            antenna_side,
            orbit_direction,
            platform,
            prf,
            pulse_length,
            chirp_slope,
            state_vectors,
            starting_range,
            range_pixel_size,
            delta_line_utc,
            first_col_time,
            first_line_time,
            first_line_utc,
            date,
            date_spaced,
            last_line_time,
            last_line_utc,
        )
    elif product_type == "GEC":
        alt_inflated_wgs84 = float(
            data["collect"]["image"]["terrain_models"]["reprojection"]["name"].split(
                "["
            )[1][:-1]
        )
        return CapellaGECMetadata(
            processing_version,
            start_timestamp,
            stop_timestamp,
            height,
            width,
            incidence_angle,
            look_angle,
            squint_angle,
            center_pixel_target_position,
            center_pixel_time,
            pixel_spacing_column,
            range_resolution,
            ground_range_resolution,
            range_looks,
            azimuth_pixel_size,
            azimuth_resolution,
            azimuth_looks,
            radiometry,
            calibration_id,
            range_sampling_frequency,
            center_frequency,
            wavelength,
            transmit_polarization,
            receive_polarization,
            look_direction,
            antenna_side,
            orbit_direction,
            platform,
            prf,
            pulse_length,
            chirp_slope,
            state_vectors,
            alt_inflated_wgs84,
        )
    else:
        assert_never(product_type)
