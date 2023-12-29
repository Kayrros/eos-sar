from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pyproj
import rasterio
from numpy.typing import ArrayLike, NDArray

from eos.sar import coordinates, range_doppler, utils
from eos.sar.const import LIGHT_SPEED_M_PER_SEC
from eos.sar.model import Arrayf32, SensorModel
from eos.sar.orbit import Orbit, StateVector
from eos.sar.projection_correction import Corrector, GeoImagePoints
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
    # for SensorModel:
    w: int
    h: int
    orbit: Orbit
    wavelength: float

    # for CosmoSkyMedModel:
    coordinate: coordinates.SLCCoordinate
    azt_init: float
    projection_tolerance: float
    localization_tolerance: float
    max_iterations: int
    coord_corrector: Corrector
    approx_centroid_lon: float
    approx_centroid_lat: float

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

        return CosmoSkyMedModel(
            coordinate=coordinate,
            azt_init=meta.image_start,
            w=meta.width,
            h=meta.height,
            orbit=orbit,
            wavelength=meta.wavelength,
            projection_tolerance=projection_tolerance,
            localization_tolerance=tolerance,
            max_iterations=20,
            coord_corrector=corrector,
            approx_centroid_lon=approx_centroid_lon,
            approx_centroid_lat=approx_centroid_lat,
        )

    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        return self.coordinate.to_azt_rng(row, col)

    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        return self.coordinate.to_row_col(azt, rng)

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
        """Projects a 3D point into the image coordinates.

        Parameters
        ----------
        x, y : ndarray or scalar
            Coordinates in the crs defined by crs parameter.
        alt: ndarray or scalar
            Altitude defined by vert_crs if provided or EARTH_WGS84 ellipsoid.
        crs : string, optional
            CRS in which the point is given
                    Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
        vert_crs: string, optional
            Vertical crs
        azt_init: ndarray or scalar, optional
            Initial azimuth time guess of the points. If not given, the first
            row time will be used. The default is None.
        as_azt_rng: bool, optional
            Returns azimuth/range instead of rows/cols. The incidence angle is unchanged.
            Defaults to False.

        Returns
        -------
        rows : ndarray or scalar
            Row coordinate in image referenced to the first line. (or azimuth if as_azt_rng=True)
        cols : ndarray or scalar
            Column coordinate in image referenced to the first column. (or range if as_azt_rng=True)
        i : ndarray or scalar
            Incidence angle.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        alt = np.atleast_1d(alt)

        if vert_crs is None:
            src_crs = crs
        else:
            src_crs = pyproj.crs.CompoundCRS(
                name="ukn_reference", components=[crs, vert_crs]
            )

        transformer = pyproj.Transformer.from_crs(src_crs, "epsg:4978", always_xy=True)

        # convert to geocentric cartesian
        gx, gy, gz = transformer.transform(x, y, alt)

        if azt_init is not None:
            err_msg = "Init azimuth time should be scalar or have the\
                 same length of the points"
            azt_init = utils.check_input_len(azt_init, len(x), err_msg)
        else:
            azt_init = self.azt_init * np.ones_like(x)

        azt, rng, i = range_doppler.iterative_projection(
            self.orbit,
            gx,
            gy,
            gz,
            azt_init=azt_init,
            max_iterations=self.max_iterations,
            tol=self.projection_tolerance,
        )

        if not self.coord_corrector.empty():
            # create a geo_im_pt
            geo_im_pt = GeoImagePoints(gx, gy, gz, azt, rng)

            # apply corrections
            geo_im_pt = self.coord_corrector.estimate_and_apply(geo_im_pt)

            azt, rng = geo_im_pt.get_azt_rng(squeeze=True)

        if as_azt_rng:
            return azt, rng, i

        # convert to row and col
        row, col = self.to_row_col(azt, rng)

        return row, col, i

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
        """Localize a point in the image at a certain altitude.

        Parameters
        ----------
        row : ndarray or scalar
            row coordinate in image referenced to the first line.
        col : ndarray or scalar
            column coordinate in image referenced to the first column.
        alt : ndarray or scalar
            Altitude above the EARTH_WGS84 ellipsoid.
        crs : string, optional
            CRS in which the point is returned
                    Defaults to 'epsg:4326' (i.e. WGS 84 - 'lonlat').
        vert_crs: string, optional
            Vertical crs in which the point is returned
        x_init: ndarray or scalar, optional
            Initial guess of the x component. The default is None.
        y_init: ndarray or scalar, optional
            Initial guess of the y component. The default is None.
        z_init: ndarray or scalar, optional
            Initial guess of the z component. The default is None.

        Returns
        -------
        x, y, z : ndarray or scalar
            Coordinates of the point in the crs

        Notes
        -----
        If no initial guess for the 3D point is given, the initial point for
        the iterative localization is taken at the centroid of the approx
        geometry of the model, with altitudes given by the alt array.
        """
        # make sure we work with numpy arrays
        row = np.atleast_1d(row)
        col = np.atleast_1d(col)
        alt = np.atleast_1d(alt)

        # image coordinates to range and az time
        azt, rng = self.to_azt_rng(row, col)

        if vert_crs is None:
            dst_crs = crs
        else:
            dst_crs = pyproj.crs.CompoundCRS(
                name="ukn_reference", components=[crs, vert_crs]
            )

        if (x_init is not None) and (y_init is not None) and (z_init is not None):
            to_gxyz = pyproj.Transformer.from_crs(dst_crs, "epsg:4978", always_xy=True)
            out_len = len(alt)
            err_msg = "{} length should be the same as row/col/alt len"
            x_init = utils.check_input_len(x_init, out_len, err_msg.format("x_init"))
            y_init = utils.check_input_len(y_init, out_len, err_msg.format("y_init"))
            z_init = utils.check_input_len(z_init, out_len, err_msg.format("z_init"))
        else:
            # initial geocentric point xyz definition
            # from lon, lat, alt to x, y, z
            to_gxyz = pyproj.Transformer.from_crs(
                "epsg:4326", "epsg:4978", always_xy=True
            )

            x_init = self.approx_centroid_lon * np.ones_like(alt)
            y_init = self.approx_centroid_lat * np.ones_like(alt)
            z_init = alt

        gx_init, gy_init, gz_init = to_gxyz.transform(x_init, y_init, z_init)

        # First localization, no correction is enabled
        # localize each point
        gx, gy, gz = range_doppler.iterative_localization(
            self.orbit,
            azt,
            rng,
            alt,
            (gx_init, gy_init, gz_init),
            max_iterations=self.max_iterations,
            tol=self.localization_tolerance,
        )

        if not self.coord_corrector.empty():
            # create a geo_im_pt
            geo_im_pt = GeoImagePoints(gx, gy, gz, azt, rng)

            # apply corrections
            geo_im_pt = self.coord_corrector.estimate_and_apply(geo_im_pt, inverse=True)

            azt, rng = geo_im_pt.get_azt_rng()

            # Perform localization again with corrected coords
            # Should converge quickly (probably one iteration)
            gx, gy, gz = range_doppler.iterative_localization(
                self.orbit,
                azt,
                rng,
                alt,
                (gx, gy, gz),
                max_iterations=self.max_iterations,
                tol=self.localization_tolerance,
            )

        todst = pyproj.Transformer.from_crs("epsg:4978", dst_crs, always_xy=True)
        x, y, z = todst.transform(gx, gy, gz)

        return x, y, z
