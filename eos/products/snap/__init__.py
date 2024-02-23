from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import dateutil.parser
import numpy as np
import pyproj
import rasterio
import xmltodict
from numpy.typing import ArrayLike, NDArray

from eos.sar import coordinates, io, range_doppler, utils
from eos.sar.const import LIGHT_SPEED_M_PER_SEC
from eos.sar.io import ImageReader, Window
from eos.sar.model import Arrayf32, SensorModel
from eos.sar.orbit import Orbit, StateVector
from eos.sar.projection_correction import Corrector, GeoImagePoints


@dataclass(frozen=True)
class SnapMetadata:
    product_id: str
    state_vectors: list[StateVector]
    orbit_direction: Literal["ascending", "descending"]
    look_side: Literal["left", "right"]
    width: int
    height: int
    approx_geom: list[tuple[float, float]]
    image_start: float
    image_stop: float
    azimuth_frequency: float
    slant_range_time: float
    range_frequency: float
    range_pixel_spacing: float
    azimuth_pixel_spacing: float
    range_sampling_rate: float
    wavelength: float
    bands: dict[str, str]

    @property
    def azimuth_time_interval(self) -> float:
        return 1.0 / self.azimuth_frequency

    @property
    def range_time_interval(self):
        return 1.0 / self.range_sampling_rate

    def get_image_path(self, band_name: str) -> str:
        return self.bands[band_name]

    def try_open_as_complex(self) -> ImageReader:
        real_paths = [
            path for name, path in self.bands.items() if name.startswith("i_")
        ]
        if len(real_paths) != 1:
            raise Exception("cannot find the real band")
        real_path = real_paths[0]
        imag_paths = [
            path for name, path in self.bands.items() if name.startswith("q_")
        ]
        if len(imag_paths) != 1:
            raise Exception("cannot find the imag band")
        imag_path = imag_paths[0]

        real = rasterio.open(real_path)
        imag = rasterio.open(imag_path)

        profile = real.profile.copy()
        profile["dtype"] = np.complex64

        @dataclass(frozen=True)
        class Reader(ImageReader):
            profile: dict[str, Any]

            def read(
                self,
                indexes: Optional[Union[int, Sequence[int]]],
                window: Window,  # the window argument is not optional in eos.sar.io since we want to work with crop first
                **kwargs: Any,
            ) -> NDArray[Any]:
                re = real.read(indexes, window=window, **kwargs)
                im = imag.read(indexes, window=window, **kwargs)
                return (re + 1j * im).astype(np.complex64)

        return Reader(profile=profile)


def _parse_product(
    prd: dict[str, Any], band_path_per_name: dict[str, str]
) -> SnapMetadata:
    attrs = prd["MDATTR"]
    elems = prd["MDElem"]

    product_id = [a for a in attrs if a["@name"] == "PRODUCT"][0]["#text"]

    look_side = [a for a in attrs if a["@name"] == "antenna_pointing"][0]["#text"]
    assert look_side in ("right", "left")

    orbit_direction = [a for a in attrs if a["@name"] == "PASS"][0]["#text"]
    assert orbit_direction in ("ASCENDING", "DESCENDING")

    assert [a for a in attrs if a["@name"] == "SAMPLE_TYPE"][0]["#text"] == "COMPLEX"
    assert float([a for a in attrs if a["@name"] == "azimuth_looks"][0]["#text"]) == 1.0
    assert float([a for a in attrs if a["@name"] == "range_looks"][0]["#text"]) == 1.0
    range_spacing = float(
        [a for a in attrs if a["@name"] == "range_spacing"][0]["#text"]
    )
    azimuth_spacing = float(
        [a for a in attrs if a["@name"] == "azimuth_spacing"][0]["#text"]
    )
    radar_frequency = float(
        [a for a in attrs if a["@name"] == "radar_frequency"][0]["#text"]
    )
    line_time_interval = float(
        [a for a in attrs if a["@name"] == "line_time_interval"][0]["#text"]
    )
    height = int([a for a in attrs if a["@name"] == "num_output_lines"][0]["#text"])
    width = int([a for a in attrs if a["@name"] == "num_samples_per_line"][0]["#text"])
    assert (
        int([a for a in attrs if a["@name"] == "is_terrain_corrected"][0]["#text"]) == 0
    )
    slant_range_to_first_pixel = float(
        [a for a in attrs if a["@name"] == "slant_range_to_first_pixel"][0]["#text"]
    )
    range_sampling_rate = float(
        [a for a in attrs if a["@name"] == "range_sampling_rate"][0]["#text"]
    )

    first_line_time = [a for a in attrs if a["@name"] == "first_line_time"][0]["#text"]
    first_line_time = dateutil.parser.parse(first_line_time)

    last_line_time = [a for a in attrs if a["@name"] == "last_line_time"][0]["#text"]
    last_line_time = dateutil.parser.parse(last_line_time)

    first_near_lat = float(
        [a for a in attrs if a["@name"] == "first_near_lat"][0]["#text"]
    )
    first_near_lon = float(
        [a for a in attrs if a["@name"] == "first_near_long"][0]["#text"]
    )
    first_far_lat = float(
        [a for a in attrs if a["@name"] == "first_far_lat"][0]["#text"]
    )
    first_far_lon = float(
        [a for a in attrs if a["@name"] == "first_far_long"][0]["#text"]
    )
    last_near_lat = float(
        [a for a in attrs if a["@name"] == "last_near_lat"][0]["#text"]
    )
    last_near_lon = float(
        [a for a in attrs if a["@name"] == "last_near_long"][0]["#text"]
    )
    last_far_lat = float([a for a in attrs if a["@name"] == "last_far_lat"][0]["#text"])
    last_far_lon = float(
        [a for a in attrs if a["@name"] == "last_far_long"][0]["#text"]
    )

    approx_geom = [
        (first_near_lon, first_near_lat),
        (first_far_lon, first_far_lat),
        (last_far_lon, last_far_lat),
        (last_near_lon, last_near_lat),
    ]

    # sar-commons/src/main/java/eu/esa/sar/commons/SARUtils.java:getRadarWavelength
    wavelength = LIGHT_SPEED_M_PER_SEC / (radar_frequency * 1e6)

    statevectors = [e for e in elems if e["@name"] == "Orbit_State_Vectors"][0][
        "MDElem"
    ]
    svs: list[StateVector] = []
    for svelem in statevectors:
        svelem = svelem["MDATTR"]
        time = [a for a in svelem if a["@name"] == "time"][0]["#text"]
        time = dateutil.parser.parse(time)
        x = float([a for a in svelem if a["@name"] == "x_pos"][0]["#text"])
        y = float([a for a in svelem if a["@name"] == "y_pos"][0]["#text"])
        z = float([a for a in svelem if a["@name"] == "z_pos"][0]["#text"])
        vx = float([a for a in svelem if a["@name"] == "x_vel"][0]["#text"])
        vy = float([a for a in svelem if a["@name"] == "y_vel"][0]["#text"])
        vz = float([a for a in svelem if a["@name"] == "z_vel"][0]["#text"])
        sv = StateVector(
            time=time.timestamp(),
            position=(x, y, z),
            velocity=(vx, vy, vz),
        )
        svs.append(sv)

    range_frequency = LIGHT_SPEED_M_PER_SEC / (2.0 * range_spacing)
    slant_range_time = slant_range_to_first_pixel / range_spacing / range_frequency

    bands = ([a for a in attrs if a["@name"] == "Slave_bands"] or [{"#text": ""}])[0][
        "#text"
    ].split(" ")
    # for primary products, bands will be empty, so we fill it in parse_snap_metadatas
    bands = {b: band_path_per_name[b] for b in bands if b in band_path_per_name}

    return SnapMetadata(
        product_id=product_id,
        state_vectors=svs,
        orbit_direction="ascending" if orbit_direction == "ASCENDING" else "descending",
        look_side="right" if look_side == "right" else "left",
        width=width,
        height=height,
        approx_geom=approx_geom,
        image_start=first_line_time.timestamp(),
        image_stop=last_line_time.timestamp(),
        azimuth_frequency=1.0 / line_time_interval,
        slant_range_time=slant_range_time,
        range_frequency=range_frequency,
        range_pixel_spacing=range_spacing,
        azimuth_pixel_spacing=azimuth_spacing,
        range_sampling_rate=range_sampling_rate,
        wavelength=wavelength,
        bands=bands,
    )


def parse_snap_metadatas(path: str) -> list[SnapMetadata]:
    root = os.path.dirname(os.path.realpath(path))
    xml_content = open(path).read()
    data = xmltodict.parse(xml_content)["Dimap_Document"]

    # image interpretation (band id <> band name)
    band_infos = data["Image_Interpretation"]["Spectral_Band_Info"]
    band_name_per_index: dict[int, str]
    band_name_per_index = {
        int(info["BAND_INDEX"]): info["BAND_NAME"] for info in band_infos
    }
    # data access (filepath <> band id)
    data_files = data["Data_Access"]["Data_File"]
    band_path_per_name: dict[str, str]
    band_path_per_name = {
        band_name_per_index[int(d["BAND_INDEX"])]: os.path.join(
            root, d["DATA_FILE_PATH"]["@href"].rstrip(".hdr") + ".img"
        )
        for d in data_files
    }

    sources = data["Dataset_Sources"]["MDElem"]["MDElem"]
    datasets = []

    primary = [s for s in sources if s["@name"] == "Abstracted_Metadata"][0]
    primary_metadata = _parse_product(primary, band_path_per_name)
    datasets.append(primary_metadata)

    secondaries = [s for s in sources if s["@name"] == "Slave_Metadata"]
    secondaries = secondaries[0]["MDElem"] if secondaries else []
    used_bands: set[str] = set()
    for sec in secondaries:
        metadata = _parse_product(sec, band_path_per_name)
        datasets.append(metadata)
        used_bands |= metadata.bands.keys()

    # assign all unused bands to the primary product
    primary_metadata.bands.update(
        {
            name: path
            for name, path in band_path_per_name.items()
            if name not in used_bands
        }
    )

    return datasets


@dataclass(frozen=True)
class SnapModel(SensorModel):
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
        meta: SnapMetadata, orbit_degree: int, corrector: Corrector = Corrector()
    ) -> SnapModel:
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

        return SnapModel(
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


if __name__ == "__main__":
    import fire

    def main(file):
        from eos.dem import DEMStitcherSource
        from eos.sar.ortho import LanczosInterpolation, Orthorectifier
        from eos.sar.roi import Roi

        metas = parse_snap_metadatas(file)
        print(len(metas), "products")
        meta = metas[0]

        model = SnapModel.from_metadata(meta, orbit_degree=5)

        lon, lat = zip(*meta.approx_geom)
        bounds = model.projection(lon, lat, alt=np.zeros_like(lon))
        print(bounds)
        print(meta.bands)

        reader = meta.try_open_as_complex()
        roi = Roi(0, 0, meta.width, meta.height)
        arr = io.read_window(reader, roi)
        assert arr.dtype == np.complex64
        arr = np.abs(arr)
        np.save("/tmp/a", arr)

        dem_source = DEMStitcherSource()
        dem = model.fetch_dem(dem_source, roi)
        orthorectifier = Orthorectifier.from_roi(model, roi, 1, dem=dem)
        raster = orthorectifier.apply(arr, LanczosInterpolation)

        profile = reader.profile.copy()  # type: ignore
        profile["crs"] = orthorectifier.crs
        profile["transform"] = orthorectifier.transform
        profile["width"] = raster.shape[1]
        profile["height"] = raster.shape[0]
        profile["dtype"] = raster.dtype
        with rasterio.open("/tmp/b.tif", "w+", **profile) as dst:
            dst.write(raster, 1)

    fire.Fire(main)
