from __future__ import annotations

import abc
import concurrent.futures
import dataclasses
import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import dateutil.parser
import numpy as np
import rasterio
import rasterio.io
import rasterio.session
import shapely.geometry
from numpy.typing import NDArray
from rasterio import DatasetReader
from typing_extensions import override

import eos.cache
import eos.dem
import eos.sar.regist
from eos.products import sentinel1
from eos.products.sentinel1 import orbit_catalog
from eos.products.sentinel1.metadata import Sentinel1GRDMetadata
from eos.products.sentinel1.product import (
    PhoenixSentinel1GRDProductInfo,
    Sentinel1GRDProductInfo,
)
from eos.products.sentinel1.proj_model import Sentinel1GRDModel
from eos.sar import (  # type: ignore[attr-defined]
    io,
    simulator,
)
from eos.sar.io import read_window
from eos.sar.orbit import Orbit
from eos.sar.roi import Roi
from eos.sar.rtc import RadiometricTerrainCorrector

GRDProductProvider = Callable[[str], Sentinel1GRDProductInfo]

# rasterio profile default settings used to store slices as tiffs.
SLICE_BASE_PROFILE = {
    "driver": "GTiff",
    "count": 1,
    "nodata": np.nan,  # particularly important for boundless reads
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    # compression only allows for a 25% size reduction, but 10x slower read and write speed
    "compress": None,
}


@dataclass
class CannotFindOrGenerateSlice(Exception):
    slice_spec: SliceSpec


@dataclass(frozen=True)
class SlicingSpec:
    """
    Defines slicing parameters, in particular for the temporal grid inside one relative orbit.
    The origin (0) represents the ANX time.

    The margin parameters define the overlap between slices (in azimuth) and additional margin in range.
    These margins are necessary since the generated slices are registered to products when calibrating.
    """

    milliseconds_per_slice: int
    """
    For example, 3070 milliseconds is approximatly 2050 rows, depending on `azimuth_time_interval` of the product.
    Note: this field is an integer so that we can format it into a string without rounding errors.
    """
    row_margin: int
    """
    Row margin before/after the slice.
    """
    near_col_margin: int
    """
    Column margin on the left of the slice.
    """
    far_col_margin: int
    """
    Column margin on the right of the slice.
    """

    @property
    def seconds_per_slice(self) -> float:
        return self.milliseconds_per_slice / 1000

    def compute_approx_start_anx(self, slice_id: int) -> float:
        """Approximate start time of the given slice_id. It does not consider the margins."""
        return slice_id * self.seconds_per_slice

    def compute_approx_center_anx(self, slice_id: int) -> float:
        """Approximate time in seconds since ANX of the given slice_id."""
        return self.compute_approx_start_anx(slice_id) + self.seconds_per_slice / 2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SlicingSpec:
        return SlicingSpec(
            milliseconds_per_slice=d["milliseconds_per_slice"],
            row_margin=d["row_margin"],
            near_col_margin=d["near_col_margin"],
            far_col_margin=d["far_col_margin"],
        )


@dataclass(frozen=True)
class SliceSpec:
    """
    Specification of a slice.
    """

    slicing_spec: SlicingSpec
    relative_orbit_number: int
    slice_id: int
    approx_lonlat: shapely.geometry.Point = dataclasses.field(repr=False, compare=False)
    """
    This field is only used to speed-up catalog queries, and can be very coarse (~5° precision). 
    Thus two slices can be equal even if they have different approx_lonlat values.
    """

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SliceSpec:
        return SliceSpec(
            slicing_spec=SlicingSpec.from_dict(d["slicing_spec"]),
            relative_orbit_number=d["relative_orbit_number"],
            slice_id=d["slice_id"],
            approx_lonlat=d["approx_lonlat"],
        )


@dataclass(frozen=True)
class LivingSlice:
    """
    GRD Metadata and image reader corresponding to a slice.

    This object is useful to represent existing slice without requiring them to be in-memory.
    The reader can be called to access a ROI.
    """

    meta: Sentinel1GRDMetadata
    reader: DatasetReader

    @property
    def proj_model(self) -> Sentinel1GRDModel:
        """Projection model associated with the slice."""
        orbit = Orbit(self.meta.state_vectors)
        proj_model = sentinel1.proj_model.grd_model_from_meta(self.meta, orbit)
        return proj_model


@dataclass(frozen=True)
class SliceProvider:
    """
    Glue class to store and generate slices on-demand.
    The generator is optional, in which case this object only relies on the storage.
    """

    storage: SliceStorage
    generator: Optional[SliceGenerator]

    def get_slice(self, slice_spec: SliceSpec) -> Optional[LivingSlice]:
        """Fetch an existing slice from the storage. Returns None if the slice does not exist."""
        return self.storage.load_slice(slice_spec)

    def get_slice_or_generate(self, slice_spec: SliceSpec) -> Optional[LivingSlice]:
        """
        Fetch an existing slice from the storage, or generate it and store it.
        Returns None if the slice is not in storage and self.generator is None.
        """
        if living_slice := self.get_slice(slice_spec):
            return living_slice
        if self.generator and (info := self.generator.generate_slice(slice_spec)):
            meta, array = info
            return self.storage.store_slice(slice_spec, meta, array)
        return None


@dataclass(frozen=True)
class SliceGenerationOptions:
    """
    Options for the slice generations.
    Changing these options should not affect drastically the simulation.
    """

    col_margins_for_tiling: int
    """Number of columns added during the tiling when simulating the SAR image to avoid overlay issues. `100` might be enough."""
    col_step_for_tiling: int
    """Number of columns for the tiling. Allows to reduce memory usage and speed-up the generation using multithreading.
    This also helps to keep simulated ROI small, reducing errors due to the simulator assumptions."""
    max_workers: Optional[int]
    """
    See ThreadPoolExecutor.
    Given the that the RTC cython code still grabs the GIL a lot, the optimal choice is ~4 workers for now.
    """


class SliceGenerator(abc.ABC):
    @abc.abstractmethod
    def generate_slice(
        self, slice_spec: SliceSpec
    ) -> tuple[Sentinel1GRDMetadata, NDArray[np.float32]]:
        """Generate a slice given a slice specification."""


@dataclass(frozen=True)
class SliceStorage(abc.ABC):
    @abc.abstractmethod
    def load_slice(self, slice_spec: SliceSpec) -> Optional[LivingSlice]:
        """
        Fetch a slice from storage given a slice specification.
        Returns None if the slice does not exist.
        """

    @abc.abstractmethod
    def store_slice(
        self,
        slice_spec: SliceSpec,
        meta: Sentinel1GRDMetadata,
        array: NDArray[np.float32],
    ) -> LivingSlice:
        """
        Store the given in-memory slice to storage.
        """


def _slice_meta_to_json(meta: Sentinel1GRDMetadata, slice_spec: SliceSpec) -> str:
    data = {
        "meta": meta.to_dict(),
        "slice_spec": slice_spec.to_dict(),
    }
    buf = json.dumps(data, indent=2, default=str)
    return buf


def _slice_meta_from_json(
    ref_slice_spec: SliceSpec, content: str
) -> Sentinel1GRDMetadata:
    data = json.loads(content)

    # consistency check that we are indeed talking about the right slice
    actual_slice_spec = SliceSpec.from_dict(data["slice_spec"])
    assert actual_slice_spec == ref_slice_spec, (actual_slice_spec, ref_slice_spec)

    meta = Sentinel1GRDMetadata.from_dict(data["meta"])
    return meta


def _formatted_slicespec(slice_spec: SliceSpec, prefix: str) -> str:
    ron = slice_spec.relative_orbit_number
    slice_id = slice_spec.slice_id
    millisecs = slice_spec.slicing_spec.milliseconds_per_slice
    row_margin = slice_spec.slicing_spec.row_margin
    near_col_margin = slice_spec.slicing_spec.near_col_margin
    far_col_margin = slice_spec.slicing_spec.far_col_margin
    return f"{prefix}orbit-{ron:03d}_slice_id-{slice_id}_millisecs-{millisecs}_rowmargin-{row_margin}_nearcolmargin-{near_col_margin}_farcolmargin-{far_col_margin}"


@dataclass(frozen=True)
class LocalSliceStorage(SliceStorage):
    """Slice storage on a given filesystem directory."""

    directory: Path
    filename_prefix: str = ""

    @override
    def load_slice(self, slice_spec: SliceSpec) -> Optional[LivingSlice]:
        fmt = _formatted_slicespec(slice_spec, self.filename_prefix)
        if not io.exists(f"{self.directory}/{fmt}.tif"):
            return None

        meta_path = f"{self.directory}/{fmt}-meta.json"
        content = io.read_file_as_str(meta_path)
        meta = _slice_meta_from_json(slice_spec, content)

        src = rasterio.open(f"{self.directory}/{fmt}.tif")
        return LivingSlice(meta=meta, reader=src)

    @override
    def store_slice(
        self,
        slice_spec: SliceSpec,
        meta: Sentinel1GRDMetadata,
        array: NDArray[np.float32],
    ) -> LivingSlice:
        fmt = _formatted_slicespec(slice_spec, self.filename_prefix)

        with open(f"{self.directory}/{fmt}-meta.json", "w") as dst:
            dst.write(_slice_meta_to_json(meta, slice_spec))

        profile = SLICE_BASE_PROFILE | {
            "width": array.shape[1],
            "height": array.shape[0],
            "dtype": array.dtype,
        }
        with rasterio.open(f"{self.directory}/{fmt}.tif", "w", **profile) as dst:
            dst.write(array, 1)
        src = rasterio.open(f"{self.directory}/{fmt}.tif")

        return LivingSlice(meta=meta, reader=src)


@dataclass(frozen=True)
class S3SliceStorage(SliceStorage):
    s3_client: Any
    s3_session: Any
    bucket: str
    key_prefix: str

    @override
    def load_slice(self, slice_spec: SliceSpec) -> Optional[LivingSlice]:
        fmt = _formatted_slicespec(slice_spec, self.key_prefix)
        if not io.exists(f"s3://{self.bucket}/{fmt}.tif", self.s3_client):
            return None

        meta_path = f"s3://{self.bucket}/{fmt}-meta.json"
        content = io.read_file_as_str(meta_path, self.s3_client)
        meta = _slice_meta_from_json(slice_spec, content)

        with rasterio.Env(rasterio.session.AWSSession(self.s3_session)):
            src = rasterio.open(f"s3://{self.bucket}/{fmt}.tif")

        return LivingSlice(meta=meta, reader=src)

    @override
    def store_slice(
        self,
        slice_spec: SliceSpec,
        meta: Sentinel1GRDMetadata,
        array: NDArray[np.float32],
    ) -> LivingSlice:
        fmt = _formatted_slicespec(slice_spec, self.key_prefix)

        content = _slice_meta_to_json(meta, slice_spec)
        self.s3_client.put_object(
            Body=content,
            Bucket=self.bucket,
            Key=f"{fmt}-meta.json",
        )

        profile = SLICE_BASE_PROFILE | {
            "width": array.shape[1],
            "height": array.shape[0],
            "dtype": array.dtype,
        }

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(array, 1)
            memfile.seek(0)
            buf = memfile.read()
            self.s3_client.put_object(
                Body=buf,
                Bucket=self.bucket,
                Key=f"{fmt}.tif",
            )

        with rasterio.Env(rasterio.session.AWSSession(self.s3_session)):
            src = rasterio.open(f"s3://{self.bucket}/{fmt}.tif")

        return LivingSlice(meta=meta, reader=src)


class CannotFindReferenceMetadataException(Exception): ...


@dataclass(frozen=True)
class PhxProductBasedSliceGenerator(SliceGenerator):
    """
    This generator looks at the Sentinel-1 catalog to find products overlapping with the requested slice.
    The projection model of the resulting simulation will be derived from found products.
    """

    phx_collection: Any
    """ Kayrros specific: collection: esa-sentinel-1-csar-l1-grd with a source """
    anx: dict[str, Any]
    """ Kayrros specific: collection: esa-sentinel-1-csar-l1-grd, static_assets: "ANX" """
    dem_source: eos.dem.DEMSource
    options: SliceGenerationOptions
    orbit_catalog_backend: Optional[orbit_catalog.Sentinel1OrbitCatalogBackend]
    product_provider: GRDProductProvider

    @override
    def generate_slice(
        self, slice_spec: SliceSpec
    ) -> tuple[Sentinel1GRDMetadata, NDArray[np.float32]]:
        # TODO: get consecutive products for assembly
        product_info = self._search_reference_product_info(slice_spec)
        if not product_info:
            raise CannotFindReferenceMetadataException(
                f"cannot find reference for the slice {slice_spec}"
            )

        meta, proj_model = _make_proj_model([product_info], self.orbit_catalog_backend)
        roi = _compute_roi_for_slice(slice_spec, meta, proj_model)
        simulation = _simulate_slice(self.options, proj_model, roi, self.dem_source)
        new_meta = _adjust_grd_metadata(meta, proj_model, roi)
        return new_meta, simulation

    def _search_reference_product_info(
        self, slice_spec: SliceSpec
    ) -> Optional[Sentinel1GRDProductInfo]:
        import phoenix as phx

        relative_orbit_number = slice_spec.relative_orbit_number
        slice_id = slice_spec.slice_id
        approx_lonlat = slice_spec.approx_lonlat

        filters = [
            phx.catalog.Field("sentinel1:sensor_mode") == "IW",
            phx.catalog.Field("sentinel1:relative_orbit_number")
            == relative_orbit_number,
            phx.catalog.Geometry.intersects(approx_lonlat.buffer(5.0)),
            # skip products that don't have a POE aux file
            # because we use the POE ANX time (from self.anx) later
            phx.catalog.Field("datetime")
            < datetime.datetime.now() - datetime.timedelta(days=22),
        ]

        items = list(self.phx_collection.list_items(filters=filters, results=10000))
        # reverse search, assuming newer products have better orbits (probably not true anymore since 2024)
        items = items[::-1]

        for it in items:
            # TODO: handle orbit crossing
            slice_ids = _phx_item_to_slice_ids(it, slice_spec.slicing_spec, self.anx)

            if slice_id in slice_ids:
                product = self.product_provider(it.id)
                return product

        return None


@dataclass(frozen=True)
class SpecificProductSliceGenerator(SliceGenerator):
    """
    This generator is mostly for testing purposes.
    """

    product_info: Sentinel1GRDProductInfo
    dem_source: eos.dem.DEMSource
    options: SliceGenerationOptions

    @override
    def generate_slice(
        self, slice_spec: SliceSpec
    ) -> tuple[Sentinel1GRDMetadata, NDArray[np.float32]]:
        meta, proj_model = _make_proj_model(
            [self.product_info], orbit_catalog_backend=None
        )
        roi = _compute_roi_for_slice(slice_spec, meta, proj_model)
        simulation = _simulate_slice(self.options, proj_model, roi, self.dem_source)
        new_meta = _adjust_grd_metadata(meta, proj_model, roi)
        return new_meta, simulation


def _to_timestamp(s: str) -> float:
    return dateutil.parser.parse(s).timestamp()


def _phx_item_to_slice_ids(
    item: Any, slicing_spec: SlicingSpec, anx: dict[str, Any]
) -> list[int]:
    """
    Return the list of slice id that intersect with the given GRD item.
    This function requires the item to be old enough to have POE ANX processed.
    """
    seconds_per_slice = slicing_spec.seconds_per_slice

    platform = f"sentinel-{item.id[1:3].lower()}"
    absolute_orbit_number = item.properties["sentinel1:orbit_number"]
    times = anx["platform"][platform][absolute_orbit_number]
    anx_time = times["poe"]

    anx_timestamp = _to_timestamp(anx_time)
    begin_timestamp = _to_timestamp(item.properties["sentinel1:begin_position"])
    end_timestamp = _to_timestamp(item.properties["sentinel1:end_position"])

    begin_anx = begin_timestamp - anx_timestamp
    end_anx = end_timestamp - anx_timestamp

    first_slice = math.floor(begin_anx / seconds_per_slice)
    last_slice = math.ceil(end_anx / seconds_per_slice)

    return list(range(first_slice, last_slice))


def _make_proj_model(
    products: list[Sentinel1GRDProductInfo],
    orbit_catalog_backend: Optional[orbit_catalog.Sentinel1OrbitCatalogBackend],
) -> tuple[Sentinel1GRDMetadata, Sentinel1GRDModel]:
    assert len(products) > 0

    # search for better state vectors, but it's not a big deal if we don't find them
    if orbit_catalog_backend:
        query = orbit_catalog.Sentinel1OrbitCatalogQuery(
            product_ids=[p.product_id for p in products],
            quality=orbit_catalog.BestEffort,
        )
        statevectors = orbit_catalog.search(orbit_catalog_backend, query).single()
    else:
        statevectors = None

    class_to_pol = {"SV": "vv", "DV": "vv", "SH": "hh", "DH": "hh"}
    polarization_class = products[0].product_id[14:16]
    polarization = class_to_pol[polarization_class]

    assembler = sentinel1.assembler.Sentinel1GRDAssembler.from_products(
        products, polarization, statevectors
    )
    meta = assembler.get_metadata()
    proj_model = assembler.get_proj_model()

    return meta, proj_model


def _compute_roi_for_slice(
    slice_spec: SliceSpec,
    meta: Sentinel1GRDMetadata,
    proj_model: Sentinel1GRDModel,
) -> Roi:
    slicing_spec = slice_spec.slicing_spec
    azt0 = meta.anx_time + slice_spec.slice_id * slicing_spec.seconds_per_slice
    row0 = math.floor(proj_model.coordinate.to_row(azt0))
    row1 = math.floor(
        proj_model.coordinate.to_row(azt0 + slicing_spec.seconds_per_slice)
    )

    row_margin = slicing_spec.row_margin
    near_col_margin = slicing_spec.near_col_margin
    far_col_margin = slicing_spec.far_col_margin

    roi = Roi(
        col=-near_col_margin,
        row=row0 - row_margin,
        w=proj_model.w + near_col_margin + far_col_margin,
        h=row1 - row0 + row_margin * 2,
    )

    return roi


def _simulate_slice(
    options: SliceGenerationOptions,
    proj_model: Sentinel1GRDModel,
    roi: Roi,
    dem_source: eos.dem.DEMSource,
) -> NDArray[np.float32]:
    dem = proj_model.fetch_dem(dem_source, roi)
    sarsim = simulator.SARSimulator(proj_model, dem)
    margin = options.col_margins_for_tiling

    def process(subroi: Roi):
        subroimargin, roi_inside = subroi.add_custom_margin(((0, 0), (margin, margin)))
        buf = sarsim.simulate(subroimargin, extends_roi_n_grid=100).astype(np.float32)
        assert buf.shape == subroimargin.get_shape()
        buf = roi_inside.crop_array(buf)
        assert buf.shape == subroi.get_shape()
        return buf, subroi

    shape = roi.get_shape()
    simulation = np.full(shape, np.nan, np.float32)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=options.max_workers
    ) as executor:
        futures = []
        x_step = options.col_step_for_tiling
        for x in range(roi.col, roi.w, x_step):
            subroi = Roi(col=x, row=roi.row, w=x_step, h=roi.h)
            subroi = subroi.clip(roi)
            future = executor.submit(process, subroi)
            futures.append(future)

        for fut in futures:
            buf, subroi = fut.result()
            translated = subroi.translate_roi(col=-roi.col, row=-roi.row)
            translated.crop_array(simulation)[:, :] = buf

    return simulation


def _adjust_grd_metadata(
    meta: Sentinel1GRDMetadata,
    proj_model: Sentinel1GRDModel,
    roi: Roi,
) -> Sentinel1GRDMetadata:
    """
    Warning: this does not adjust the SRGR coefficients, range coordinates have to be offset to use the proj model.
    TODO: re-fit the SRGR coefficients to match the ROI. See the HACK note in `calibrate`.
    """
    old_meta = meta.to_dict()
    del old_meta["width"]
    del old_meta["height"]
    del old_meta["image_start"]
    del old_meta["image_end"]
    del old_meta["srgr"]
    del old_meta["approx_geom"]
    del old_meta["approx_altitude"]
    del old_meta["state_vectors"]
    new_meta = Sentinel1GRDMetadata(
        width=roi.w,
        height=roi.h,
        srgr=meta.srgr,  # NOTE: for now we don't touch the SRGR coordinate system
        image_start=float(proj_model.coordinate.to_azt(roi.row)),
        image_end=float(proj_model.coordinate.to_azt(roi.row + roi.h - 1)),
        approx_geom=meta.approx_geom,  # TODO
        approx_altitude=meta.approx_altitude,  # TODO
        # this is just because the to_dict() converts the statevectors to dict
        # but the constructor of Sentinel1GRDMetadata requires objects
        state_vectors=meta.state_vectors,
        **old_meta,
    )
    return new_meta


def calibrate(
    slice_provider: SliceProvider,
    slicing_spec: SlicingSpec,
    meta: Sentinel1GRDMetadata,
    proj_model: Sentinel1GRDModel,
    roi: Roi,
    array: NDArray[np.float32],
    debug: bool = False,
) -> None:
    assert array.shape == roi.get_shape()

    relative_orbit_number = meta.relative_orbit_number
    t_anx = meta.anx_time

    first_line = roi.row
    last_line = roi.row + roi.h - 1
    first_time = proj_model.coordinate.to_azt(first_line)
    last_time = proj_model.coordinate.to_azt(last_line)

    first_slice = math.floor((first_time - t_anx) / slicing_spec.seconds_per_slice)
    last_slice = math.ceil((last_time - t_anx) / slicing_spec.seconds_per_slice)

    approx_lonlat = shapely.geometry.Point(
        proj_model.approx_centroid_lon, proj_model.approx_centroid_lat
    )
    approx_alt = float(np.mean(meta.approx_altitude))

    # keep track of rows that were calibrated during the iterations
    touched_rows = np.zeros(roi.h, dtype=np.bool_)
    if debug:
        touched_dbg = np.full(roi.get_shape(), dtype=np.int16, fill_value=-99)
        rtc_mosaic = np.zeros_like(array)

    # TODO: handle orbit crossing
    for slice_id in range(first_slice, last_slice):
        slice_spec = SliceSpec(
            slicing_spec=slicing_spec,
            relative_orbit_number=relative_orbit_number,
            slice_id=slice_id,
            approx_lonlat=approx_lonlat,
        )

        living_slice = slice_provider.get_slice_or_generate(slice_spec)
        if not living_slice:
            raise CannotFindOrGenerateSlice(slice_spec)

        # poor's man registration: estimate a translation from a single arbitrary pixel
        # TODO: this can be improved by registering with an affine transform, maybe using gcps from the slices
        slice_proj_model = living_slice.proj_model
        # using the correct altitude doesn't affect much the result, within the error of the simulator anyway
        alt = approx_alt
        lon, lat, _ = slice_proj_model.localization(
            0,
            # HACK: we evaluate the localization at -near_col_margin, because the unmodified srgr cannot take into account
            # the fact that the slice actually starts at -near_col_margin, so we do it manually here
            -slicing_spec.near_col_margin,
            alt,
        )
        dy, dx, _ = proj_model.projection(lon, lat, alt)

        # compute the roi of the slice that is interesting for this calibration
        slice_roi = Roi(
            0,
            0,
            living_slice.meta.width,
            # -1 because the simulation tends to leave the last line a bit weaker.
            # This is an issue with the simulation.
            living_slice.meta.height - 1,
        )
        slice_in_product = slice_roi.translate_roi(int(dx), int(dy))

        if not slice_in_product.intersects_roi(roi):
            continue

        clipped_slice_in_product = slice_in_product.clip(roi)
        clipped_slice_in_roi = clipped_slice_in_product.translate_roi(
            -roi.col, -roi.row
        )

        # residual transform to apply
        res_dx = int(dx) - dx
        res_dy = int(dy) - dy
        A = eos.sar.regist.translation_matrix(res_dx, res_dy)

        # define the part of the slice that will be read
        # add some margins to allow for resampling the simulation
        slice_read_roi = clipped_slice_in_product.translate_roi(
            -slice_in_product.col, -slice_in_product.row
        )
        margin = 5
        margin_slice_read_roi = slice_read_roi.add_margin(margin)

        # read the slice
        slice_array = read_window(
            living_slice.reader,
            margin_slice_read_roi,
            get_complex=False,
            boundless=True,
        )

        # resample the slice (nans are added on borders by the resampling)
        dst_shape = slice_read_roi.get_shape()
        A2 = eos.sar.regist.change_resamp_mat_orig(margin, margin, 0, 0, A)
        slice_array_resamp = eos.sar.regist.apply_affine(
            slice_array,
            A2,
            dst_shape,
            # it's important to use bilinear here to avoid ringing and introducing negative values
            interpolation=eos.sar.regist.LinearInterpolation,
        )

        row = clipped_slice_in_roi.row
        h = clipped_slice_in_roi.h

        # the mask is a 1d array, indicating which rows should be considered
        mask = (
            ~np.all(np.isnan(slice_array_resamp), axis=1) & ~touched_rows[row : row + h]
        )

        # calibrate the array using the registered slice on the rows defined by `mask`
        arr = clipped_slice_in_roi.crop_array(array)
        normalized = eos.sar.rtc.normalize(arr, slice_array_resamp)
        arr[mask] = normalized[mask]

        # mark the rows as touched, if they were effectively not nan (due to resampling for example)
        assert not touched_rows[row : row + h][mask].any()
        touched_rows[row : row + h] |= mask

        if debug:
            # REMOVEME: this is used to check that the touched_rows strategy works as expected
            assert np.all(touched_dbg[row : row + h, :][mask] == -99)
            touched_dbg[row : row + h, :][mask] = np.arange(
                slice_read_roi.row,
                slice_read_roi.row + slice_read_roi.h,
            )[mask, None]
            if False:
                np.save("/tmp/touched_dbg", touched_dbg)
                np.save(f"/tmp/s{slice_id}.npy", slice_array)
                np.save(f"/tmp/r{slice_id}.npy", slice_array_resamp)
                np.save(f"/tmp/m{slice_id}.npy", mask[None, :, None, None])

                # check the slice mosaicing
                t = clipped_slice_in_roi.crop_array(rtc_mosaic)
                t[mask] += slice_array_resamp[mask]

    if debug:
        if False:
            np.save("/tmp/touchedrows", touched_rows[None, :, None, None])
            np.save("/tmp/test.npy", rtc_mosaic)
        assert np.all(touched_dbg != -99)
    assert np.all(touched_rows)


if __name__ == "__main__":

    def main(
        product_id: str = "S1A_IW_GRDH_1SDV_20220527T130626_20220527T130651_043397_052EB3_7733",
        calibration: str | None = None,
    ):
        import os
        import time

        import phoenix as phx

        client = phx.catalog.Client()
        collection = client.get_collection("esa-sentinel-1-csar-l1-grd")
        anx = collection.static_assets.get("ANX").download_as_bytes()
        anx = json.loads(anx)

        dem_source = eos.dem.DEMStitcherSource()

        orbit_catalog_backend = orbit_catalog.PhoenixSentinel1OrbitCatalogBackend(
            collection_source=phx.catalog.Client()
            .get_collection("esa-sentinel-1-csar-aux")
            .at("aws:proxima:kayrros-prod-sentinel-aux")
        )

        generator = PhxProductBasedSliceGenerator(
            phx_collection=collection.at("aws:proxima:sentinel-s1-l1c"),
            anx=anx,
            dem_source=dem_source,
            options=SliceGenerationOptions(
                col_margins_for_tiling=100,
                col_step_for_tiling=2048,
                max_workers=4,
            ),
            orbit_catalog_backend=orbit_catalog_backend,
            product_provider=PhoenixSentinel1GRDProductInfo.from_product_id,
        )

        os.makedirs("tmp", exist_ok=True)
        slice_provider = SliceProvider(
            storage=LocalSliceStorage(directory=Path("./tmp")),
            generator=generator,
        )

        slicing_spec = SlicingSpec(
            milliseconds_per_slice=3070,
            row_margin=20,
            near_col_margin=100,
            far_col_margin=100,
        )

        product = PhoenixSentinel1GRDProductInfo.from_product_id(product_id)
        meta = sentinel1.metadata.extract_grd_metadata(product.get_xml_annotation("VV"))

        orbit_catalog_backend = orbit_catalog.PhoenixSentinel1OrbitCatalogBackend(
            collection_source=phx.catalog.Client()
            .get_collection("esa-sentinel-1-csar-aux")
            .at("aws:proxima:kayrros-prod-sentinel-aux")
        )
        query = orbit_catalog.Sentinel1OrbitCatalogQuery(
            product_ids=[product.product_id], quality=orbit_catalog.BestEffort
        )
        statevectors = orbit_catalog.search(orbit_catalog_backend, query).single()
        if statevectors is not None:
            meta = meta.with_new_state_vectors(statevectors, "")

        orbit = Orbit(meta.state_vectors)
        proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit)

        reader = product.get_image_reader("VV")
        if calibration:
            cal_xml = product.get_xml_calibration("VV")
            noise_xml = product.get_xml_noise("VV")
            ipf = product.ipf
            calibrator = sentinel1.calibration.Sentinel1Calibrator(
                cal_xml, noise_xml, ipf
            )
            reader = sentinel1.calibration.CalibrationReader(
                reader, calibrator, method=calibration
            )

        h, w = meta.height, meta.width
        roi = Roi(0, 0, w, h)
        roi = Roi(10000, 8000, 3000, 3500)
        print(roi)

        array: NDArray[np.float32]
        array = read_window(reader, roi, get_complex=False, out_dtype=np.float32)  # type: ignore
        old = array.copy()

        print("calibrate new product")
        t = time.time()
        calibrate(
            slice_provider, slicing_spec, meta, proj_model, roi, array, debug=True
        )
        print("calibrate new product", time.time() - t)

        np.save("./tmp/a.npy", old)
        np.save("./tmp/b.npy", array)
        del array

        dem = proj_model.fetch_dem(dem_source, roi)
        rtc = RadiometricTerrainCorrector(proj_model, dem, roi)
        corrected = rtc.apply(old)
        np.save("./tmp/c.npy", corrected)
        np.save("./tmp/c-sim.npy", rtc.get_simulation())

    import fire

    fire.Fire(main)
