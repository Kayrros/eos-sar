import concurrent.futures
import datetime
import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator, Optional, Union

import boto3
import shapely.wkt
import tqdm

import eos.cache
import eos.dem
import eos.products.sentinel1
import eos.products.sentinel1.catalog as s1_catalog
import eos.sar
from eos.cache import Cache
from eos.products.sentinel1 import orbit_catalog
from eos.products.sentinel1.metadata import Sentinel1BurstMetadata
from eos.products.sentinel1.overlap import Bsint, Osid
from eos.products.sentinel1.product import (
    CDSEUnzippedSafeSentinel1SLCProductInfo,
    Sentinel1SLCProductInfo,
)
from eos.sar.roi_provider import GeometryRoiProvider, RoiProvider
from teosar import inout
from teosar.workflow import (
    OvlPrimaryPipeline,
    OvlSecondaryPipeline,
    Pipeline,
    PrimaryPipeline,
    SecondaryPipeline,
)

logger = logging.getLogger(__name__)

ProductProvider = Callable[[str], Sentinel1SLCProductInfo]


def get_bsids_for_product(
    product_provider: ProductProvider, polarization: str, product_id: str
) -> list[str]:
    product = product_provider(product_id)
    xmls = [
        product.get_xml_annotation(swath=swath, pol=polarization)
        for swath in ("IW1", "IW2", "IW3")
    ]
    bursts: list[Sentinel1BurstMetadata] = sum(
        [eos.products.sentinel1.metadata.extract_bursts_metadata(xml) for xml in xmls],
        start=[],
    )
    bsids = [b.bsid for b in bursts]
    return bsids


def get_bsids_for_products(
    product_provider: ProductProvider, polarization: str, product_ids: list[str]
) -> tuple[dict[str, list[str]], dict[str, Exception]]:
    get = partial(get_bsids_for_product, product_provider, polarization)
    bsids: dict[str, list[str]] = {}
    errors: dict[str, Exception] = {}
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=10, mp_context=mp_context
    ) as pool:
        futures = {pool.submit(get, pid): pid for pid in product_ids}
        for future in concurrent.futures.as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                errors[pid] = e
            else:
                bsids[pid] = result

    return bsids, errors


def remove_weird_products(
    product_provider: ProductProvider, product_ids: list[list[str]], polarization: str
) -> list[list[str]]:
    def pid2datatake(pid: str) -> str:
        idx = len("S1A_IW_SLC__1SDV_20211202T173302_20211202T173329_040833_")
        return pid[idx : idx + 6]

    logger.info("getting bsids for products")
    bsids_per_pid, errored = get_bsids_for_products(
        product_provider, polarization, list(sum(product_ids, []))
    )
    logger.info("getting bsids for products, DONE")
    for pid, exception in errored.items():
        logger.info(f"skipped product {pid} ({exception})")

    by_datatake: dict[str, list[str]] = {}
    for pid in [pid for pids in product_ids for pid in pids]:
        by_datatake.setdefault(pid2datatake(pid), []).append(pid)

    good_datatakes: set[str] = set()
    for datatake, pids in by_datatake.items():
        all_valid = all(pid not in errored for pid in pids)
        if not all_valid:
            logger.info(
                f"skipped datatake {datatake} because one of its product has errored"
            )
            continue

        # check whether some bsids are duplicated in list of products
        bsids: list[str] = sum((bsids_per_pid[pid] for pid in pids), [])
        if len(bsids) == len(set(bsids)):
            good_datatakes.add(datatake)
        else:
            logger.warn(
                f"{pids} have duplicated burst ids, datake {datatake} removed completely from the timeseries."
            )

    good_product_ids: list[list[str]] = []
    for pids in product_ids:
        datatake = pid2datatake(pids[0])
        if datatake in good_datatakes:
            good_product_ids.append(pids)

    return good_product_ids


class BackendFactory(ABC):
    # Note: We could break this into three different factories if needed

    @abstractmethod
    def create_slc_catalog_backend(self) -> s1_catalog.Sentinel1SLCCatalogBackend: ...

    @abstractmethod
    def create_orbit_catalog_backend(
        self,
    ) -> orbit_catalog.Sentinel1OrbitCatalogBackend: ...

    @abstractmethod
    def create_product_provider(self) -> ProductProvider: ...


class PhoenixBackendFactory(BackendFactory):
    def create_slc_catalog_backend(self) -> s1_catalog.Sentinel1SLCCatalogBackend:
        """
        Helper function to get slc backend using phoenix default client for SLC collection at ASF source.
        """
        import phoenix.catalog

        client = phoenix.catalog.Client()
        collection = client.get_collection("esa-sentinel-1-csar-l1-slc").at(
            "asf:daac:sentinel-1"
        )
        backend = s1_catalog.PhoenixSentinel1SLCCatalogBackend(
            collection_source=collection
        )

        return backend

    def create_orbit_catalog_backend(
        self,
    ) -> orbit_catalog.Sentinel1OrbitCatalogBackend:
        """
        Helper function to get orbit aux backend using phoenix with proxima source.
        """
        import phoenix.catalog

        backend = orbit_catalog.PhoenixSentinel1OrbitCatalogBackend(
            collection_source=phoenix.catalog.Client()
            .get_collection("esa-sentinel-1-csar-aux")
            .at("aws:proxima:kayrros-prod-sentinel-aux")
        )
        return backend

    def create_product_provider(self) -> ProductProvider:
        """
        Helper function to get a slc product info provider using phoenix with ASF source + burster.
        """

        def get_product(product_id: str) -> Sentinel1SLCProductInfo:
            return eos.products.sentinel1.product.PhoenixSentinel1ProductInfo.from_product_id(
                product_id
            )

        return get_product


@dataclass(frozen=True)
class CDSEBackendFactory(BackendFactory):
    cdse_access_key_id: str
    cdse_secret_access_key: str
    cdse_username: str
    cdse_password: str

    def create_slc_catalog_backend(self) -> s1_catalog.Sentinel1SLCCatalogBackend:
        backend = s1_catalog.CDSESentinel1SLCCatalogBackend()
        return backend

    def create_orbit_catalog_backend(
        self,
    ) -> orbit_catalog.Sentinel1OrbitCatalogBackend:
        backend = orbit_catalog.CDSESentinel1OrbitCatalogBackend(
            self.cdse_username, self.cdse_password
        )
        return backend

    def create_product_provider(self) -> ProductProvider:
        def get_product(product_id: str) -> Sentinel1SLCProductInfo:
            session = boto3.Session(
                aws_access_key_id=self.cdse_access_key_id,
                aws_secret_access_key=self.cdse_secret_access_key,
            )

            cdse_backend = s1_catalog.CDSESentinel1SLCCatalogBackend()
            return CDSEUnzippedSafeSentinel1SLCProductInfo.from_product_id(
                cdse_backend, session, product_id
            )

        return get_product


def main(
    dstdir: str,
    geometry,
    orbit: int,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    orbit_type: str = "orbpoe",
    polarization: str = "vv",
    calibrate: str = "sigma",
    get_complex: bool = True,
    bistatic: bool = True,
    apd: bool = True,
    intra_pulse: bool = True,
    alt_fm_mismatch: bool = True,
    dem_sampling_ratio: float = 1.0,
    primary_id: int = 0,
    ncpu: int = 16,
    last_n_prods: Optional[int] = None,
    roi_provider: Optional[RoiProvider] = None,
    dem_source: Optional[eos.dem.DEMSource] = None,
    backend_factory: Optional[BackendFactory] = None,
    cache: Cache = eos.cache.no_cache(),
):
    if isinstance(geometry, str):
        geometry = shapely.wkt.loads(geometry)

    # prepare backend factory
    if backend_factory is None:
        # This works for Kayrros users who have configured Phoenix access
        backend_factory = PhoenixBackendFactory()

    # query phoenix
    prod_pol = {"vv": ["SV", "DV"], "vh": ["DV"], "hh": ["SH", "DH"], "hv": ["DH"]}[
        polarization
    ]
    query = s1_catalog.Sentinel1CatalogQuery(
        geometry=geometry,
        start_date=startdate,
        end_date=enddate,
        relative_orbit_number=orbit,
        polarization=prod_pol,  # type: ignore
    )
    logger.info("querying the catalog")
    slc_catalog_backend = backend_factory.create_slc_catalog_backend()
    pids_by_date = s1_catalog.search_slc(
        slc_catalog_backend, query, cache
    ).product_ids_per_date
    logger.info("catalog query done")

    all_product_ids = [
        sorted(pids_by_date[date]) for date in sorted(pids_by_date.keys())
    ]

    product_provider = backend_factory.create_product_provider()
    logger.info("remove weird products...")
    key = (all_product_ids, polarization)
    product_ids = cache.get_or_put(
        key,
        list[list[str]],
        lambda: remove_weird_products(
            product_provider,  # type: ignore
            all_product_ids,
            polarization,
        ),
    )
    assert product_ids is not None
    logger.info("remove weird products DONE")

    if last_n_prods is not None:
        product_ids = product_ids[-last_n_prods:]
    if roi_provider is None:
        roi_provider = GeometryRoiProvider(geometry)
    return run_ts_on_prods(
        dstdir,
        roi_provider,
        product_ids,
        primary_id,
        orbit_type,
        polarization,
        calibrate,
        get_complex,
        bistatic,
        apd,
        intra_pulse,
        alt_fm_mismatch,
        dem_sampling_ratio,
        ncpu,
        dem_source,
        product_provider=product_provider,
        cache=cache,
        orbit_backend=backend_factory.create_orbit_catalog_backend(),
    )


def get_orbits(
    backend: orbit_catalog.Sentinel1OrbitCatalogBackend,
    product_ids: list[list[str]],
    orbit_type,
    cache: Cache,
) -> orbit_catalog.Sentinel1OrbitCatalogResult:
    assert orbit_type in (True, False, None, "orbpoe", "orbres")
    orbit_quality: list[orbit_catalog.OrbitFileType] = {  # type: ignore
        True: orbit_catalog.BestEffort,
        False: [],
        None: [],
        "orbpoe": [orbit_catalog.OrbitFileType.PRECISE],
        "orbres": [orbit_catalog.OrbitFileType.RESTITUTED],
    }[orbit_type]
    query = orbit_catalog.Sentinel1OrbitCatalogQuery(
        product_ids=list(sum(product_ids, [])), quality=orbit_quality
    )
    orbits = orbit_catalog.search(backend, query, cache=cache)
    return orbits


def run_ts_on_prods(
    dstdir: str,
    roi_provider: RoiProvider,
    product_ids: list[list[str]],
    primary_id: int = 0,
    orbit_type: str = "orbpoe",
    polarization: str = "vv",
    calibrate: str = "sigma",
    get_complex: bool = True,
    bistatic: bool = True,
    apd: bool = True,
    intra_pulse: bool = True,
    alt_fm_mismatch: bool = True,
    dem_sampling_ratio: float = 1.0,
    ncpu: int = 16,
    dem_source: Optional[eos.dem.DEMSource] = None,
    *,
    product_provider: ProductProvider,
    cache: Cache,
    orbit_backend: orbit_catalog.Sentinel1OrbitCatalogBackend,
) -> list[Pipeline]:
    os.makedirs(dstdir, exist_ok=True)
    directory_builder = inout.DirectoryBuilder(dstdir)

    # get the ephemerides (orbits)
    logger.info(f"getting orbit data (type={orbit_type})")
    orbits = get_orbits(orbit_backend, product_ids, orbit_type, cache)
    logger.info("getting orbit data DONE")

    if dem_source is None:
        dem_source = eos.dem.get_any_source()
    primary_pipeline = PrimaryPipeline(
        product_ids[primary_id], directory_builder, dem_source
    )
    primary_pipeline.execute(
        product_provider,
        orbits.for_product_id(product_ids[primary_id][0]),
        polarization,
        roi_provider,
        dem_sampling_ratio,
        bistatic,
        apd,
        intra_pulse,
        alt_fm_mismatch,
        calibrate,
        get_complex,
    )

    def process_product(product_id):
        secondary_pipeline = SecondaryPipeline(product_id, directory_builder)
        success = secondary_pipeline.execute(
            product_provider,
            orbits.for_product_id(product_id[0]),
            polarization,
            primary_pipeline.registrator,
            primary_pipeline.deburster,
            calibrate,
            get_complex,
            primary_pipeline.proj_model,
            primary_pipeline.roi,
            primary_pipeline.heights,
        )
        return secondary_pipeline, success

    pipelines_map: Iterator[tuple[SecondaryPipeline, bool]]
    if ncpu == 1:
        pipelines_map = map(process_product, product_ids[1:])
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu)
        pipelines_map = pool.map(process_product, product_ids[1:])

    pipelines: list[Pipeline] = [primary_pipeline]
    for pipeline, success in tqdm.tqdm(pipelines_map, total=len(product_ids) - 1):
        if success:
            pipelines.append(pipeline)

    pipelines = sorted(pipelines, key=lambda x: x.date)

    # save inputs to file
    inout.save_inputs_to_file(
        directory_builder.get_proc_path(),
        ncpu=ncpu,
        orbit_type=orbit_type,
        get_complex=get_complex,
        bistatic=bistatic,
        apd=apd,
        intra_pulse=intra_pulse,
        alt_fm_mismatch=alt_fm_mismatch,
        dem_sampling_ratio=dem_sampling_ratio,
        product_ids=[p.product_ids for p in pipelines],
        primary_id=primary_id,
        roi=primary_pipeline.roi.to_roi(),
    )

    return pipelines


def main_ovl(
    dstdir,
    product_ids_per_date,
    orbit_type="orbpoe",
    polarization="vv",
    calibrate="sigma",
    get_complex=True,
    bistatic=True,
    apd=True,
    intra_pulse=True,
    alt_fm_mismatch=True,
    dem_sampling_ratio=0.1,
    primary_id=0,
    osids_of_interest=None,
    dem_source: Optional[eos.dem.DEMSource] = None,
    product_provider: Optional[ProductProvider] = None,
    cache: Cache = eos.cache.no_cache(),
    orbit_backend: Optional[orbit_catalog.Sentinel1OrbitCatalogBackend] = None,
) -> list[Union[OvlPrimaryPipeline, OvlSecondaryPipeline]]:
    # destination path
    os.makedirs(dstdir, exist_ok=True)
    directory_builder = inout.OvlDirectoryBuilder(dstdir)

    if dem_source is None:
        dem_source = eos.dem.get_any_source()
    primary_pipeline = OvlPrimaryPipeline(
        product_ids_per_date[primary_id], directory_builder, dem_source
    )

    # Necessary to get eos product info objects
    if product_provider is None or orbit_backend is None:
        backend_factory = PhoenixBackendFactory()
        if product_provider is None:
            product_provider = backend_factory.create_product_provider()
        if orbit_backend is None:
            orbit_backend = backend_factory.create_orbit_catalog_backend()

    # Necessary to get the ephemerides (orbits)
    orbits = get_orbits(orbit_backend, product_ids_per_date, orbit_type, cache=cache)

    primary_pipeline.execute(
        product_provider,
        orbits.for_product_id(product_ids_per_date[primary_id][0]),
        dem_sampling_ratio,
        bistatic,
        apd,
        intra_pulse,
        alt_fm_mismatch,
        polarization,
        calibrate,
        get_complex,
        reramp=True,
        swaths=("iw1", "iw2", "iw3"),
        osids_of_interest=osids_of_interest,
    )

    ovl_roi_in_swath_per_bsint = {}
    all_bsint_of_interest: set[Bsint] = set()
    for (
        swath,
        bsint_of_interest,
    ) in primary_pipeline.bsint_of_interest_per_swath.items():
        all_rois_per_bsint = primary_pipeline.ovl_roi_info_per_swath[
            swath
        ].get_swath_rois_per_bsint()
        ovl_roi_in_swath_per_bsint.update(
            {k: v for k, v in all_rois_per_bsint.items() if k in bsint_of_interest}
        )
        all_bsint_of_interest = all_bsint_of_interest.union(bsint_of_interest)

    all_osids_of_interest: set[Osid] = set()
    for (
        swath,
        osids_of_interest,
    ) in primary_pipeline.osids_of_interest_per_swath.items():
        all_osids_of_interest = all_osids_of_interest.union(osids_of_interest)

    directory_reader = inout.OvlDirectoryReader(directory_builder)

    def height_provider(bsint):
        return directory_reader.read_radarcoded_dem(bsint)

    def process_product(product_ids):
        secondary_pipeline = OvlSecondaryPipeline(product_ids, directory_builder)
        success = secondary_pipeline.execute(
            product_provider,
            orbits.for_product_id(product_ids[0]),
            primary_pipeline.registrator_per_swath,
            primary_pipeline.overlap_resamplers_per_swath,
            polarization,
            calibrate,
            get_complex,
            True,
            primary_pipeline.osids_of_interest_per_swath,
            primary_pipeline.swath_models_per_swath,
            height_provider,
            ovl_roi_in_swath_per_bsint,
        )
        return secondary_pipeline, success

    n_products = len(product_ids_per_date)
    secondary_product_ids = (
        product_ids_per_date[:primary_id] + product_ids_per_date[primary_id + 1 :]
    )
    pipelines_map = map(process_product, secondary_product_ids)

    pipelines: list[Union[OvlPrimaryPipeline, OvlSecondaryPipeline]] = [
        primary_pipeline
    ]
    for pipeline, success in tqdm.tqdm(pipelines_map, total=n_products - 1):
        if success:
            pipelines.append(pipeline)

    pipelines = sorted(pipelines, key=lambda x: x.date)

    # save inputs to file
    inout.save_inputs_to_file(
        directory_builder.get_proc_path(),
        product_ids_per_date=[p.product_ids for p in pipelines],
        orbit_type=orbit_type,
        polarization=polarization,
        calibrate=calibrate,
        get_complex=get_complex,
        bistatic=bistatic,
        apd=apd,
        intra_pulse=intra_pulse,
        alt_fm_mismatch=alt_fm_mismatch,
        dem_sampling_ratio=dem_sampling_ratio,
        primary_id=primary_id,
        osids_of_interest=[str(o) for o in all_osids_of_interest],
        bsint_of_interest=[str(b) for b in all_bsint_of_interest],
    )

    return pipelines
