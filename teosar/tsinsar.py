from functools import partial
import os
import concurrent.futures
from typing import Callable, Literal, Optional, Union
from eos.products.sentinel1.assembler import SLCOrbitProvider
from eos.products.sentinel1.metadata import Sentinel1BurstMetadata
from eos.products.sentinel1.product import Sentinel1SLCProductInfo
import tqdm
import shapely.wkt

import eos.products.sentinel1
import eos.sar
import eos.dem
import eos.products.sentinel1.catalog as s1_catalog

from teosar import inout
from teosar.workflow import PrimaryPipeline, SecondaryPipeline, OvlPrimaryPipeline, OvlSecondaryPipeline
from teosar import utils

ProductProvider = Callable[[str], Sentinel1SLCProductInfo]

OrbitTypes = Optional[Union[Literal["orbpoe", "orbres"], bool]]
"""
None | False: no refined state vectors
True: automatically pick the orbit file (poe then res)
orbpoe | orbres: force the use of a specific orbit file
"""

def get_bsids_for_product(product_provider: ProductProvider,
                          polarization: str,
                          product_id: str
                          ) -> list[str]:
    product = product_provider(product_id)
    xmls = [product.get_xml_annotation(swath=swath, pol=polarization)
            for swath in ("IW1", "IW2", "IW3")]
    bursts = sum([eos.products.sentinel1.metadata.extract_bursts_metadata(xml) for xml in xmls], start=[])
    bsids = [b.bsid for b in bursts]
    return bsids


def get_bsids_for_products(product_provider: ProductProvider,
                           polarization: str,
                           product_ids: list[str]
                           ) -> dict[str, list[str]]:
    # NOTE: this function could be a lot faster if we used the slc_bsid phoenix plugin

    with concurrent.futures.ProcessPoolExecutor() as pool:
        get = partial(get_bsids_for_product, product_provider, polarization)
        all_bsids = pool.map(get, product_ids)

    return {pid: bsids for pid, bsids in zip(product_ids, all_bsids)}


def remove_weird_products(product_provider: ProductProvider,
                          product_ids: list[list[str]],
                          polarization: str
                          ) -> list[list[str]]:
    def pid2datatake(pid: str) -> str:
        idx = len("S1A_IW_SLC__1SDV_20211202T173302_20211202T173329_040833_")
        return pid[idx:idx+6]

    print('getting bsids for products')
    bsids_per_pid = get_bsids_for_products(product_provider, polarization, list(sum(product_ids, [])))
    print('getting bsids for products, DONE')

    by_datatake: dict[str, list[str]] = {}
    for pid in [pid for pids in product_ids for pid in pids]:
        by_datatake.setdefault(pid2datatake(pid), []).append(pid)

    good_datatakes: set[str] = set()
    for datatake, pids in by_datatake.items():
        # check whether some bsids are duplicated in list of products
        bsids = sum((bsids_per_pid[pid] for pid in pids), [])
        if len(bsids) == len(set(bsids)):
            good_datatakes.add(datatake)
        else:
            print(f"warning: {pids} have duplicated burst ids, datake {datatake} removed completely from the timeseries.")

    good_product_ids: list[list[str]] = []
    for pids in product_ids:
        datatake = pid2datatake(pids[0])
        if datatake in good_datatakes:
            good_product_ids.append(pids)

    return good_product_ids


def fetch_orbits(
    pid: str, bursts: list[Sentinel1BurstMetadata], force_type: OrbitTypes
) -> list[Sentinel1BurstMetadata]:
    import phoenix.catalog
    client = phoenix.catalog.Client()
    sv, orig = eos.products.sentinel1.orbits.retrieve_statevectors_using_phoenix(
        client, pid, bursts, force_type=force_type
    )
    return [b.with_new_state_vectors(sv, orig) for b in bursts]


def get_orbit_provider(orbit_type: OrbitTypes) -> Optional[SLCOrbitProvider]:
    orbit_provider = None

    if orbit_type:
        if orbit_type == True:
            orbit_type = None

        process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=10)
        orbit_provider = lambda pid, bursts: process_pool.submit(
            fetch_orbits, pid, bursts, orbit_type
        ).result()

    return orbit_provider


def get_phx_catalog() -> s1_catalog.Sentinel1Catalog:
    import phoenix.catalog
    client = phoenix.catalog.Client()
    collection = client.get_collection("esa-sentinel-1-csar-l1-slc").at(
        "asf:daac:sentinel-1"
    )
    backend = s1_catalog.PhoenixSentinel1CatalogBackend(collection_source=collection)
    catalog = s1_catalog.Sentinel1Catalog(backend=backend)
    return catalog


def main(dstdir, geometry, orbit, startdate, enddate,
         orbit_type="orbpoe",
         polarization="vv",
         calibrate="sigma",
         get_complex=True,
         bistatic=True,
         apd=True,
         intra_pulse=True,
         alt_fm_mismatch=True,
         dem_sampling_ratio=1,
         primary_id=0,
         ncpu=16,
         last_n_prods=None,
         roi_provider: Optional[utils.ProjectionRoiProvider] = None,
         dem_source: Optional[eos.dem.DEMSource] = None,
         product_provider: Optional[ProductProvider] = None,
    ):

    print("Getting products with Phoenix")
    if type(geometry) == str:
        geometry = shapely.wkt.loads(geometry)

    # query phoenix
    prod_pol = {"vv": ["SV", "DV"], "vh": ["DV"], "hh": ["SH", "DH"], "hv": ["DV"]}[polarization]
    query = s1_catalog.Sentinel1CatalogQuery(
        geometry=geometry,
        start_date=startdate,
        end_date=enddate,
        relative_orbit_number=orbit,
        polarization=prod_pol,
    )

    pids_by_date = get_phx_catalog().search_slc(query).product_ids_per_date
    product_ids = [sorted(pids_by_date[date]) for date in sorted(pids_by_date.keys())]
    print('catalog query done')

    if product_provider is None:
        product_provider = eos.products.sentinel1.product.PhoenixSentinel1ProductInfo.from_product_id

    product_ids = remove_weird_products(product_provider, product_ids, polarization)

    if last_n_prods is not None:
        product_ids = product_ids[-last_n_prods:]
    if roi_provider is None:
        roi_provider = utils.GeometryRoiProvider(geometry)
    return run_ts_on_prods(dstdir, roi_provider, product_ids, primary_id,
                           orbit_type, polarization, calibrate, get_complex,
                           bistatic, apd, intra_pulse, alt_fm_mismatch,
                           dem_sampling_ratio, ncpu, dem_source,
                           product_provider=product_provider)

def run_ts_on_prods(dstdir,
                    roi_provider: utils.ProjectionRoiProvider,
                    product_ids,
                    primary_id=0,
                    orbit_type="orbpoe",
                    polarization="vv",
                    calibrate="sigma",
                    get_complex=True,
                    bistatic=True,
                    apd=True,
                    intra_pulse=True,
                    alt_fm_mismatch=True,
                    dem_sampling_ratio=1,
                    ncpu=16,
                    dem_source: Optional[eos.dem.DEMSource] = None,
                    *,
                    product_provider: ProductProvider,
                    ):
    # destination path
    os.makedirs(dstdir, exist_ok=True)
    directory_builder = inout.DirectoryBuilder(dstdir)

    # Necessary to get the ephemerides (orbits)
    assert orbit_type in (True, False, None, "orbpoe", "orbres")
    orbit_provider = get_orbit_provider(orbit_type)

    if dem_source is None:
        dem_source = eos.dem.get_any_source()
    primary_pipeline = PrimaryPipeline(product_ids[primary_id], directory_builder, dem_source)
    primary_pipeline.execute(product_provider, orbit_provider, polarization, roi_provider ,
                dem_sampling_ratio, bistatic, apd, intra_pulse, alt_fm_mismatch,
                calibrate, get_complex)

    # save inputs to file
    inout.save_inputs_to_file(directory_builder.get_proc_path(),
        ncpu=ncpu,
        orbit_type=orbit_type,
        get_complex=get_complex,
        bistatic=bistatic,
        apd=apd,
        intra_pulse=intra_pulse,
        alt_fm_mismatch=alt_fm_mismatch,
        dem_sampling_ratio=dem_sampling_ratio,
        product_ids=product_ids,
        primary_id=primary_id,
        roi=primary_pipeline.roi.to_roi()
        )

    def process_product(product_id):
        secondary_pipeline = SecondaryPipeline(product_id, directory_builder)
        secondary_pipeline.execute(product_provider, orbit_provider, polarization,
                    primary_pipeline.registrator, primary_pipeline.deburster,
                    calibrate, get_complex, primary_pipeline.proj_model, primary_pipeline.roi,
                    primary_pipeline.heights)
        return secondary_pipeline

    if ncpu == 1:
        pipelines_map = map(process_product, product_ids[1:])
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu)
        pipelines_map = pool.map(process_product, product_ids[1:])

    pipelines = [primary_pipeline]
    for pipeline in tqdm.tqdm(pipelines_map, total=len(product_ids)-1):
        pipelines.append(pipeline)

    pipelines = sorted(pipelines, key=lambda x: x.date)
    return pipelines

def main_ovl(dstdir, product_ids_per_date,
             orbit_type="orbpoe",
             polarization="vv",
             calibrate="sigma",
             get_complex=True,
             bistatic=True,
             apd=True,
             intra_pulse=True,
             alt_fm_mismatch=True,
             dem_sampling_ratio=.1,
             primary_id=0,
             osids_of_interest=None,
             dem_source: Optional[eos.dem.DEMSource] = None,
             product_provider: Optional[ProductProvider] = None,
             ):

    # destination path
    os.makedirs(dstdir, exist_ok=True)
    directory_builder = inout.OvlDirectoryBuilder(dstdir)

    if dem_source is None:
        dem_source = eos.dem.get_any_source()
    primary_pipeline = OvlPrimaryPipeline(product_ids_per_date[primary_id], directory_builder, dem_source)

    # Necessary to get eos produt info objects
    if product_provider is None:
        product_provider = eos.products.sentinel1.product.PhoenixSentinel1ProductInfo.from_product_id
    # Necessary to get the ephemerides (orbits)
    assert orbit_type in (True, False, None, "orbpoe", "orbres")
    orbit_provider = get_orbit_provider(orbit_type)

    primary_pipeline.execute(product_provider, orbit_provider, dem_sampling_ratio, bistatic, apd, intra_pulse, alt_fm_mismatch,
                polarization, calibrate, get_complex, reramp=True, swaths=("iw1", "iw2", "iw3"), osids_of_interest=osids_of_interest)

    ovl_roi_in_swath_per_bsint = {}
    all_bsint_of_interest = set()
    for swath, bsint_of_interest in primary_pipeline.bsint_of_interest_per_swath.items():
        all_rois_per_bsint = primary_pipeline.ovl_roi_info_per_swath[swath].get_swath_rois_per_bsint()
        ovl_roi_in_swath_per_bsint.update({k:v for k,v in all_rois_per_bsint.items() if k in bsint_of_interest})
        all_bsint_of_interest = all_bsint_of_interest.union(bsint_of_interest)

    all_osids_of_interest = set()
    for swath, osids_of_interest in primary_pipeline.osids_of_interest_per_swath.items():
        all_osids_of_interest = all_osids_of_interest.union(osids_of_interest)

    # save inputs to file
    inout.save_inputs_to_file(
        directory_builder.get_proc_path(),
        product_ids_per_date=product_ids_per_date,
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
        bsint_of_interest = [str(b) for b in all_bsint_of_interest]
        )

    directory_reader = inout.OvlDirectoryReader(directory_builder)
    def height_provider(bsint):
        return directory_reader.read_radarcoded_dem(bsint)

    def process_product(product_ids):
        secondary_pipeline = OvlSecondaryPipeline(product_ids, directory_builder)
        secondary_pipeline.execute(product_provider, orbit_provider, primary_pipeline.registrator_per_swath,
                        primary_pipeline.overlap_resamplers_per_swath, polarization,
                        calibrate, get_complex, True, primary_pipeline.osids_of_interest_per_swath,
                        primary_pipeline.swath_models_per_swath, height_provider,
                        ovl_roi_in_swath_per_bsint)
        return secondary_pipeline

    n_products = len(product_ids_per_date)
    secondary_product_ids = product_ids_per_date[:primary_id] + product_ids_per_date[primary_id + 1: ]
    pipelines_map = map(process_product, secondary_product_ids)

    pipelines = [primary_pipeline]
    for pipeline in tqdm.tqdm(pipelines_map, total=n_products-1):
        pipelines.append(pipeline)

    pipelines = sorted(pipelines, key=lambda x: x.date)

    return pipelines
