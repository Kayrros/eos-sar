# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fire",
#     "dem-stitcher",
#     "kayrros-eos-sar[teosar-light,kayrros]",
# ]
#
# [tool.uv.sources]
# kayrros-eos-sar = { path = "../", editable = true }
#
# ///
import glob
import logging
import os
from datetime import datetime, timezone
from typing import Sequence

import fire
import tqdm
from shapely.geometry import shape

import eos.sar
from eos.dem import DEM, DEMSource, write_crop_to_file
from eos.products.capella.slc_cropper import CapellaCrop, crop_images
from eos.sar.dem_to_radar import dem_radarcoding
from eos.sar.model import SensorModel
from eos.sar.roi import Roi
from eos.sar.roi_provider import GeometryRoiProvider, RoiProvider
from teosar.inout import (
    DirectoryBuilder,
    DirectoryReader,
    dict_to_json,
    json_to_dict,
    save_img,
    save_inputs_to_file,
)
from teosar.utils import Ifg, estimate_corrections

logging.basicConfig(level=logging.INFO)


# %%
def get_date_str(capella_pid: str) -> str:
    start = 22  # len("CAPELLA_C02_SM_SLC_HH_")
    end = 30  # len("CAPELLA_C02_SM_SLC_HH_20220319")
    date_str = capella_pid[start:end]
    return date_str


def date_str_to_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)


def get_datetime(capella_pid: str) -> datetime:
    return date_str_to_datetime(get_date_str(capella_pid))


def crop_and_store(
    raster_paths: list[str],
    primary_id: int,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dir_builder: DirectoryBuilder,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex: bool = True,
    use_apd: bool = True,
    refine_regist: bool = True,
    calibrate: bool = True,
) -> tuple[list[CapellaCrop], DEM]:
    (crops, dem) = crop_images(
        raster_paths,
        primary_id,
        roi_provider,
        dem_source,
        dem_sampling_ratio,
        get_complex=get_complex,
        use_apd=use_apd,
        refine_regist=refine_regist,
        calibrate=calibrate,
    )

    # save the dem on the disk
    geo_dem_path = dir_builder.get_geo_dem_path()
    write_crop_to_file(dem.array, dem.transform, dem.crs, geo_dem_path)
    # save all arrays
    for crop in crops:
        save_img(
            dir_builder.get_img_path(get_date_str(crop.product_id)),
            crop.array,
        )

    log_per_pid = {}
    # log primary processing
    for crop in crops:
        pid = crop.product_id
        log_per_pid[pid] = {
            # TODO add to_dict() method for CapellaMetadata
            # "meta": crop.meta.to_dict(),
            "roi": crop.roi.to_roi(),
            "resampling_matrix": crop.resampling_matrix.tolist(),
            "translation": crop.translation,
        }

    # save logs
    for pid, log in log_per_pid.items():
        dict_to_json(log, dir_builder.get_meta_path(get_date_str(pid)))

    return crops, dem


def compute_simulations_and_store(
    primary_model: SensorModel,
    dem: DEM,
    roi: Roi,
    secondary_models: Sequence[SensorModel],
    secondary_dates: list[str],
    dir_builder: DirectoryBuilder,
):
    print("Radarcoding DEM")
    # radarcode the dem
    heights = dem_radarcoding(dem, primary_model, roi, margin=100)
    # write
    save_img(dir_builder.get_radar_dem_path(), heights)

    print("Computing Simulations")
    # Simulate phases and write
    for secondary_model, date in zip(secondary_models, secondary_dates):
        flat_earth_phase, topo_phase = estimate_corrections(
            primary_model, roi, secondary_model, heights
        )
        save_img(dir_builder.get_flat_path(date), flat_earth_phase)
        save_img(dir_builder.get_topo_path(date), topo_phase)


def compute_ifgs_coher_consec_and_store(
    dstdir: str, filter_size: tuple[int, int] = (3, 3)
):
    dir_builder = DirectoryBuilder(dstdir)
    # Here we can pick up the computation from info saved on disk
    dir_reader = DirectoryReader(dir_builder)
    proc_dict = json_to_dict(dir_builder.get_proc_path())
    product_ids = proc_dict["product_ids"]
    dates = [get_date_str(pid) for pid in product_ids]
    print("Computing consecutive ifgs")
    # here we do consec ifgs
    all_ifg_dates = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]

    suffix = f"{filter_size[0]}_{filter_size[1]}"
    keys = [
        "ifgs_consec_init",
        "ifgs_consec_flat",
        "ifgs_consec_topocorr",
        f"coher_consec_{suffix}",
    ]
    out_ifgs = {key: os.path.join(dstdir, key) for key in keys}
    for key in keys:
        os.makedirs(out_ifgs[key], exist_ok=True)
    for i in tqdm.trange(len(all_ifg_dates)):
        ifg_dates = all_ifg_dates[i]
        ifg = Ifg(dir_reader, *ifg_dates)
        init_ifg = ifg.get_init_interf()
        flattened = ifg.get_flattened()
        topo_corrected = ifg.get_topo_corrected()
        _, coherence = ifg.multilook(
            topo_corrected, filter_size, compute_coherence=True, undersample=False
        )
        fname = f"{ifg_dates[0]}_{ifg_dates[1]}.tif"
        # save ifgs
        save_img(os.path.join(out_ifgs[keys[0]], fname), init_ifg)
        save_img(os.path.join(out_ifgs[keys[1]], fname), flattened)
        save_img(os.path.join(out_ifgs[keys[2]], fname), topo_corrected)
        # save phase mlooked ifgs
        save_img(os.path.join(out_ifgs[keys[3]], fname), coherence)


def main(orb_str: str, use_apd=True, refine_regist=True, calibrate=True):
    """
    This main function is specific to the Piton de la Fournaise InSAR stack but can be adapted.
    It is not optimal in any way in the way the arrays are kept in memory or stored on the disk.
    It is only meant to give an example of a workflow that works and that can be adapted.

    Assumes you have the directory capella_insar next to the script,
    and that the f"Piton_Fournaise_{orb_str}_eSM" are in the capella_insar directory.
    """
    assert orb_str in ["RO27", "RO37", "RO13"]
    experiment_path = "capella_insar/"
    data_path = os.path.join(experiment_path, f"Piton_Fournaise_{orb_str}_eSM")

    tif_files = glob.glob(os.path.join(data_path, "*/CAPELLA_*_SLC_*.tif"))
    tif_files = sorted(tif_files, key=lambda x: get_datetime(os.path.basename(x)))

    # Here the roi was chosen in vpv by opening the primary image
    # but since the goal is to have a RoiProvider, one can start
    # from a geometry with GeometryRoiProvider or from a lon, lat with CentroidRoiProvider
    context = {
        "type": "Polygon",
        "coordinates": [
            [
                [55.710055847821984, -21.228751422821833],
                [55.696410042496844, -21.246291437162835],
                [55.720180154998701, -21.261573139634052],
                [55.733605866689579, -21.244034944105934],
                [55.710055847821984, -21.228751422821833],
            ]
        ],
    }

    geometry = shape(context)
    roi_provider = GeometryRoiProvider(geometry)

    dem_source = eos.dem.DEMStitcherSource()

    result_path = os.path.join(
        experiment_path,
        "results",
        f"{orb_str}_useapd{use_apd}_refineregist{refine_regist}_calibrate{calibrate}",
    )
    os.makedirs(result_path, exist_ok=True)
    dir_builder = DirectoryBuilder(result_path)
    primary_id = 0
    dem_sampling_ratio = 0.2

    crops, dem = crop_and_store(
        tif_files,
        primary_id,
        roi_provider,
        dem_source,
        dir_builder,
        dem_sampling_ratio=dem_sampling_ratio,
        use_apd=use_apd,
        refine_regist=refine_regist,
        calibrate=calibrate,
    )

    product_ids = [c.product_id for c in crops]
    roi = crops[primary_id].roi

    # save inputs to file
    save_inputs_to_file(
        dir_builder.get_proc_path(),
        dem_sampling_ratio=dem_sampling_ratio,
        product_ids=product_ids,
        primary_id=primary_id,
        roi=roi.to_roi(),
    )

    secondary_dates = [
        get_date_str(product_ids[i]) for i in range(len(product_ids)) if i != primary_id
    ]
    secondary_models = [
        crops[i].model for i in range(len(product_ids)) if i != primary_id
    ]
    primary_model = crops[primary_id].model
    # Computes simulations and saves them on the disk
    compute_simulations_and_store(
        primary_model, dem, roi, secondary_models, secondary_dates, dir_builder
    )

    filter_size = (3, 3)
    # Computes the consecutive ifgs and coherence maps and save on disk
    # here result_path is taken as input to prove that we can do ifgs from results stored in directory only
    compute_ifgs_coher_consec_and_store(result_path, filter_size)


# %%
if __name__ == "__main__":
    fire.Fire(main)
