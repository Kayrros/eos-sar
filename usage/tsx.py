"""
uv run --extra teosar-light usage/tsx.py

TSX can be processed without the teosar-light extra, but this extra is used to
make it easier to save the results into the directory structure.
"""

import glob
import os
from typing import Sequence

import tqdm
from shapely import from_wkt

from eos.dem import DEM, DEMSource, DEMStitcherSource, write_crop_to_file
from eos.products.terrasarx.cropper import TSXCrop, crop_images, pid_from_xml_path
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


def date_from_pid(tsx_pid: str) -> str:
    date = tsx_pid.split("_")[-2][:8]
    return date


def crop_and_store(
    xml_metadata_files: list[str],
    raster_paths: list[str],
    primary_id: int,
    roi_provider: RoiProvider,
    dem_source: DEMSource,
    dir_builder: DirectoryBuilder,
    dem_sampling_ratio: float = 0.3,
    *,
    get_complex=True,
) -> tuple[list[TSXCrop], DEM]:
    (crops, dem) = crop_images(
        xml_metadata_files,
        raster_paths,
        primary_id,
        roi_provider,
        dem_source,
        dem_sampling_ratio,
        get_complex=get_complex,
    )

    # save the dem on the disk
    geo_dem_path = dir_builder.get_geo_dem_path()
    write_crop_to_file(dem.array, dem.transform, dem.crs, geo_dem_path)
    # save all arrays
    for crop in crops:
        save_img(
            dir_builder.get_img_path(date_from_pid(crop.product_id)),
            crop.array,
        )

    log_per_pid = {}
    # log primary processing
    for crop in crops:
        pid = crop.product_id
        log_per_pid[pid] = {
            "meta": crop.meta.to_dict(),
            "roi": crop.roi.to_roi(),
            "resampling_matrix": crop.resampling_matrix.tolist(),
        }
    # save logs
    for pid, log in log_per_pid.items():
        dict_to_json(log, dir_builder.get_meta_path(date_from_pid(pid)))

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
    dates = [date_from_pid(pid) for pid in product_ids]
    print("Computing consecutive ifgs")
    # here we do consec ifgs
    all_ifg_dates = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]

    suffix = f"{filter_size[0]}_{filter_size[1]}"
    keys = ["ifgs_consec", f"coher_consec_{suffix}"]
    out_ifgs = {key: os.path.join(dstdir, key) for key in keys}
    for key in keys:
        os.makedirs(out_ifgs[key], exist_ok=True)
    for i in tqdm.trange(len(all_ifg_dates)):
        ifg_dates = all_ifg_dates[i]
        ifg = Ifg(dir_reader, *ifg_dates)
        topo_corrected = ifg.get_topo_corrected()
        _, coherence = ifg.multilook(
            topo_corrected, filter_size, compute_coherence=True, undersample=False
        )
        fname = f"{ifg_dates[0]}_{ifg_dates[1]}.tif"
        # save ifgs
        save_img(os.path.join(out_ifgs[keys[0]], fname), topo_corrected)
        # save phase mlooked ifgs
        save_img(os.path.join(out_ifgs[keys[1]], fname), coherence)


if __name__ == "__main__":
    datapath = "./fujairah"
    dstdir = "./tsx_fujairah"
    os.makedirs(dstdir, exist_ok=True)
    xml_metadata_files = sorted(
        glob.glob(os.path.join(datapath, "dims*/*/*/T*.xml")),
        key=lambda x: date_from_pid(pid_from_xml_path(x)),
    )

    # Take this from SNAP InSAR overview which automatically selects best primary
    # TODO do this ourself based on teosar/pairs.py script
    primary_date = "20200619"
    for i, xml_path in enumerate(xml_metadata_files):
        if primary_date in xml_path:
            primary_id = i
            break

    def get_raster_path(xml_path):
        return glob.glob(
            os.path.join(os.path.dirname(xml_path), "IMAGEDATA", "IMAGE*.cos")
        )[0]

    raster_paths = [get_raster_path(xml_path) for xml_path in xml_metadata_files]

    geometry_wkt = (
        "POLYGON ((56.332794189453125 25.22136878967285, 56.37843322753906 "
        "25.22136878967285, 56.37843322753906 25.159748077392578, "
        "56.332794189453125 25.159748077392578, 56.332794189453125 "
        "25.22136878967285, 56.332794189453125 25.22136878967285))"
    )
    geometry = from_wkt(geometry_wkt)
    roi_provider = GeometryRoiProvider(
        geometry,
        # If you want to impose a min_width or height for small aois,
        # set this to a bigger value,  here not necessary
        min_width=0,
        min_height=0,
    )

    dem_source = DEMStitcherSource()

    dem_sampling_ratio = 0.3
    filter_size = (3, 3)

    dir_builder = DirectoryBuilder(dstdir)

    crops, dem = crop_and_store(
        xml_metadata_files,
        raster_paths,
        primary_id,
        roi_provider,
        dem_source,
        dir_builder,
        dem_sampling_ratio=dem_sampling_ratio,
        get_complex=True,
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
        date_from_pid(product_ids[i])
        for i in range(len(product_ids))
        if i != primary_id
    ]
    secondary_models = [
        crops[i].model for i in range(len(product_ids)) if i != primary_id
    ]
    primary_model = crops[primary_id].model
    # Computes simulations and saves them on the disk
    compute_simulations_and_store(
        primary_model, dem, roi, secondary_models, secondary_dates, dir_builder
    )

    # Computes the consecutive ifgs and coherence maps and save on disk
    # here dstdir is taken as input to prove that we can do ifgs from results stored in directory only
    compute_ifgs_coher_consec_and_store(dstdir, filter_size)
