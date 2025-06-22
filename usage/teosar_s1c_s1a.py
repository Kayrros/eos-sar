# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "dem-stitcher",
#     "kayrros-eos-sar[teosar-light,kayrros]",
#     "python-dotenv",
#     "shapely",
#     "fire"
# ]
#
# [tool.uv.sources]
# kayrros-eos-sar = { path = "../", editable = true }
#
# ///

"""
For phoenix backend, you can set the following env vars:
    PHX_USERNAME
    PHX_PASSWORD
    EARTHDATA_USERNAME
    EARTHDATA_PASSWORD
    BURSTERIO_CACHE_PATH # only if you want to cache downloaded s1 bursts
    AWS_PROFILE
AND connect to the Kayrros VPN

!!!!! If you are external to Kayrros:
    replace "kayrros-eos-sar[teosar-light,kayrros]" with "kayrros-eos-sar[teosar-light]"
    and use the CDSE backend

For CDSE backend, you can set the following env vars:
    CDSE_ACCESS_KEY_ID
    CDSE_SECRET_ACCESS_KEY
    CDSE_USERNAME
    CDSE_PASSWORD

the script can be run from the cli.
you can create a virtual env and install packages shown above then run it.
Or you can use the uv tool:

Examples:
    Running timeseries on predefined product ids:
        uv run teosar_s1c_s1a.py run_on_predefined_pids

    Running timeseries with catalog query:
        phoenix backend + VPN
            uv run --index-url "https://pypi.dev-kayrros.ovh/" teosar_s1c_s1a.py run_with_catalog_query --use_cdse=False
        cdse backend
            uv run teosar_s1c_s1a.py run_with_catalog_query --use_cdse=True

    creating ifgs from coregistered stack:
        uv run teosar_s1c_s1a.py generate_ifgs results/catalog_query

    creating ifgs from predefined pids stack:
        uv run teosar_s1c_s1a.py generate_ifgs results/predefined_pids
"""

import datetime
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fire import Fire
from shapely.geometry import shape

from eos.cache import no_cache
from eos.dem import DEMStitcherSource
from eos.products.sentinel1.assembler import Sentinel1Assembler
from eos.products.sentinel1.burst_resamp import Sentinel1BurstResample
from eos.products.sentinel1.los import (
    get_los_squinted_mosaic,
    get_los_ZeroDoppler_mosaic,
)
from eos.sar.roi import Roi
from eos.sar.roi_provider import GeometryRoiProvider
from teosar import inout
from teosar.tsinsar import (
    BackendFactory,
    CDSEBackendFactory,
    PhoenixBackendFactory,
    main,
    run_ts_on_prods,
)
from teosar.utils import Ifg, RoiCuttingInfo, filt_interf, pid2date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# %%
def parse_date_str(date_str: str) -> datetime.datetime:
    """
    example:
    20250331T083159
    """
    return datetime.datetime.strptime(date_str, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )


def run_on_predefined_pids():
    s1c_pids = [
        [
            "S1C_IW_SLC__1SDV_20250412T083159_20250412T083230_001857_00381E_A63D",
        ],
    ]

    s1a_pids = [
        [
            "S1A_IW_SLC__1SSV_20250418T083307_20250418T083335_058808_074969_CCA4",
            "S1A_IW_SLC__1SSV_20250418T083332_20250418T083359_058808_074969_B7B7",
        ],
    ]

    context = {
        "coordinates": [
            [
                [150.6662698269912, -26.84159872059694],
                [150.71043327460126, -26.823179575200072],
                [150.63938772844563, -26.683653898895805],
                [150.5827433065091, -26.70209567703425],
                [150.6662698269912, -26.84159872059694],
            ]
        ],
        "type": "Polygon",
    }

    geometry = shape(context)
    dstdir = "./results/predefined_pids"
    roi_provider = GeometryRoiProvider(geometry)

    # sort by date
    product_ids = sorted(
        s1c_pids + s1a_pids, key=lambda x: parse_date_str(x[0].split("_")[5])
    )
    dem_source = DEMStitcherSource(tiles_cache_dir=Path("/tmp/dem-stitcher"))

    backend_factory = CDSEBackendFactory(
        cdse_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
        cdse_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
        cdse_username=os.environ["CDSE_USERNAME"],
        cdse_password=os.environ["CDSE_PASSWORD"],
    )

    run_ts_on_prods(
        dstdir,
        roi_provider,
        product_ids,
        primary_id=0,
        orbit_type="orbres",
        polarization="vv",
        ncpu=1,
        dem_source=dem_source,
        product_provider=backend_factory.create_product_provider(),
        orbit_backend=backend_factory.create_orbit_catalog_backend(),
        cache=no_cache(),
    )


def run_with_catalog_query(use_cdse: bool = True):
    context = {
        "coordinates": [
            [
                [150.6662698269912, -26.84159872059694],
                [150.71043327460126, -26.823179575200072],
                [150.63938772844563, -26.683653898895805],
                [150.5827433065091, -26.70209567703425],
                [150.6662698269912, -26.84159872059694],
            ]
        ],
        "type": "Polygon",
    }
    if use_cdse:
        logger.info("Using CDSE backend")
        backend_factory: BackendFactory = CDSEBackendFactory(
            cdse_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
            cdse_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
            cdse_username=os.environ["CDSE_USERNAME"],
            cdse_password=os.environ["CDSE_PASSWORD"],
        )
    else:
        logger.info("Using Phoenix backend")
        backend_factory = PhoenixBackendFactory()
    main(
        dstdir="./results/catalog_query",
        geometry=shape(context),
        orbit=111,
        startdate=datetime.datetime(2025, 2, 1),
        enddate=datetime.datetime(2025, 4, 13),
        orbit_type=True,  # BestEffort: Precise, otherwise restituted
        dem_source=DEMStitcherSource(tiles_cache_dir=Path("/tmp/dem-stitcher")),
        backend_factory=backend_factory,
    )


def generate_ifgs(dstdir: str):
    with open(os.path.join(dstdir, "proc.json"), "r") as f:
        proc_dict = json.load(f)

    dir_builder = inout.DirectoryBuilder(dstdir)
    dir_reader = inout.DirectoryReader(dir_builder)
    dates = [pid2date(pids[0]) for pids in proc_dict["product_ids"]]
    outpath = os.path.join(dstdir, "consec_ifg")
    outpath_filt = os.path.join(dstdir, "consec_ifg_filt")
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(outpath_filt, exist_ok=True)
    for i in range(len(dates) - 1):
        j = i + 1
        ifg = Ifg(dir_reader, dates[i], dates[j]).get_init_interf()
        fname = f"{dates[i]}_{dates[j]}.tif"
        inout.save_img(os.path.join(outpath, fname), ifg)

        filtered = filt_interf(ifg, nworkers=12)
        inout.save_img(os.path.join(outpath_filt, fname), filtered)


def generate_los(teosar_dir: str):
    with open(os.path.join(teosar_dir, "proc.json"), "r") as f:
        proc_dict = json.load(f)

    dir_builder = inout.DirectoryBuilder(teosar_dir)
    dates = [pid2date(pids[0]) for pids in proc_dict["product_ids"]]

    meta_primary = inout.json_to_dict(
        dir_builder.get_meta_path(dates[proc_dict["primary_id"]])
    )
    primary_asm = Sentinel1Assembler.from_dict(meta_primary["asm"])
    primary_mosaic_model = primary_asm.get_mosaic_model()

    c, r, w, h = proc_dict["roi"]
    roi = Roi(c, r, w, h)

    cropped_proj = primary_mosaic_model.to_cropped_mosaic(
        roi
    )  # geometrical model on your ROI

    roicuttinginfo = RoiCuttingInfo.from_dict(meta_primary["roi_cutting_info"])
    assert roicuttinginfo.roi == roi
    resamplers_on_roi = {
        k: Sentinel1BurstResample.from_dict(v)
        for k, v in meta_primary["debursting"]["resamplers_on_roi"].items()
    }

    grid_size_col = 50
    grid_size_row = 50
    ellipsoid_alt = 0
    normalized: bool = True
    polynom_degree = 7
    estimate_in_ENU = True

    los_mosaic = get_los_squinted_mosaic(
        roicuttinginfo.write_rois,
        resamplers_on_roi,
        roi.h,
        roi.w,
        cropped_proj,
        grid_size_col=grid_size_col,
        grid_size_row=grid_size_row,
        polynom_degree=polynom_degree,
        ellipsoid_alt=ellipsoid_alt,
        normalized=normalized,
        estimate_in_ENU=estimate_in_ENU,
    )

    los_ZD_mosaic = get_los_ZeroDoppler_mosaic(
        roi.h,
        roi.w,
        cropped_proj,
        grid_size_col=grid_size_col,
        grid_size_row=grid_size_row,
        polynom_degree=polynom_degree,
        ellipsoid_alt=ellipsoid_alt,
        normalized=normalized,
        estimate_in_ENU=estimate_in_ENU,
    )

    if estimate_in_ENU:
        channels = "ENU"
    else:
        channels = "XYZ"

    fname = "los_mosaic_squinted"
    for i, channel in enumerate(channels):
        fname_channel = f"{fname}_{channel}.tif"
        inout.save_img(os.path.join(teosar_dir, fname_channel), los_mosaic[..., i])

    fname = "los_mosaic_ZeroDoppler"
    for i, channel in enumerate(channels):
        fname_channel = f"{fname}_{channel}.tif"
        inout.save_img(os.path.join(teosar_dir, fname_channel), los_ZD_mosaic[..., i])


# %%
if __name__ == "__main__":
    Fire(
        {
            "run_on_predefined_pids": run_on_predefined_pids,
            "run_with_catalog_query": run_with_catalog_query,
            "generate_ifgs": generate_ifgs,
            "generate_los": generate_los,
        }
    )
