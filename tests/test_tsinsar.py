import datetime
import json
import os

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
    CDSEBackendFactory,
    main,
    run_ts_on_prods,
)
from teosar.utils import Ifg, RoiCuttingInfo, filt_interf, pid2date


def parse_date_str(date_str: str) -> datetime.datetime:
    """
    example:
    20250331T083159
    """
    return datetime.datetime.strptime(date_str, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
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


def test_run_on_predefined_pids(cdse_auth, cdse_s3_session, tmp_path):
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
    dstdir = f"{tmp_path}/predefined_pids"
    roi_provider = GeometryRoiProvider(geometry)

    # sort by date
    product_ids = sorted(
        s1c_pids + s1a_pids, key=lambda x: parse_date_str(x[0].split("_")[5])
    )

    # The workflow downloads the dem twice, so it is better to cache the tiles
    dem_source = DEMStitcherSource(tiles_cache_dir=tmp_path / "dem-stitcher")

    s3_creds = cdse_s3_session.get_credentials()
    username, password = cdse_auth
    backend_factory = CDSEBackendFactory(
        cdse_access_key_id=s3_creds.access_key,
        cdse_secret_access_key=s3_creds.secret_key,
        cdse_username=username,
        cdse_password=password,
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

    generate_ifgs(dstdir)
    generate_los(dstdir)


def test_run_with_catalog_query(cdse_auth, cdse_s3_session, tmp_path):
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

    s3_creds = cdse_s3_session.get_credentials()
    username, password = cdse_auth
    backend_factory = CDSEBackendFactory(
        cdse_access_key_id=s3_creds.access_key,
        cdse_secret_access_key=s3_creds.secret_key,
        cdse_username=username,
        cdse_password=password,
    )

    dstdir = f"{tmp_path}/catalog_query"

    main(
        dstdir=dstdir,
        geometry=shape(context),
        orbit=111,
        startdate=datetime.datetime(2025, 2, 1),
        enddate=datetime.datetime(2025, 3, 2),
        orbit_type=True,  # BestEffort: Precise, otherwise restituted
        ncpu=1,
        dem_source=DEMStitcherSource(tiles_cache_dir=tmp_path / "dem-stitcher"),
        backend_factory=backend_factory,
    )

    generate_ifgs(dstdir)
    generate_los(dstdir)
