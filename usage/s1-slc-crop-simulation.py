"""
needs CDSE_ACCESS_KEY_ID, CDSE_SECRET_ACCESS_KEY
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import boto3
import fire
import numpy as np
from eos.sar.simulator import SARSimulator
from rasterio.warp import Resampling

import eos.dem
import eos.products.sentinel1 as s1
from eos.sar.roi_provider import CentroidRoiProvider, RoiProvider

# %%
ORBIT_PASS = Literal["Ascending", "Descending"]


@dataclass(frozen=True)
class AOIParams:
    product_ids: dict[ORBIT_PASS, str]
    roi_provider: RoiProvider


AOIS: dict[str, AOIParams] = {
    "turkey": AOIParams(
        product_ids={
            "Descending": "S1B_IW_SLC__1SDV_20200329T031656_20200329T031723_020901_027A2C_9733",
            "Ascending": "S1B_IW_SLC__1SDV_20200323T150904_20200323T150932_020821_0277B3_D80F",
        },
        roi_provider=CentroidRoiProvider((41.42898, 38.6376), 500, 500),
    ),
    "atacama_s1b": AOIParams(
        product_ids={
            "Ascending": "S1B_IW_SLC__1SDV_20200426T232032_20200426T232059_021321_02878A_7EB7",
            "Descending": "S1B_IW_SLC__1SDV_20200429T100214_20200429T100241_021357_0288A2_069C",
        },
        roi_provider=CentroidRoiProvider((-70.1614, -28.6496), 500, 500),
    ),
    "atacama_s1c": AOIParams(
        product_ids={
            "Ascending": "S1C_IW_SLC__1SDV_20260206T232018_20260206T232046_006241_00C892_8116",
            "Descending": "S1C_IW_SLC__1SDV_20260116T100153_20260116T100220_005927_00BE36_53DE",
        },
        roi_provider=CentroidRoiProvider((-70.1614, -28.6496), 500, 500),
    ),
    "santa_barbara": AOIParams(
        product_ids={
            "Ascending": "S1A_IW_SLC__1SDV_20260202T015836_20260202T015904_063034_07E8FC_F518",
            "Descending": "S1A_IW_SLC__1SDV_20260202T140031_20260202T140058_063041_07E93F_D950",
        },
        roi_provider=CentroidRoiProvider((-119.932, 34.755), 500, 500),
    ),
    "ethiopia": AOIParams(
        product_ids={
            "Ascending": "S1A_IW_SLC__1SDV_20260129T153434_20260129T153501_062984_07E710_D17F",
            "Descending": "S1A_IW_SLC__1SDV_20260129T030902_20260129T030929_062976_07E6CF_BF24",
        },
        roi_provider=CentroidRoiProvider((39.5391, 9.8052), 500, 500),
    ),
}


def run(
    out: str,
    aoi: str,
    orbit_pass: ORBIT_PASS = "Ascending",
):
    product_id = AOIS[aoi].product_ids[orbit_pass]
    roi_provider = AOIS[aoi].roi_provider

    swaths = ("iw1", "iw2", "iw3")
    polarization = "vv"
    get_complex = False
    reramp = False
    calibration = "beta"

    cdse_backend = s1.catalog.CDSESentinel1SLCCatalogBackend()
    session = boto3.Session(
        aws_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
    )
    product = s1.product.CDSEUnzippedSafeSentinel1SLCProductInfo.from_product_id(
        cdse_backend, session, product_id
    )

    backend = s1.orbit_catalog.CDSESentinel1OrbitCatalogBackend(
        os.environ["CDSE_USERNAME"], os.environ["CDSE_PASSWORD"]
    )

    query = s1.orbit_catalog.Sentinel1OrbitCatalogQuery(
        product_ids=[product.product_id], quality=s1.orbit_catalog.OnlyBest
    )
    statevectors = s1.orbit_catalog.search(backend, query)

    sv = statevectors.for_product_id(product.product_id)

    # Take assembler for products of primary date
    asm = s1.assembler.Sentinel1Assembler.from_products(
        [product], polarization, statevectors=sv, swaths=swaths
    )
    any_bmeta = asm.get_single_burst_meta(next(iter(asm.bsids)))
    print("orbit direction: ", any_bmeta.orbit_pass)
    assert any_bmeta.orbit_pass == orbit_pass

    # get dem interface
    dem_source = eos.dem.DEMStitcherSource(tiles_cache_dir=Path("/tmp/dem-stitcher"))

    # get projection model
    proj_model = asm.get_mosaic_model()

    roi_in_primary, _, _ = roi_provider.get_roi(proj_model, dem_source)

    cropper = asm.get_cropper(roi_in_primary)

    crop, resamplers = cropper.crop(
        [product],
        pol=polarization,
        statevectors=sv,
        get_complex=get_complex,
        dem_source=dem_source,
        reramp=reramp,
        calibration=calibration,
    )

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "crop.npy"), crop)

    dem = proj_model.fetch_dem(dem_source, roi_in_primary)

    simulator = SARSimulator(proj_model, dem, dem_resampling=Resampling.cubic_spline)
    simulation = simulator.simulate(roi_in_primary)
    np.save(os.path.join(out, "simulation-dem-stitcher.npy"), simulation)


if __name__ == "__main__":
    fire.Fire(run)
