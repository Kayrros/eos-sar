import os
from typing import Optional

import numpy as np
import phoenix.catalog
import rasterio
import rasterio.control

import eos.dem
import eos.products.sentinel1 as sentinel1
import eos.sar
from eos.products.sentinel1 import orbit_catalog
from eos.sar.model import SensorModel
from eos.sar.roi import Roi

client = phoenix.catalog.Client()


def _get_gcps(
    model: SensorModel, roi: Roi, dem: eos.dem.DEM
) -> list[rasterio.control.GroundControlPoint]:
    h, w = roi.get_shape()
    ox, oy = roi.get_origin()
    gcps = []
    for row_ in np.linspace(0, h, num=5).astype(np.int32):
        cols = np.linspace(0, w, num=5).astype(np.int32)
        rows = [row_ for _ in cols]
        for row, col in zip(rows, cols):
            x, y, z, _ = model.localize_without_alt(oy + row, ox + col, dem=dem)
            gcps.append(rasterio.control.GroundControlPoint(row, col, x, y, z))

    return gcps


def get_product_info(product_id: str) -> sentinel1.product.Sentinel1GRDProductInfo:
    if "CDSE_ACCESS_KEY_ID" in os.environ:
        import boto3

        cdse_session = boto3.Session(
            aws_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
        )
        cdse_backend = sentinel1.catalog.CDSESentinel1GRDCatalogBackend()
        return (
            sentinel1.product.CDSEUnzippedSafeSentinel1GRDProductInfo.from_product_id(
                cdse_backend,
                cdse_session,
                product_id,
            )
        )
    else:
        return sentinel1.product.PhoenixSentinel1GRDProductInfo.from_product_id(
            product_id
        )


def main(
    output: str = "example_grd.tif",
    product_id: str = "S1A_IW_GRDH_1SDV_20220621T055930_20220621T055955_043757_053958_F640",
    pol: str = "vv",
    crop_size: int = 500,
    calibration: Optional[str] = None,
    do_rtc: bool = False,
    do_ortho: bool = False,
    rtc_after_ortho: bool = False,
) -> None:
    product = get_product_info(product_id)
    dem_source = eos.dem.get_any_source()

    import phoenix.catalog

    query = orbit_catalog.Sentinel1OrbitCatalogQuery(
        product_ids=[product_id], quality=orbit_catalog.OnlyBest
    )
    backend = orbit_catalog.PhoenixSentinel1OrbitCatalogBackend(
        collection_source=phoenix.catalog.Client()
        .get_collection("esa-sentinel-1-csar-aux")
        .at("aws:proxima:kayrros-prod-sentinel-aux")
    )
    statevectors = orbit_catalog.search(backend, query).single()
    assert statevectors is not None

    xml = product.get_xml_annotation(pol)
    meta = sentinel1.metadata.extract_grd_metadata(xml)
    meta = meta.with_new_state_vectors(statevectors, "")

    orbit = eos.sar.orbit.Orbit(meta.state_vectors)
    corr = [
        eos.sar.atmospheric_correction.ApdCorrection(orbit),
    ]
    corrector = eos.sar.projection_correction.Corrector(corr)

    proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit, corrector)

    midx, midy = meta.width // 2, meta.height // 2
    lon, lat, alt = proj_model.localization(midx, midy, 0)
    print(lon, lat, alt)

    r, c, _ = proj_model.projection(lon, lat, alt)
    print(r, c)

    roi = eos.sar.roi.Roi(
        int(c) - crop_size // 2, int(r) - crop_size // 2, crop_size, crop_size
    )

    dem = proj_model.fetch_dem(dem_source, roi)
    reader = product.get_image_reader(pol)

    if calibration:
        cal_xml = product.get_xml_calibration(pol)
        noise_xml = product.get_xml_noise(pol)
        calibrator = sentinel1.calibration.Sentinel1Calibrator(cal_xml, noise_xml)
        reader = sentinel1.calibration.CalibrationReader(
            reader, calibrator, method=calibration
        )

    raster = eos.sar.io.read_window(
        reader, roi, get_complex=False, out_dtype=np.float32
    )
    mask = sentinel1.border_noise_grd.compute_border_mask(raster)
    raster = sentinel1.border_noise_grd.apply_border_mask(raster, mask)

    profile = dict(
        width=crop_size, height=crop_size, count=1, dtype=raster.dtype, nodata=np.nan
    )

    if do_rtc:
        print("computing rtc")
        rtc = eos.sar.rtc.RadiometricTerrainCorrector(proj_model, dem, roi)
        if not rtc_after_ortho:
            raster = rtc.apply(raster)
        sim = rtc.get_simulation()

    print(raster.dtype)
    raster = raster.astype(np.float32)

    if do_ortho:
        print("computing ortho")
        res = 10.0
        orthorectifier = eos.sar.ortho.Orthorectifier.from_roi(
            proj_model, roi, res, dem=dem
        )
        raster = orthorectifier.apply(raster, eos.sar.ortho.LanczosInterpolation)
        profile["crs"] = orthorectifier.crs
        profile["transform"] = orthorectifier.transform
        profile["width"] = raster.shape[1]
        profile["height"] = raster.shape[0]

        if do_rtc and rtc_after_ortho:
            sim = orthorectifier.apply(sim, eos.sar.ortho.LanczosInterpolation)
            raster = eos.sar.rtc.normalize(raster, sim)

    with rasterio.open(output, "w+", **profile) as dst:
        dst.write(raster, 1)

        if "transform" not in profile:
            print("computing gcps")
            gcps = _get_gcps(proj_model, roi, dem)
            dst.gcps = (gcps, 4979)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
