import numpy as np
import rasterio

import phoenix.catalog
import eos.sar
import eos.dem
import eos.products.sentinel1 as sentinel1

client = phoenix.catalog.Client()


def _get_gcps(model, roi):
    h, w = roi.get_shape()
    ox, oy = roi.get_origin()
    gcps = []
    for row_ in np.linspace(0, h, num=5).astype(np.int32):
        cols = np.linspace(0, w, num=5).astype(np.int32)
        rows = [row_ for _ in cols]
        for row, col in zip(rows, cols):
            x, y, z, _ = model.localize_without_alt(oy + row, ox + col)
            gcps.append(rasterio.control.GroundControlPoint(row, col, x, y, z))

    return gcps


def main(
    output='example_grd.tif',
    product_id='S1A_IW_GRDH_1SDV_20220621T055930_20220621T055955_043757_053958_F640',
    pol='vv',
    crop_size=500,
    calibration=None,
    do_rtc=False,
    do_ortho=False,
    rtc_after_ortho=False,
):
    product = sentinel1.product.PhoenixSentinel1GRDProductInfo.from_product_id(product_id)

    xml = product.get_xml_annotation(pol)
    meta = sentinel1.metadata.extract_grd_metadata(xml)
    sentinel1.orbits.update_statevectors_using_phoenix(client, product_id, meta)

    orbit = eos.sar.orbit.Orbit(meta["state_vectors"])
    corr = [
        eos.sar.atmospheric_correction.ApdCorrection(orbit),
    ]
    corrector = eos.sar.projection_correction.Corrector(corr)

    proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit, corrector)

    midx, midy = meta['width'] // 2, meta['height'] // 2
    lon, lat, alt = proj_model.localization(midx, midy, 0)
    print(lon, lat, alt)

    r, c, _ = proj_model.projection(lon, lat, alt)
    print(r, c)

    roi = eos.sar.roi.Roi(int(c) - crop_size // 2,
                          int(r) - crop_size // 2,
                          crop_size,
                          crop_size)

    reader = product.get_image_reader(pol)

    if calibration:
        cal_xml = product.get_xml_calibration(pol)
        noise_xml = product.get_xml_noise(pol)
        calibrator = sentinel1.calibration.Sentinel1Calibrator(cal_xml, noise_xml)
        reader = sentinel1.calibration.CalibrationReader(reader, calibrator, method=calibration)

    print('reading raster')
    raster = eos.sar.io.read_window(reader, roi, get_complex=False,
                                    out_dtype=np.float32, boundless=True)
    raster[raster == 0] = np.nan

    profile = dict(
        width=crop_size,
        height=crop_size,
        count=1,
        dtype=raster.dtype,
        nodata=np.nan
    )

    if do_rtc:
        print('computing rtc')
        rtc = eos.sar.rtc.RadiometricTerrainCorrector(proj_model, roi)
        if not rtc_after_ortho:
            raster = rtc.apply(raster)
        sim = rtc.get_simulation()

    print(raster.dtype)
    raster = raster.astype(np.float32)

    if do_ortho:
        print('computing ortho')
        res = 10.0
        orthorectifier = eos.sar.ortho.Orthorectifier.from_roi(proj_model, roi, res)
        raster = orthorectifier.apply(raster, eos.sar.ortho.LanczosInterpolation)
        profile['crs'] = orthorectifier.crs
        profile['transform'] = orthorectifier.transform
        profile['width'] = raster.shape[1]
        profile['height'] = raster.shape[0]

        if do_rtc and rtc_after_ortho:
            sim = orthorectifier.apply(sim, eos.sar.ortho.LanczosInterpolation)
            raster = eos.sar.rtc.normalize(raster, sim)

    with rasterio.open(output, 'w+', **profile) as dst:
        dst.write(raster, 1)

        if 'transform' not in profile:
            print('computing gcps')
            gcps = _get_gcps(proj_model, roi)
            dst.gcps = (gcps, 4979)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
