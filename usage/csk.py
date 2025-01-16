import json
import os
from typing import Any

import numpy as np
import phoenix as phx
import shapely.geometry
from rasterio import rasterio

import eos.dem
import eos.products.cosmoskymed as cs
import eos.sar
from eos.sar import atmospheric_correction, projection_correction
from eos.sar.orbit import Orbit
from eos.sar.regist import phase_correlation_on_amplitude
from eos.sar.roi import Roi
from teosar.utils import get_gcps_localization


def save_raster_to_tif(
    out: str,
    raster,
    profile: dict[str, Any],
    *,
    gcps,
) -> None:
    profile = profile.copy()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = raster.dtype
    profile["width"] = raster.shape[-1]
    profile["height"] = raster.shape[-2]
    del profile["tiled"]
    del profile["blockxsize"]
    del profile["blockysize"]

    with rasterio.open(out, "w+", **profile) as dst:
        dst.write(raster, 1)
        dst.gcps = (gcps, 4979)


def process(
    path: str,
    out: str,
    lon: float,
    lat: float,
    W: int,
    H: int,
):
    # dem_source = eos.dem.SRTM4Source()
    dem_source = eos.dem.DEMStitcherSource()

    meta = cs.parse_cosmoskymed_metadata(path)
    json.dump(meta.__dict__, open(out + ".json", "w"), default=str, indent=2)

    if False:
        orbit = Orbit(meta.state_vectors, degree=11)
        corr = [
            atmospheric_correction.ApdCorrection(orbit),
        ]
        corrector = projection_correction.Corrector(corr)
    else:
        corrector = projection_correction.Corrector([])

    model = cs.CosmoSkyMedModel.from_metadata(
        meta, orbit_degree=11, corrector=corrector
    )

    dem = model.fetch_dem(dem_source)
    alt = dem.elevation(lon, lat)
    assert isinstance(alt, float)
    y, x, _ = model.projection(lon, lat, alt)

    roi = Roi(int(x) - W, int(y) - H, W * 2, H * 2)
    print(roi)

    open(out + ".geojson", "w").write(
        json.dumps(
            {
                "type": "Feature",
                "properties": {},
                "geometry": shapely.geometry.Polygon(
                    model.get_approx_geom(roi, dem=dem)[0]
                ).__geo_interface__,
            }
        )
    )

    reader = rasterio.open(meta.get_gdal_image_path(path))
    profile = reader.profile

    col, row, w, h = roi.to_roi()
    raster = reader.read(window=((row, row + h), (col, col + w)), boundless=True)
    raster = (raster[0] + 1j * raster[1]).astype(np.complex64)

    gcps = get_gcps_localization(model, dem, roi)
    save_raster_to_tif(out, np.abs(raster), profile, gcps=gcps)

    return model, raster, roi, dem, meta


def download_maybe(path: str, id: str) -> None:
    if not os.path.exists(path):
        it = collection.get_item(id)
        it.assets.download_to_fileobj(open(path, "wb"), "DATA")


def simulate_phases(model1, model2, roi1, dem):
    topo = eos.sar.geom_phase.TopoCorrection(model1, [model2], grid_size=50, degree=7)

    margin = 60
    approx_geom, _, _ = model1.get_approx_geom(roi1, dem=dem, margin=margin)
    heights = eos.sar.dem_to_radar.dem_radarcoding(
        dem, model1, roi=roi1, approx_geometry=approx_geom, margin=margin
    )

    topo_phase = topo.topo_phase_image(heights, primary_roi=roi1)
    flat_earth = topo.flat_earth_image(roi1)

    return topo_phase, flat_earth


collection = (
    phx.catalog.Client()
    .get_collection("asi-cosmo-skymed-csk-stripmap-l1a")
    .at("aws:proxima:kayrros-prod-cosmo-skymed")
)

id1 = "CSKS1_SCS_B_HI_13_HH_RD_FF_20190528001000_20190528001007"
id2 = "CSKS4_SCS_B_HI_13_HH_RD_FF_20190604001000_20190604001007"

lon = -96.6040765
lat = 35.9888679
W = 1700
H = 1700


def main(outdir="/tmp/t/", debug: bool = False):
    os.makedirs(outdir, exist_ok=True)

    # download and read the rasters
    path = f"{outdir}/{id1}.h5"
    download_maybe(path, id1)
    m1, img1, r1, d1, meta1 = process(path, f"{outdir}/img1.tif", lon, lat, W, H)

    path = f"{outdir}/{id2}.h5"
    download_maybe(path, id2)
    m2, img2, r2, _, meta2 = process(path, f"{outdir}/img2.tif", lon, lat, W, H)

    # compute deramping
    der1 = meta1.deramping_phases(r1)
    der2 = meta2.deramping_phases(r2)
    img1_der = img1 * np.exp(1j * der1)
    img2_der = img2 * np.exp(1j * der2)

    if debug:
        np.save(f"{outdir}/img1.npy", img1)
        np.save(f"{outdir}/img2.npy", img2)
        np.save(f"{outdir}/img1_der.npy", img1_der)
        np.save(f"{outdir}/img2_der.npy", img2_der)
        np.save(
            f"{outdir}/fft_1.npy",
            np.ascontiguousarray(np.fft.fftshift(np.fft.fft2(img1))),
        )
        np.save(
            f"{outdir}/fft_2.npy",
            np.ascontiguousarray(np.fft.fftshift(np.fft.fft2(img2))),
        )
        np.save(
            f"{outdir}/fft_1_der.npy",
            np.ascontiguousarray(np.fft.fftshift(np.fft.fft2(img1_der))),
        )
        np.save(
            f"{outdir}/fft_2_der.npy",
            np.ascontiguousarray(np.fft.fftshift(np.fft.fft2(img2_der))),
        )

    # TODO: oversample before abs
    tcol, trow = phase_correlation_on_amplitude(np.abs(img1), np.abs(img2))
    print(tcol, trow)

    # warp deramped img2 to img1
    A = eos.sar.regist.translation_matrix(-tcol, -trow)
    dst_shape = r2.get_shape()
    img2_der_regist = eos.sar.regist.apply_affine(img2_der, A, dst_shape)
    img2_der_regist = np.nan_to_num(img2_der_regist)
    if debug:
        np.save(f"{outdir}/img2_der_regist.npy", img2_der_regist)

    # warp the deramping phases and reramp the registered im2
    rer1 = eos.sar.regist.apply_affine(der2, A, dst_shape)
    img2_der_regist_rer = img2_der_regist * np.exp(1j * -rer1)
    if debug:
        np.save(f"{outdir}/img2_der_regist_rer.npy", img2_der_regist_rer)

    # simulate topo and flat earth phases
    topo_phase, flat_earth = simulate_phases(m1, m2, r1, d1)
    if debug:
        np.save(f"{outdir}/topo", topo_phase)
        np.save(f"{outdir}/flat", flat_earth)

    # compute the interferogram
    interf = img1 * np.conj(img2_der_regist_rer)
    corrected_interf = interf * np.exp(-1j * (flat_earth + topo_phase))

    coh = eos.sar.coherence.on_pair(
        img1, img2_der_regist_rer, filter_size=(5, 5), might_contain_nans=True
    )

    np.save(f"{outdir}/interf.npy", interf)
    np.save(f"{outdir}/interf_corrected.npy", corrected_interf)
    np.save(f"{outdir}/coherence.npy", coh)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
