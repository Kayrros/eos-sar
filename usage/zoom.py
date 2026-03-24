"""
Run with uv run --env-file .env --with matplotlib usage/zoom.py --result_dir usage/zoom_results
Ensure "CDSE_ACCESS_KEY_ID" and "CDSE_SECRET_ACCESS_KEY" are in the .env file
"""

import os

import boto3
import numpy as np
from matplotlib import pyplot as plt

import eos.dem
import eos.products.sentinel1 as s1
import eos.sar
from eos.products.sentinel1.catalog import CDSESentinel1SLCCatalogBackend
from eos.products.sentinel1.product import CDSEUnzippedSafeSentinel1SLCProductInfo
from eos.sar.orbit import Orbit


def save_array(result_dir, name, array):
    np.save(os.path.join(result_dir, name), array)


def close_readers(readers):
    for read in readers:
        read.close()


def get_ref_metas(products):
    xml_contents = [product.get_xml_annotation("iw2", "vv") for product in products]
    ref_metas = [
        s1.coordinate_correction.FullBistaticReference.from_burst_metadata(
            s1.metadata.extract_burst_metadata(xml_content, 0)
        )
        for xml_content in xml_contents
    ]
    return ref_metas


def inputs():
    assert "CDSE_ACCESS_KEY_ID" in os.environ and "CDSE_SECRET_ACCESS_KEY" in os.environ
    cdse_s3_session = boto3.Session(
        aws_access_key_id=os.environ["CDSE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CDSE_SECRET_ACCESS_KEY"],
    )

    product_ids = [
        "S1B_IW_SLC__1SDV_20190803T164006_20190803T164034_017424_020C57_25CA",
        "S1A_IW_SLC__1SDV_20190809T164050_20190809T164117_028495_033896_2CC5",
    ]

    catalog_backend = CDSESentinel1SLCCatalogBackend()
    products = [
        CDSEUnzippedSafeSentinel1SLCProductInfo.from_product_id(
            catalog_backend, cdse_s3_session, product_id
        )
        for product_id in product_ids
    ]

    # read the xmls of iw3 vv as strings
    xml_content = [product.get_xml_annotation("iw3", "vv") for product in products]

    # get image readers of iw3 vv
    image_readers = [product.get_image_reader("iw3", "vv") for product in products]

    # Now extract the needed metadata
    primary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[0])
    secondary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[1])

    ref_metas = get_ref_metas(products)

    return image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas


def _get_objects(burst_meta, bistatic_ref=None):
    # create an orbit
    orbit = Orbit(burst_meta["state_vectors"])
    # create a doppler
    doppler = s1.doppler_info.doppler_from_meta(burst_meta, orbit)
    # create a corrector
    corrector = s1.coordinate_correction.s1_corrector_from_meta(
        burst_meta,
        orbit,
        doppler,
        apd=True,
        bistatic=True,
        full_bistatic_reference=bistatic_ref,
        intra_pulse=True,
        alt_fm_mismatch=True,
    )
    # Now instantiate burst_model instances for projection/localization
    burst_model = s1.proj_model.burst_model_from_burst_meta(
        burst_meta, orbit, corrector
    )
    return orbit, doppler, corrector, burst_model


def plot_freq_profile(img, axis=1, fs=1, title="", result_dir=None):
    interf_freq = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    profile = np.mean(interf_freq, axis=axis)
    n = len(profile)
    frequency = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))

    indices = list(np.where(profile > np.percentile(profile, 95) * 0.05)[0])
    if len(indices) == 0:
        indices = [0, -1]

    f_low = frequency[indices[0]]
    f_high = frequency[indices[-1]]
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("log(fourier amplitude)")
    ax.plot(frequency, np.log(profile + 1e-32))
    ax.axvline(f_low, c="r", linestyle="--")
    ax.axvline(f_high, c="r", linestyle="--")
    ax.set_title(title)
    if result_dir is not None:
        plt.savefig(os.path.join(result_dir, f"{'_'.join(title.split())}.png"), dpi=250)
    # plt.show()
    plt.close()
    print(f"Specs for {title}: ")
    print(f"Center frequency {np.mean([f_low, f_high])} Hz")
    print(f"width : {f_high - f_low} Hz")
    print("################### Done ###################")


def main(result_dir="."):
    os.makedirs(result_dir, exist_ok=True)

    image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs()
    dem_source = eos.dem.SRTM4Source()

    orbit = Orbit(s1.metadata.unique_sv_from_bursts_meta(primary_bursts_meta))
    # construct primary swath model
    primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
        primary_bursts_meta, orbit
    )
    # get dem points
    dem = primary_swath_model.fetch_dem(dem_source)
    x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
        primary_swath_model, dem=dem
    )

    primary_cutter = s1.acquisition.make_primary_cutter_from_bursts_meta(
        primary_bursts_meta
    )
    #
    # If you wish to deburst a "crop" defined by a roi in the swath coordinates
    roi_in_swath = eos.sar.roi.Roi(5000, 4500, 1000, 3000)

    # compute read/write rois
    (
        bsids,
        within_burst_rois,
        write_rois,
        out_shape,
    ) = primary_swath_model.get_debursting_rois(roi_in_swath)
    primary_image_readers = {bsid: image_readers[0] for bsid in bsids}
    secondary_image_readers = {bsid: image_readers[1] for bsid in bsids}

    # construct burst models with appropriate corrections
    primary_correctors = {
        b["bsid"]: _get_objects(b, ref_metas[0])[2]
        for b in primary_bursts_meta
        if b["bsid"] in bsids
    }
    primary_bursts_meta_per_bsid = {b["bsid"]: b for b in primary_bursts_meta}

    # project in the mosaic
    azt_primary_flat, rng_primary_flat, _ = primary_swath_model.projection(
        x, y, alt, crs=crs, as_azt_rng=True
    )

    pts_in_burst_mask = {}
    azt_primary = {}
    rng_primary = {}
    for bsid in bsids:
        # Calling mask_pts_in_burst multiple times is inefficient due to the conversion from
        # from azt/rng to row/col in the burst. However, profiling shows that the dem.crop is by far slower.
        burst_mask = primary_cutter.mask_pts_in_burst(
            bsid, azt_primary_flat, rng_primary_flat
        )
        pts_in_burst_mask[bsid] = burst_mask
        azt_primary[bsid] = azt_primary_flat[burst_mask]
        rng_primary[bsid] = rng_primary_flat[burst_mask]

    def regist(
        swath_model, cutter, corrector_per_bsid, readers, metas_per_bsid, reramp=True
    ):
        burst_resampling_matrices = s1.regist.secondary_registration_estimation(
            swath_model,
            cutter,
            corrector_per_bsid,
            x,
            y,
            alt,
            crs,
            bsids,
            pts_in_burst_mask,
            primary_cutter,
            azt_primary,
            rng_primary,
        )
        # instantiate resamplers
        resamplers = {
            bsid: s1.burst_resamp.burst_resample_from_meta(
                metas_per_bsid[bsid],
                primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                burst_resampling_matrices[bsid],
                _get_objects(metas_per_bsid[bsid])[1],
            )
            for bsid in bsids
        }

        (
            debursted_crop,
            _,
            resamplers_on_roi,
        ) = s1.deburst.warp_rois_read_resample_deburst(
            bsids,
            resamplers,
            within_burst_rois,
            cutter,
            readers,
            write_rois,
            out_shape,
            get_complex=True,
            reramp=reramp,
        )

        assert debursted_crop.shape == out_shape, "crop shape mismatch"
        assert np.isnan(debursted_crop).sum() / debursted_crop.size < 0.05
        return debursted_crop, resamplers_on_roi

    primary_crop, primary_resamplers = regist(
        primary_swath_model,
        primary_cutter,
        primary_correctors,
        primary_image_readers,
        primary_bursts_meta_per_bsid,
        reramp=True,
    )
    save_array(result_dir, "primary_mosaic.npy", primary_crop)

    crop_roi = eos.sar.roi.Roi(248, 300, 600, 1000)

    zoom_factor = 2
    azf = primary_swath_model.coordinate.azimuth_frequency
    mosaic_zoomer = s1.mosaic_zoom.MosaicZoomer(
        bsids,
        write_rois,
        crop_roi,
        zoom_factor=zoom_factor,
        previous_resamplers=primary_resamplers,
    )
    # test step by step
    deramped = mosaic_zoomer.deramp(crop_roi.crop_array(primary_crop))
    plot_freq_profile(
        deramped, axis=1, fs=azf, title="primary deramped", result_dir=result_dir
    )

    zoomed = mosaic_zoomer.zoom_fourier(deramped)
    plot_freq_profile(
        zoomed,
        axis=1,
        fs=azf * zoom_factor,
        title="primary zoomed",
        result_dir=result_dir,
    )

    reramped = mosaic_zoomer.reramp(zoomed)
    plot_freq_profile(
        reramped,
        axis=1,
        fs=azf * zoom_factor,
        title="primary reramped",
        result_dir=result_dir,
    )
    # test three zooming options
    lanczos_zoom = mosaic_zoomer.resample(crop_roi.crop_array(primary_crop))
    nan_mask = np.isnan(lanczos_zoom)
    lanczos_zoom[nan_mask] = 0
    plot_freq_profile(
        lanczos_zoom,
        axis=1,
        fs=azf * zoom_factor,
        title="lanczos zoomed",
        result_dir=result_dir,
    )

    zoomed_with_fourier = mosaic_zoomer.resample_fourier(
        crop_roi.crop_array(primary_crop)
    )
    plot_freq_profile(
        zoomed_with_fourier,
        axis=1,
        fs=azf * zoom_factor,
        title="joint fourier zoomed",
        result_dir=result_dir,
    )

    zoomed_with_fourier_separate = mosaic_zoomer.resample_fourier(
        crop_roi.crop_array(primary_crop), joint_resampling=False
    )
    plot_freq_profile(
        zoomed_with_fourier_separate,
        axis=1,
        fs=azf * zoom_factor,
        title="separate fourier zoom",
        result_dir=result_dir,
    )

    # Now get a secondary mosaic
    # construct secondary swath model and burst models
    orbit = Orbit(s1.metadata.unique_sv_from_bursts_meta(secondary_bursts_meta))
    secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
        secondary_bursts_meta, orbit
    )
    secondary_cutter = s1.acquisition.make_secondary_cutter_from_bursts_meta(
        secondary_bursts_meta
    )

    secondary_correctors = {
        b["bsid"]: _get_objects(b, ref_metas[0])[2]
        for b in secondary_bursts_meta
        if b["bsid"] in bsids
    }
    secondary_bursts_meta_per_bsid = {b["bsid"]: b for b in secondary_bursts_meta}

    secondary_crop, secondary_resamplers = regist(
        secondary_swath_model,
        secondary_cutter,
        secondary_correctors,
        secondary_image_readers,
        secondary_bursts_meta_per_bsid,
        reramp=True,
    )

    save_array(result_dir, "secondary_mosaic.npy", secondary_crop)

    # zoom secondary crop and look at zoomed interferogram
    secondary_zoomer = s1.mosaic_zoom.MosaicZoomer(
        bsids, write_rois, crop_roi, zoom_factor, secondary_resamplers
    )
    # do interf after zoom and look at spectral width, check aliasing
    zoomed_secondary = secondary_zoomer.resample_fourier(
        crop_roi.crop_array(secondary_crop)
    )
    # look at spectral width primary and secondary
    plot_freq_profile(
        secondary_zoomer.deramp(crop_roi.crop_array(secondary_crop)),
        axis=1,
        fs=azf,
        title="secondary deramped",
        result_dir=result_dir,
    )
    plot_freq_profile(
        mosaic_zoomer.deramp(crop_roi.crop_array(primary_crop)),
        axis=1,
        fs=azf,
        title="primary deramped",
        result_dir=result_dir,
    )
    interf_zoomed = zoomed_with_fourier * np.conj(zoomed_secondary)
    plot_freq_profile(
        interf_zoomed,
        axis=1,
        fs=azf * zoom_factor,
        title="zoomed interf",
        result_dir=result_dir,
    )
    plot_freq_profile(
        crop_roi.crop_array(primary_crop)
        * np.conj(crop_roi.crop_array(secondary_crop)),
        axis=1,
        fs=azf,
        title="interf not zoomed",
        result_dir=result_dir,
    )

    save_array(result_dir, "primary_crop.npy", crop_roi.crop_array(primary_crop))
    save_array(result_dir, "secondary_crop.npy", crop_roi.crop_array(secondary_crop))
    save_array(result_dir, "primary_zoomed_fourier.npy", zoomed_with_fourier)
    save_array(result_dir, "primary_zoomed_lanczos.npy", lanczos_zoom)
    save_array(
        result_dir, "primary_zoomed_fourier_separate.npy", zoomed_with_fourier_separate
    )
    save_array(result_dir, "secondary_zoomed_fourier.npy", zoomed_secondary)
    save_array(result_dir, "zoomed_interf.npy", interf_zoomed)


if __name__ == "__main__":
    import time

    import fire

    start = time.time()
    fire.Fire(main)
    duration = time.time() - start
    print(f"Finished! Took {duration} seconds")
