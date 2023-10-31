import os

import numpy as np
import pytest

import eos.dem
import eos.products.sentinel1
import eos.products.sentinel1 as s1
import eos.sar
from eos.products.sentinel1.coordinate_correction import FullBistaticReference
from eos.products.sentinel1.doppler_info import Sentinel1Doppler
from eos.products.sentinel1.metadata import Sentinel1BurstMetadata
from eos.products.sentinel1.proj_model import Sentinel1BurstModel
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector


def get_ref_metas(ref_xml_paths):
    xml_contents = [eos.sar.io.read_xml_file(xml_path) for xml_path in ref_xml_paths]
    ref_metas = [
        FullBistaticReference.from_burst_metadata(
            s1.metadata.extract_burst_metadata(xml_content, 0)
        )
        for xml_content in xml_contents
    ]
    return ref_metas


def close_readers(readers):
    for read in readers:
        read.close()


@pytest.fixture(scope="module")
def inputs():
    xml_folder = (
        "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/annotation"
    )

    tiff_folder = (
        "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/measurement"
    )

    xml_basenames = [
        "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml",
        "s1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml",
    ]

    tiff_basenames = [
        "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff",
        "s1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff",
    ]

    ref_basenames = [
        "s1b-iw2-slc-vv-20190803t164006-20190803t164034-017424-020c57-005.xml",
        "s1a-iw2-slc-vv-20190809t164051-20190809t164117-028495-033896-005.xml",
    ]

    # list of our xmls
    xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames]

    tiff_paths = [os.path.join(tiff_folder, p) for p in tiff_basenames]

    # read the xmls as strings
    xml_content = []
    for xml_path in xml_paths:
        xml_content.append(eos.sar.io.read_xml_file(xml_path))

    image_readers = [eos.sar.io.open_image(p) for p in tiff_paths]

    # Now extract the needed metadata
    primary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[0])
    secondary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[1])

    ref_metas = get_ref_metas(
        [os.path.join(xml_folder, ref_base) for ref_base in ref_basenames]
    )

    yield image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas
    close_readers(image_readers)


@pytest.fixture(scope="module")
def dem(inputs):
    image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs

    orbit = Orbit(s1.metadata.unique_sv_from_bursts_meta(primary_bursts_meta))
    # construct primary swath model
    primary_swath_model = (
        eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
            primary_bursts_meta, orbit
        )
    )
    # get dem points
    dem_source = eos.dem.get_any_source()
    dem = primary_swath_model.fetch_dem(dem_source)
    x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
        primary_swath_model, dem=dem
    )
    return x, y, alt, crs


def _get_objects(
    burst_meta: Sentinel1BurstMetadata,
    ref_meta=None,
) -> tuple[Orbit, Sentinel1Doppler, Corrector, Sentinel1BurstModel]:
    # create an orbit
    orbit = Orbit(burst_meta.state_vectors)
    # create a doppler
    doppler = s1.doppler_info.doppler_from_meta(burst_meta, orbit)
    # create a corrector
    corrector = s1.coordinate_correction.s1_corrector_from_meta(
        burst_meta,
        orbit,
        doppler,
        apd=True,
        bistatic=True,
        full_bistatic_reference=ref_meta,
        intra_pulse=True,
        alt_fm_mismatch=True,
    )
    # Now instantiate burst_model instances for projection/localization
    burst_model = s1.proj_model.burst_model_from_burst_meta(
        burst_meta, orbit, corrector
    )
    return orbit, doppler, corrector, burst_model


class Test_Resample_Stitch:
    def test_burst_intersection(self, inputs):
        image_readers, primary_bursts_meta, secondary_bursts_meta, _ = inputs
        # simulate a difference in bursts footprints
        # this is to test, we could also choose a pair of images with
        # different burst ids in the products
        primary_bursts_meta = primary_bursts_meta[:-2]
        secondary_bursts_meta = secondary_bursts_meta[2:]

        # get the indices of the common bursts
        prim_burst_ids, sec_burst_ids = s1.deburst.get_bursts_intersection(
            [len(primary_bursts_meta), len(secondary_bursts_meta)],
            [
                primary_bursts_meta[0].relative_burst_id,
                secondary_bursts_meta[0].relative_burst_id,
            ],
        )
        assert np.all(sec_burst_ids == np.arange(5))
        assert np.all(prim_burst_ids == np.arange(2, 7))

    def mat_estim(self, prim_mod, sec_mod, x, y, alt, crs):
        # project in primary
        row_primary, col_primary, _ = prim_mod.projection(x, y, alt, crs=crs)

        # project in secondary and estimate registration
        A = eos.sar.regist.orbital_registration(
            row_primary, col_primary, sec_mod, x, y, alt, crs
        )
        return A

    def test_burst_matrix_estimation(self, inputs, dem):
        image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs

        # start by testing the burst resampling feature
        b_index = 3

        # Now instantiate burst_model instances for projection/localization
        primary_burst_model = _get_objects(primary_bursts_meta[b_index], ref_metas[0])[
            3
        ]
        secondary_burst_model = _get_objects(
            secondary_bursts_meta[b_index], ref_metas[1]
        )[3]

        A = self.mat_estim(primary_burst_model, secondary_burst_model, *dem)
        assert np.any(A != np.eye(3))
        A_inv = self.mat_estim(secondary_burst_model, primary_burst_model, *dem)
        assert np.any(A_inv != np.eye(3))

        assert np.allclose(A.dot(A_inv), np.eye(3), rtol=1e-2, atol=1e-2)

    def test_burst_registration(self, inputs):
        image_readers, primary_bursts_meta, secondary_bursts_meta, _ = inputs

        # start by testing the burst resampling feature
        b_index = 3
        primary_burst_meta = primary_bursts_meta[b_index]
        secondary_burst_meta = secondary_bursts_meta[b_index]

        # A hardcoded for faster test
        A = np.array(
            [
                [9.99998152e-01, -2.67751722e-07, -5.73518074e00],
                [1.49618113e-05, 1.00019496e00, 4.55021115e01],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

        # resampler on the complex secondary burst
        col_dst, row_dst, w_dst, h_dst = primary_burst_meta.burst_roi
        col_src, row_src, w_src, h_src = secondary_burst_meta.burst_roi

        secondary_doppler = _get_objects(secondary_burst_meta)[1]
        burst_resampler = s1.burst_resamp.burst_resample_from_meta(
            secondary_burst_meta,
            dst_burst_shape=(h_dst, w_dst),
            matrix=A,
            doppler=secondary_doppler,
        )

        assert np.all(burst_resampler.matrix == burst_resampler.burst_matrix)

        # Region of interest inside the burst in the primary (col, row, w, h)
        dst_roi_in_burst = eos.sar.roi.Roi(90, 90, 150, 100)
        src_roi_in_burst = burst_resampler.src_roi_from_dst_roi(dst_roi_in_burst)
        resampler_on_roi = burst_resampler.get_resampler_on_different_roi(
            dst_roi_in_burst, src_roi_in_burst
        )

        assert np.any(resampler_on_roi.matrix != resampler_on_roi.burst_matrix)

        # translate the roi origin from the burst to the tiff coordinates
        secondary_tiff_roi = src_roi_in_burst.translate_roi(col_src, row_src)

        # read the roi inside the secondary burst
        secondary_burst_array = eos.sar.io.read_window(
            image_readers[1], secondary_tiff_roi
        )

        # resample
        resampled_secondary_array = resampler_on_roi.resample(secondary_burst_array)

        resamp_h, resamp_w = resampled_secondary_array.shape

        assert resamp_h == 100
        assert resamp_w == 150
        assert (
            np.isnan(resampled_secondary_array).sum() / resampled_secondary_array.size
            < 0.05
        )
        assert resampled_secondary_array.dtype == np.complex64

        # If you wish to resample the amplitude

        # resample amplitude if you want to do this only (without phase )
        resampled_secondary_amplitude = resampler_on_roi.resample(
            np.abs(secondary_burst_array)
        )

        resamp_h, resamp_w = resampled_secondary_amplitude.shape

        assert resamp_h == 100
        assert resamp_w == 150
        assert (
            np.isnan(resampled_secondary_amplitude).sum()
            / resampled_secondary_amplitude.size
            < 0.05
        )
        assert resampled_secondary_amplitude.dtype == np.float32

        # make sure that the abs of the resampled array is close to the resampling of the abs
        #   the difference is actually quite large, because of the deramping/reramping
        assert (
            np.abs(
                np.abs(resampled_secondary_array) - resampled_secondary_amplitude
            ).mean()
            < 10
        )

    def test_debursting(self, inputs, dem):
        image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs

        x, y, alt, crs = dem

        orbit = Orbit(s1.metadata.unique_sv_from_bursts_meta(primary_bursts_meta))
        # construct primary swath model and acquisition cutter
        primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
            primary_bursts_meta, orbit
        )
        primary_cutter = s1.acquisition.make_primary_cutter_from_bursts_meta(
            primary_bursts_meta
        )

        # If you wish to deburst a "crop" defined by a roi in the swath coordinates
        roi_in_swath = eos.sar.roi.Roi(5000, 750, 100, 3000)

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
            b.bsid: _get_objects(b, ref_metas[0])[2]
            for b in primary_bursts_meta
            if b.bsid in bsids
        }
        primary_bursts_meta = {b.bsid: b for b in primary_bursts_meta}

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

        def regist(swath_model, cutter, corrector_per_bsid, readers, metas_per_bsid):
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
                reramp=True,
            )

            assert debursted_crop.shape == out_shape, "crop shape mismatch"
            assert np.isnan(debursted_crop).sum() / debursted_crop.size < 0.05
            return debursted_crop, resamplers_on_roi

        primary_crop, primary_resamplers = regist(
            primary_swath_model,
            primary_cutter,
            primary_correctors,
            primary_image_readers,
            primary_bursts_meta,
        )

        # construct secondary swath model and burst models
        orbit = Orbit(s1.metadata.unique_sv_from_bursts_meta(secondary_bursts_meta))
        secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
            secondary_bursts_meta, orbit
        )
        secondary_cutter = s1.acquisition.make_secondary_cutter_from_bursts_meta(
            secondary_bursts_meta
        )

        secondary_correctors = {
            b.bsid: _get_objects(b, ref_metas[0])[2]
            for b in secondary_bursts_meta
            if b.bsid in bsids
        }
        secondary_bursts_meta = {b.bsid: b for b in secondary_bursts_meta}

        secondary_crop, _ = regist(
            secondary_swath_model,
            secondary_cutter,
            secondary_correctors,
            secondary_image_readers,
            secondary_bursts_meta,
        )

        # test mosaic zoom
        crop_roi = eos.sar.roi.Roi(1, 1500, 90, 300)
        zoom_factor = 2
        mosaic_zoomer = s1.mosaic_zoom.MosaicZoomer(
            bsids,
            write_rois,
            crop_roi,
            zoom_factor=zoom_factor,
            previous_resamplers=primary_resamplers,
        )

        lanczos_zoom = mosaic_zoomer.resample(crop_roi.crop_array(primary_crop))
        assert np.isnan(lanczos_zoom).sum() / lanczos_zoom.size < 0.1
        zoomed_with_fourier = mosaic_zoomer.resample_fourier(
            crop_roi.crop_array(primary_crop)
        )
        assert np.isnan(zoomed_with_fourier).sum() / zoomed_with_fourier.size < 0.05
        zoomed_with_fourier_separate = mosaic_zoomer.resample_fourier(
            crop_roi.crop_array(primary_crop), joint_resampling=False
        )
        assert (
            np.isnan(zoomed_with_fourier_separate).sum()
            / zoomed_with_fourier_separate.size
            < 0.05
        )
