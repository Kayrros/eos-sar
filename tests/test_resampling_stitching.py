import numpy as np
import os
import eos.products.sentinel1 as s1
import eos.sar
import pytest


def extract_keys(big_dict, list_keys):
    o = {}
    for key in list_keys:
        o[key] = big_dict[key]
    return o


def get_ref_metas(ref_xml_paths):
    xml_contents = [eos.sar.io.read_xml_file(xml_path) for xml_path in ref_xml_paths]
    keys = ['slant_range_time',
            'samples_per_burst',
            'range_frequency']
    ref_metas = [extract_keys(eos.products.sentinel1.metadata.extract_burst_metadata(
        xml_content, 0), keys) for xml_content in xml_contents]
    return ref_metas


def close_readers(readers):
    for read in readers:
        read.close()


@pytest.fixture(scope="module")
def inputs():

    xml_folder = 's3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/annotation'

    tiff_folder = 's3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/measurement'

    xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                     's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']

    tiff_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                      's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']

    ref_basenames = ['s1b-iw2-slc-vv-20190803t164006-20190803t164034-017424-020c57-005.xml',
                     's1a-iw2-slc-vv-20190809t164051-20190809t164117-028495-033896-005.xml']

    # list of our xmls
    xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames]

    tiff_paths = [os.path.join(tiff_folder, p) for p in tiff_basenames]

    # read the xmls as strings
    xml_content = []
    for xml_path in xml_paths:
        xml_content.append(eos.sar.io.read_xml_file(xml_path))

    image_readers = [eos.sar.io.open_image(p) for p in tiff_paths]

    # Now extract the needed metadata
    primary_bursts_meta = s1.metadata.extract_bursts_metadata(
        xml_content[0])
    secondary_bursts_meta = s1.metadata.extract_bursts_metadata(
        xml_content[1])

    ref_metas = get_ref_metas([os.path.join(xml_folder, ref_base) for ref_base in ref_basenames])

    yield image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas
    close_readers(image_readers)


@pytest.fixture(scope="module")
def dem(inputs):
    image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs
    # construct primary swath model
    primary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
        primary_bursts_meta)
    # get dem points
    x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(primary_swath_model)
    return x, y, alt, crs


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
            [primary_bursts_meta[0]['relative_burst_id'], secondary_bursts_meta[0]['relative_burst_id']]
        )
        assert np.all(sec_burst_ids == np.arange(5))
        assert np.all(prim_burst_ids == np.arange(2, 7))

    def mat_estim(self, prim_mod, sec_mod, x, y, alt, crs):

        # project in primary
        row_primary, col_primary, _ = prim_mod.projection(
            x, y, alt, crs=crs)

        # project in secondary and estimate registration
        A = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                                sec_mod, x, y, alt, crs)
        return A

    def test_burst_matrix_estimation(self, inputs, dem):

        image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs

        # start by testing the burst resampling feature
        b_index = 3
        # Now instantiate burst_model instances for projection/localization
        primary_burst_model = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            primary_bursts_meta[b_index], bistatic_correction=True,
            full_bistatic_correction_reference=ref_metas[0],
            apd_correction=True,
            intra_pulse_correction=True)

        secondary_burst_model = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            secondary_bursts_meta[b_index], bistatic_correction=True,
            full_bistatic_correction_reference=ref_metas[1],
            apd_correction=True,
            intra_pulse_correction=True)

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
        A = np.array([[9.99998152e-01, -2.67751722e-07, -5.73518074e+00],
                      [1.49618113e-05, 1.00019496e+00, 4.55021115e+01],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # resampler on the complex secondary burst
        col_dst, row_dst, w_dst, h_dst = primary_burst_meta['burst_roi']
        col_src, row_src, w_src, h_src = secondary_burst_meta['burst_roi']
        resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(secondary_burst_meta,
                                                                                 dst_burst_shape=(
                                                                                     h_dst, w_dst),
                                                                                 matrix=A)

        # Region of interest inside the burst in the primary (col, row, w, h)
        dst_roi_in_burst = eos.sar.roi.Roi(90, 90, 150, 100)

        # warp the roi to the secondary, and add a margin of 5 pixels on each side
        src_roi_in_burst = dst_roi_in_burst.warp_valid_roi((h_dst, w_dst), (h_src, w_src),
                                                           A, margin=5)

        # set the resampler to work on rois inside the burst
        # this will adapt the resampling matrix to the roi origins
        # and will adapt the deramping origin (since deramping depends on pixel position)
        resampler.set_inside_burst(dst_roi_in_burst, src_roi_in_burst)

        assert np.any(resampler.matrix != resampler.burst_matrix)

        # translate the roi origin from the burst to the tiff coordinates
        secondary_tiff_roi = src_roi_in_burst.translate_roi(col_src, row_src)

        # read the roi inside the secondary burst
        secondary_burst_array = eos.sar.io.read_window(
            image_readers[1], secondary_tiff_roi)

        # resample
        resampled_secondary_array = resampler.resample(secondary_burst_array)

        resamp_h, resamp_w = resampled_secondary_array.shape

        assert resamp_h == 100
        assert resamp_w == 150
        assert np.isnan(resampled_secondary_array).sum() / resampled_secondary_array.size < 0.05
        assert resampled_secondary_array.dtype == np.complex64

        # If you wish to resample the amplitude

        # resample amplitude if you want to do this only (without phase )
        resampled_secondary_amplitude = resampler.resample(np.abs(secondary_burst_array))

        resamp_h, resamp_w = resampled_secondary_amplitude.shape

        assert resamp_h == 100
        assert resamp_w == 150
        assert np.isnan(resampled_secondary_amplitude).sum() / resampled_secondary_amplitude.size < 0.05
        assert resampled_secondary_amplitude.dtype == np.float32

        # make sure that the abs of the resampled array is close to the resampling of the abs
        #   the difference is actually quite large, because of the deramping/reramping
        assert np.abs(np.abs(resampled_secondary_array) - resampled_secondary_amplitude).mean() < 10

        # you can reset the resampler in case burst resampling is needed
        resampler.set_to_default_roi()

        assert np.all(resampler.matrix == resampler.burst_matrix)

    def test_debursting(self, inputs, dem):

        image_readers, primary_bursts_meta, secondary_bursts_meta, ref_metas = inputs

        corrections = {
            'bistatic_correction': True,
            'apd_correction': True,
            'intra_pulse_correction': True,
        }

        x, y, alt, crs = dem

        # construct primary swath model
        primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(primary_bursts_meta)

        # If you wish to deburst a "crop" defined by a roi in the swath coordinates
        roi_in_swath = eos.sar.roi.Roi(5000, 750, 40, 3000)

        # compute read/write rois
        bsids, read_rois_no_correc, write_rois_no_correc, out_shape = primary_swath_model.get_read_write_rois(
            roi_in_swath)
        primary_image_readers = {bsid: image_readers[0] for bsid in bsids}
        secondary_image_readers = {bsid: image_readers[1] for bsid in bsids}

        # construct burst models with appropriate corrections
        primary_burst_models = {b['bsid']: s1.proj_model.burst_model_from_burst_meta(
            b, full_bistatic_correction_reference=ref_metas[0],
            **corrections) for b in primary_bursts_meta if b['bsid'] in bsids}
        primary_bursts_meta = {b['bsid']: b for b in primary_bursts_meta}

        # estimate the matrices and resample
        rows_no_correc_global, cols_no_correc_global, _, _, pts_in_burst_mask, \
            burst_resampling_matrices = s1.regist.primary_registration_estimation(
                primary_swath_model, primary_burst_models, x, y, alt, crs, bsids)

        primary_debursted_crop, _, _ =  \
            eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
                read_rois_no_correc, bsids, primary_swath_model,
                primary_swath_model, burst_resampling_matrices,
                primary_bursts_meta, primary_image_readers,
                write_rois_no_correc, out_shape)

        # construct secondary swath model and burst models
        secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(secondary_bursts_meta)

        secondary_burst_models = {b['bsid']: s1.proj_model.burst_model_from_burst_meta(
            b, full_bistatic_correction_reference=ref_metas[1],
            **corrections) for b in secondary_bursts_meta if b['bsid'] in bsids}
        secondary_bursts_meta = {b['bsid']: b for b in secondary_bursts_meta}

        # estimate the matrices and resample
        burst_resampling_matrices = \
            s1.regist.secondary_registration_estimation(
                secondary_swath_model, secondary_burst_models, x, y, alt, crs,
                bsids, pts_in_burst_mask, primary_swath_model, rows_no_correc_global,
                cols_no_correc_global, global_rows_fit=True)

        secondary_debursted_crop, _, _ = \
            eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
                read_rois_no_correc, bsids, primary_swath_model,
                secondary_swath_model, burst_resampling_matrices,
                secondary_bursts_meta, secondary_image_readers,
                write_rois_no_correc, out_shape,
                get_complex=True)

        assert primary_debursted_crop.shape == roi_in_swath.get_shape(), "crop shape mismatch"
        assert secondary_debursted_crop.shape == roi_in_swath.get_shape(), "crop shape mismatch"
