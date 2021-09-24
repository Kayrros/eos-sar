import numpy as np
import os
import eos.products.sentinel1 as s1
import eos.sar

class Test_Resample_Stitch:

    def get_readers(self):

        xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'

        tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'

        xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                             's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']

        tiff_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                          's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']

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

        return image_readers, primary_bursts_meta, secondary_bursts_meta

    def close_readers(self, readers):
        for read in readers:
            read.close()

    def test_burst_intersection(self):

        image_readers, primary_bursts_meta, secondary_bursts_meta = self.get_readers()
        # simulate a difference in bursts footprints
        # this is to test, we could also choose a pair of images with
        # different burst ids in the products
        primary_bursts_meta = primary_bursts_meta[:-2]
        secondary_bursts_meta = secondary_bursts_meta[2:]

        # get the indices of the common bursts
        prim_burst_ids, sec_burst_ids = s1.deburst.get_bursts_intersection(
            len(primary_bursts_meta),
            primary_bursts_meta[0]['relative_burst_id'],
            len(secondary_bursts_meta),
            secondary_bursts_meta[0]['relative_burst_id']
        )
        assert np.all(sec_burst_ids == np.arange(5))
        assert np.all(prim_burst_ids == np.arange(2,7))
        self.close_readers(image_readers)

    def mat_estim(self, prim_mod, sec_mod):

        # get dem points
        x, y, raster, transform, crs = eos.sar.regist.dem_points(prim_mod.approx_geom,
                                                                  source='SRTM30',
                                                                  datum='ellipsoidal'
                                                                  )

        # you can mask some pixels to speed up the projection
        mask = np.random.binomial(n=1, p=0.01, size=x.shape).astype(bool)
        x = x[mask]
        y = y[mask]
        raster = raster[mask]

        # project in primary
        row_primary, col_primary, _ = prim_mod.projection(
            x.ravel(), y.ravel(), raster.ravel(), crs=crs)

        # project in secondary and estimate registration
        A = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                                sec_mod, x, y, raster, crs)
        return A

    def test_burst_matrix_estimation(self):

        image_readers, primary_bursts_meta, secondary_bursts_meta = self.get_readers()

        # start by testing the burst resampling feature
        b_index = 3
        primary_burst_meta = primary_bursts_meta[b_index]
        secondary_burst_meta = secondary_bursts_meta[b_index]

        # Now instantiate burst_model instances for projection/localization
        primary_burst_model = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            primary_burst_meta)
        secondary_burst_model = eos.products.sentinel1.proj_model.burst_model_from_burst_meta(
            secondary_burst_meta)
        A = self.mat_estim(primary_burst_model, secondary_burst_model)
        assert np.any(A != np.eye(3))
        A_inv = self.mat_estim( secondary_burst_model, primary_burst_model)
        assert np.any(A_inv != np.eye(3))

        assert np.allclose(A.dot(A_inv), np.eye(3), rtol=1e-2 , atol = 1e-2)

        self.close_readers(image_readers)

    def test_burst_registration(self):

        image_readers, primary_bursts_meta, secondary_bursts_meta = self.get_readers()

        # start by testing the burst resampling feature
        b_index = 3
        primary_burst_meta = primary_bursts_meta[b_index]
        secondary_burst_meta = secondary_bursts_meta[b_index]

        # A hardcoded for faster test
        A = np.array([[ 9.99998152e-01, -2.67751722e-07, -5.73518074e+00],
                      [ 1.49618113e-05,  1.00019496e+00,  4.55021115e+01],
                      [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        # resampler on the complex secondary burst
        col_dst, row_dst, w_dst, h_dst = primary_burst_meta['burst_roi']
        col_src, row_src, w_src, h_src = secondary_burst_meta['burst_roi']
        resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(secondary_burst_meta,
                                                                                  dst_burst_shape=(
                                                                                      h_dst, w_dst),
                                                                                  matrix=A, degree=11)

        # Region of interest inside the burst in the primary (col, row, w, h)
        dst_roi_in_burst = eos.sar.roi.Roi(90, 90, 150, 100)

        # warp the roi to the secondary, and add a margin of 5 pixels on each side
        src_roi_in_burst = dst_roi_in_burst.warp_valid_roi((h_dst, w_dst), (h_src, w_src),
                                                        A, margin=5, inplace=False)

        # set the resampler to work on rois inside the burst
        # this will adapt the resampling matrix to the roi origins
        # and will adapt the deramping origin (since deramping depends on pixel position)
        resampler.set_inside_burst(dst_roi_in_burst, src_roi_in_burst)

        assert np.any(resampler.matrix != resampler.burst_matrix)

        # translate the roi origin from the burst to the tiff coordinates
        secondary_tiff_roi = src_roi_in_burst.translate_roi(col_src, row_src,
                                                            inplace=False)

        # read the roi inside the secondary burst
        secondary_burst_array = eos.sar.io.read_window(
            image_readers[1], secondary_tiff_roi)

        # resample
        resampled_secondary_array = resampler.resample(secondary_burst_array)

        resamp_h, resamp_w = resampled_secondary_array.shape

        assert resamp_h == 100
        assert resamp_w == 150
        assert np.isnan(resampled_secondary_array).sum() / resampled_secondary_array.size < 0.05

        ################### If you wish to resample the amplitude

        # resample amplitude if you want to do this only (without phase )
        resampled_secondary_amplitude = resampler.resample(np.abs(secondary_burst_array))


        resamp_h, resamp_w = resampled_secondary_amplitude.shape

        assert resamp_h == 100
        assert resamp_w == 150
        assert np.isnan(resampled_secondary_amplitude).sum() / resampled_secondary_amplitude.size < 0.05


        # you can reset the resampler in case burst resampling is needed
        resampler.set_to_default_roi()

        assert np.all(resampler.matrix == resampler.burst_matrix)

        self.close_readers(image_readers)

    def test_primary_debursting(self):

        image_readers, primary_bursts_meta, secondary_bursts_meta = self.get_readers()

        # construct primary swath model
        primary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
            primary_bursts_meta)

        # If you wish to deburst a "crop" defined by a roi in the swath coordinates
        roi_in_swath = eos.sar.roi.Roi(5000, 750, 40, 3000)

        # deburst
        for get_complex in [True, False]: 
            debursted_crop, burst_ids, rois_read, rois_write = eos.products.sentinel1.deburst.deburst_in_primary_swath(
                   primary_swath_model, image_readers[0], roi_in_swath, get_complex)
            h, w = debursted_crop.shape
            assert h == 3000
            assert w == 40
        self.close_readers(image_readers)

    def test_secondary_regist_deburst(self):

        image_readers, primary_bursts_meta, secondary_bursts_meta = self.get_readers()

        # construct primary swath model
        primary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
            primary_bursts_meta)
        # construct secondary swath model
        secondary_swath_model = eos.products.sentinel1.proj_model.swath_model_from_bursts_meta(
            secondary_bursts_meta)
        # hardcoded A to reduce runtime
        A_swath = np.array([[ 9.99998183e-01, -2.37111941e-07, -5.72787168e+00],
                         [ 9.33229568e-06,  1.00019457e+00,  4.54404670e+01],
                         [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        # define the roi in the primary swath
        # Here, if you set a region of interest within the swath
        # in the primary burst, only this region will be considered
        primary_swath_roi = eos.sar.roi.Roi(5000, 750, 40, 3000)

        burst_ids, read_rois, write_rois, out_shape = primary_swath_model.get_read_write_rois(
        primary_swath_roi)

        # Now for the secondary
        # estimate the rois where we need to read data
        # and the associated resampler
        secondary_read_rois, resamplers = s1.deburst.secondary_rois_and_resamplers(
            primary_swath_model, read_rois,
            burst_ids, secondary_swath_model,
            secondary_bursts_meta, A_swath)

        # Secondary reading/resampling/ debursting
        for get_complex in [True, False]: 
            secondary_debursted_crop = s1.deburst.read_resample_and_deburst(
                image_readers[1], secondary_read_rois,
                resamplers, write_rois, out_shape, get_complex)
    
            h, w = secondary_debursted_crop.shape
            assert h == 3000
            assert w == 40
            assert np.isnan(secondary_debursted_crop).sum() / secondary_debursted_crop.size < 0.05