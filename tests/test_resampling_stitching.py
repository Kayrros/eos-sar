import numpy as np
import os
import eos.products.sentinel1 as s1
import eos.sar

class Test_Resample_Stitch: 
    
    def get_readers(self): 
        
        xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
        
        tiff_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/measurement'
        # prepare oio config 
        prof_name = 'oio'
        en_url = 'https://s3.kayrros.org'
            
        xml_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                             's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']
        
        tiff_basenames = ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                          's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']
        
        # list of our xmls
        xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames ]
        
        tiff_paths = [os.path.join(tiff_folder, p) for p in tiff_basenames]
        
        image_readers = [eos.sar.io.open_image(p, profile_name=prof_name, 
                                               endpoint_url=en_url)
                         for p in tiff_paths]
        
        # read the xmls as strings
        xml_content = []
        for xml_path in xml_paths: 
                xml_content.append( eos.sar.io.read_xml_file(
                                        xml_path, profile_name=prof_name,
                                        endpoint_url=en_url))
        
        # Now extract the needed metadata
        primary_bursts_meta = s1.metadata.extract_bursts_metadata(
            xml_content[0])
        secondary_bursts_meta = s1.metadata.extract_bursts_metadata(
            xml_content[1])
        
        return image_readers, primary_bursts_meta, secondary_bursts_meta
    
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
        
    def test_burst_registration(self): 
        
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
        
        # Now estimate the registration matrix
        
        # get dem points
        x, y, raster, transform, crs = eos.sar.regist.dem_points(primary_burst_model,
                                                                  source='SRTM30',
                                                                  datum='ellipsoidal'
                                                                  )
        
        # you can mask some pixels to speed up the projection
        mask = np.random.binomial(n=1, p=0.01, size=x.shape).astype(bool)
        x = x[mask]
        y = y[mask]
        raster = raster[mask]
        
        # project in primary
        row_primary, col_primary, _ = primary_burst_model.projection(
            x.ravel(), y.ravel(), raster.ravel(), crs=crs)
        
        # project in secondary and estimate registration
        A = eos.sar.regist.orbital_registration(row_primary, col_primary,
                                                secondary_burst_model, x, y, raster, crs)
        
        assert np.any(A != np.eye(3))
        
        # resampler on the complex secondary burst
        col_dst, row_dst, w_dst, h_dst = primary_burst_meta['burst_roi']
        col_src, row_src, w_src, h_src = secondary_burst_meta['burst_roi']
        resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(secondary_burst_meta,
                                                                                  dst_burst_shape=(
                                                                                      h_dst, w_dst),
                                                                                  matrix=A, degree=11)
        
        # Region of interest inside the burst in the primary (col, row, w, h)
        dst_roi_in_burst = (90, 90, 150, 100)
        
        # warp the roi to the secondary, and add a margin of 5 pixels on each side
        src_roi_in_burst = eos.sar.roi.warp_valid_rois(dst_roi_in_burst,
                                                       (h_dst, w_dst), (h_src, w_src),
                                                        A, margin=5)
        
        # set the resampler to work on rois inside the burst
        # this will adapt the resampling matrix to the roi origins
        # and will adapt the deramping origin (since deramping depends on pixel position)
        resampler.set_inside_burst(dst_roi_in_burst, src_roi_in_burst)
        
        assert np.any(resampler.matrix != resampler.burst_matrix)  
        
        # translate the roi origin from the burst to the tiff coordinates
        secondary_tiff_roi = eos.sar.roi.translate_roi(
            src_roi_in_burst, col_src, row_src)
        
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
        # it is important to use resampler.matrix after the resampler has been set 
        # on the two rois. The matrix will be adapted to the rois location. 
        
        _, _, w, h = dst_roi_in_burst
        
        resampled_secondary_amplitude = eos.sar.regist.apply_affine(matrix=resampler.matrix,
                                                                    src_array=np.abs(
                                                                        secondary_burst_array),
                                                                    destination_array_shape=(h, w))
        
        resamp_h, resamp_w = resampled_secondary_amplitude.shape
        
        assert resamp_h == 100
        assert resamp_w == 150
        assert np.isnan(resampled_secondary_amplitude).sum() / resampled_secondary_amplitude.size < 0.05
        
        
        # you can reset the resampler in case burst resampling is needed
        resampler.set_to_default_roi()
        
        assert np.all(resampler.matrix == resampler.burst_matrix)
        
        