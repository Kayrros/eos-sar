# eos-sar

This package provides access to some generic sar processing algorithms. 

Currently, algorithms specific to **Sentinel1** bursts have been implemented. 

## Physical sensor model 

This allows the user to project a 3D point into the burst, and vice-versa. 

For localization:

    # create a Sentinel1Model
    s1model = s1m.Sentinel1Model(xml=xml_path)
    burst_meta = eos.products.sentinel1.metadata.fill_meta(s1model, burst_id)
    # create a Sentinel1BurstModel
    bmod = eos.products.sentinel1.burst_model.burst_model_from_burst_meta(burst_meta, 
                                                                        degree=11,
                                                                        bistatic_correction=True,
                                                                        apd_correction=True,
                                                                        max_iterations=20,
                                                                        tolerance=0.001)
    # create a grid of points
    x, y, w, h = bmod.burst_roi
    Cols, Rows = np.meshgrid(np.linspace(0, w-1, 10), np.linspace(0, h-1, 10))
    cols, rows = Cols.ravel(), Rows.ravel()
    alts = np.zeros_like(cols)

    # localize the points
    lon, lat, alt = bmod.localization(rows, cols, alts)
    
Then for projection

    # now project these points back in the burst
    rows_pred, cols_pred, i_pred = bmod.projection(lon, lat, alt)

## Burst registration

eos also provides the necessary tools to perform the estimation of the registration matrix 
from a digital elevation model and a physical sensor model. 

Suppose we have: 
*  `xml_folder`: The folder containing the sentinel1 xml metadata.
* `tiff_folder`: The folder containing the tiff sentinel1 images. 
* `output_folder`: The folder where we wish to save the outputs ( here only the dem is saved). 



        # list of our xmls 
        xml_paths = [os.path.join(xml_folder, p) for p in 
                        ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml',
                        's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml']
                    ]
        tiff_paths = [os.path.join(tiff_folder, p) for p in 
                        ['s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.tiff',
                        's1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.tiff']
                    ]

        # let's build s1m.Sentinel1Model instances 
        primary_s1m = s1m.Sentinel1Model(xml_paths[0])
        secondary_s1m = s1m.Sentinel1Model(xml_paths[1])

        # burst id in subswath
        # here, by "chance", the 4th burst is the same geographical location in both products
        burst_id = 3 

        # Now extract the needed metadata
        primary_burst_meta = eos.products.sentinel1.metadata.fill_meta(primary_s1m, bid = burst_id)
        secondary_burst_meta = eos.products.sentinel1.metadata.fill_meta(secondary_s1m, bid = burst_id)

        # Now instantiate burst_model instances for projection/localization
        primary_burst_model = eos.products.sentinel1.burst_model.burst_model_from_burst_meta(primary_burst_meta)
        secondary_burst_model = eos.products.sentinel1.burst_model.burst_model_from_burst_meta(secondary_burst_meta)

        # Now estimate the registration matrix 

        # get dem points
        x, y, raster, transform, crs = eos.sar.regist.dem_points(primary_burst_model,
                                                                source = 'SRTM30', 
                                                                datum = 'ellipsoidal', 
                                                                outfile = os.path.join(output_folder, 'dem.tif')
                                                                )
        # project in primary 
        row_primary, col_primary, _ = primary_burst_model.projection(x.ravel(), y.ravel(), raster.ravel(), crs = crs)

        # project in secondary and estimate registration
        A = eos.sar.regist.orbital_registration( row_primary, col_primary,
                                                secondary_burst_model, x, y, raster, crs
                                                ) 
        # Now read the secondary array 
        secondary_burst_array =  eos.products.sentinel1.io.read_burst(tiff_paths[1], secondary_burst_meta['burst_roi'])

        # resample the complex secondary burst
        _, _, w, h = primary_burst_meta['burst_roi']
        resampler = eos.products.sentinel1.burst_resamp.burst_resample_from_meta(secondary_burst_meta, dst_shape=(h, w),
                                                    matrix = A, degree=11)
        resampled_secondary_array = resampler.resample(secondary_burst_array)

Thus we have successfully resampled a complex secondary burst onto the primary burst frame. 
