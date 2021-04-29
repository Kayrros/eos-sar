# eos-sar

This package provides access to some generic sar processing algorithms. 

Currently, algorithms specific to **Sentinel1** bursts have been implemented. 

## Physical sensor model 

This allows the user to project a 3D point into the burst, and vice-versa. 

For localization:

    # create a Sentinel1Model
    s1model = s1m.Sentinel1Model(xml=xml_path)
    # create a Sentinel1BurstModel
    bmod = eos.products.sentinel1.burst_model.burst_model_from_s1m(
                    s1model, burst=burst_id,
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
    lon, lat, alt = bmod.localization(rows, cols, alts)`
    Then, for projection: 

Then for projection

    # now project these points back in the burst
    rows_pred, cols_pred, i_pred = bmod.projection(lon, lat, alt)

## Burst registration

eos also provides the necessary tools to perform the estimation of the registration matrix 
from a digital elevation model and a physical sensor model. 

First we need to instantiate the primary and secondary burst models: 

    # set up burst models
    primary_burst_model = eos.products.sentinel1.burst_model.burst_model_from_s1m(s1model=models[0],burst=burst_id)
    secondary_burst_model = eos.products.sentinel1.burst_model.burst_model_from_s1m(s1model=models[1],burst=burst_id)

To estimate an Affine resampling matrix:
 
    # get dem  
    x, y, raster, transform, crs = eos.sar.regist.dem_points(primary_burst_model,
                                                            source = 'SRTM30', 
                                                            datum = 'ellipsoidal', 
                                                            outfile = os.path.join(base_path, 'dem.tif')
                                                            )
    # project in primary 
    row_primary, col_primary, _ = primary_burst_model.projection(x.ravel(), y.ravel(), raster.ravel(), crs = crs)

    # project in secondary and compute matrix
    A = eos.sar.regist.orbital_registration( row_primary, col_primary,
                                            secondary_burst_model, x, y, raster, crs
                                            ) 
                                    
Then we need to read the secondary burst array

    secondary_burst_array = eos.products.sentinel1.io.read_burst(measurements_path[1], secondary_burst_model)

Then to resample the complex burst 

    # resample the complex secondary burst
    _, _, w, h = primary_burst_model.burst_roi
    resampler = eos.products.sentinel1.burst_resamp.s1resample_from_s1m(models[1], burst=burst_id, dst_shape=(h, w),
                                    matrix=A, degree=11)
    resampled_secondary_array = resampler.resample(secondary_burst_array)





