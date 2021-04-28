import numpy as np
from scipy import ndimage
import multidem


def affine_transformation(src, dst):
    """Estimate a 2D affine transform from a list of point correspondences.


    Parameters
    ----------
    src : Nx2 ndarray
        Source points.
    dst : Nx2 ndarray
        Destination points.

    Returns
    -------
    A: 3x3 ndarray
        Affine transform that maps the points of src onto the points of dst
        in homogeneous coordinates.

    Notes
    -----
    This function implements the Gold-Standard algorithm for estimating an
    affine transform, described in Hartley & Zisserman page 130 (second
    edition).

    """

    # check that there are at least 3 points
    if len(src) < 3:
        print("ERROR: estimation.affine_transformation needs 3 correspondences")
        return np.eye(3)

    # translate the input points so that the centroid is at the origin.
    tsrc = -np.mean(src, 0)
    tdst = -np.mean(dst, 0)
    src = src + tsrc
    dst = dst + tdst

    # compute the Nx4 matrix A
    A = np.hstack((src, dst))

    # two singular vectors corresponding to the two largest singular values of
    # matrix A. See Hartley and Zissermann for details.  These are the first
    # two lines of matrix V (because np.linalg.svd returns V^T)
    _, _, V = np.linalg.svd(A, full_matrices=False)
#    print(S)
    v1 = V[0, :]
    v2 = V[1, :]

    # compute blocks B and C, then H
    tmp = np.vstack((v1, v2)).T
    assert(np.shape(tmp) == (4, 2))
    B = tmp[0:2, :]
    C = tmp[2:4, :]
    H = np.dot(C, np.linalg.inv(B))

    # return A
    A = np.eye(3)
    A[0:2, 0:2] = H
    A[0:2, 2] = np.dot(H, tsrc) - tdst
    return A


def dem_points(primary_model, multidem_config,
               grid_len=None, outfile=None):
    """Query dem points.


    Parameters
    ----------
    primary_model : eos.sar.model.SensorModel
        Sensor model for the primary image.
    multidem_config : dict
        Configuration of multidem querys.
        Keys should be ['source', 'datum', 'interpolation'].
    grid_len : int, optional
        Number of points queried from the dem. 
        If given, a sparse meshgrid of points of grid_len x grid_len is used. 
        Otherwise, all the dem needs to be downloaded and then all the 
        dem points will be returned. The default is None.
    outfile : string, optional
        Path to save the dem if passed as argument.
        Ignored if grid_len given. The default is None.

    Returns
    -------
    x : ndarray
        x coordinate.
    y : ndarray
        y coordinate.
    alt : ndarray
        Altitude.
    crs : Any crs type accepted by pyproj
        crs of the returned points.

    """
    # burst approx bbox
    lons = [P[0] for P in primary_model.approx_geom]
    lats = [P[1] for P in primary_model.approx_geom]
    bounds = (min(lons), min(lats), max(lons), max(lats))

    if grid_len:

        # get grid_len points equidistant between each other and bounds
        lat_arr = np.linspace(bounds[1], bounds[3], grid_len)
        lon_arr = np.linspace(bounds[0], bounds[2], grid_len)
        x, y = np.meshgrid(lon_arr, lat_arr)

        x = x.ravel()
        y = y.ravel()

        alt = multidem.elevation(x, y, source=multidem_config['source'],
                                 interpolation=multidem_config['interpolation'],
                                 datum=multidem_config['datum'])
        # crs = rasterio_crs("EPSG:4979") if datum == "ellipsoidal" else rasterio_crs("EPSG:4326+5773")
        # TODO: check https://github.com/pyproj4/pyproj/issues/757
        crs = 'epsg:4326'
    else:
        # query for dem
        raster, transform, crs = multidem.crop(bounds,
                                               source=multidem_config['source'],
                                               datum=multidem_config['datum'])
        if outfile:
            # save dem
            multidem.write_crop_to_file(raster, transform, crs, outfile)

        # get dem points in crs
        col, row = np.meshgrid(
            np.arange(raster.shape[1]), np.arange(raster.shape[0]))
        col = col.ravel()
        row = row.ravel()

        alt = raster.ravel()
        # Add 0.5 for pixel is area
        col = col + 0.5
        row = row + 0.5
        # to earth coordinates
        x, y = transform * (col, row)

    return x, y, alt, crs


def orbital_registration(primary_model, secondary_model, multidem_config,
                         grid_len=None, outfile=None):
    """Compute registration matrix between primary and secondary model. 


    Parameters
    ----------
    primary_model : eos.sar.model.SensorModel
        Sensor model for the primary image.
    secondary_model : eos.sar.model.SensorModel
        Sensor model for the secondary image.
    multidem_config : dict
        Configuration of multidem querys.
        Keys should be ['source', 'datum', 'interpolation'].
    grid_len : int, optional
        Number of projected points used during the registration.
        If given, a sparse meshgrid of points of grid_len x grid_len is used. 
        Otherwise, all the dem needs to be downloaded and then all the 
        dem points will be projected. The default is None.
    outfile : string, optional
        Path to save the dem if passed as argument.
        Ignored if grid_len given. The default is None.

    Returns
    -------
    matrix : ndarray
        Affine registration matrix from primary_model coordinates
        to secondary_model coordinates.

    Notes
    -----
    Due to multidem current implementation, only  multidem_config['datum'] == "ellipsoidal"
    is accepted at the moment

    """

    assert multidem_config['datum'] == "ellipsoidal", "multidem seems to only support this now"
    x, y, alt, crs = dem_points(primary_model, multidem_config, grid_len=grid_len,
                                outfile=outfile)

    # project in the two images
    # some points will exceed the burst bounds, however this does not harm registration
    row_primary, col_primary, _ = primary_model.projection(x, y, alt, crs=crs)
    row_secondary, col_secondary, _ = secondary_model.projection(
        x, y, alt, crs=crs)

    # fit affine matrix
    primary_pts = np.column_stack([row_primary, col_primary])
    secondary_pts = np.column_stack([row_secondary, col_secondary])

    # affine matrix from primary to secondary burst
    matrix = affine_transformation(primary_pts, secondary_pts)

    return matrix


def apply_affine(matrix, src_array, destination_array_shape, order=3):
    """Resamples an image with the provided matrix using the spline interpolation


    Parameters
    ----------
    matrix : 3x3 ndarray
        Inverse transform matrix from destination to source frame.
    src_array : ndarray
        Image to be resampled, single band, complex64 support available.
    destination_array_shape : tuple
        (h, w) of destination image frame.
    order : int, optional
        order of the spline interpolation used. The default is 3.

    Returns
    -------
    resampled : ndarray
        Resampled image.

    """
    if src_array.dtype == np.complex64:
        
        h, w = src_array.shape
        
        crop = src_array.copy(order='C')  # ensures contiguous data and C order
        crop = crop.view(dtype=np.float32)  # now data is float
        crop = crop.reshape([h, w, 2])  # now data is float2
        
        crop1 = ndimage.affine_transform(
            input=crop[:, :, 0], matrix=matrix, order=order,
            output_shape=destination_array_shape, cval=np.nan)
        
        crop2 = ndimage.affine_transform(
            input=crop[:, :, 1], matrix=matrix, order=order,
            output_shape=destination_array_shape, cval=np.nan)
        
        h, w = destination_array_shape
        
        resampled = np.zeros((h, w, 2), dtype=np.float32)
        resampled[:, :, 0] = crop1
        resampled[:, :, 1] = crop2
        
        resampled.reshape((h, w*2))
        resampled = resampled.view(dtype=np.complex64).squeeze()
    
    else:
        resampled = ndimage.affine_transform(
            input=src_array, matrix=matrix, order=order,
            output_shape=destination_array_shape, cval=np.nan)
    
    return resampled	