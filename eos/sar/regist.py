"""Generic registration functions."""
import numpy as np
import cv2
import abc
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
        print("ERROR: estimation.affine_transformation\
              needs 3 correspondences")
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


def dem_points(geometry, source="SRTM30", datum="ellipsoidal",
               outfile=None):
    """Query dem points.

    Parameters
    ----------
    geometry : list of tuple (lon,lat)
        Geometry of the primary image, one point per corner of the image
    source : str
        DEM source "SRTM30" (default), "TDM90", "SRTM90" or "SRTM90-CGIAR-CSI"
    datum : str
        "ellipsoidal" (height above WGS84 ellipsoid, default), or
        "orthometric" (height above EGM96 geoid).
    outfile : string, optional
        Path to save the dem if passed as argument.
        The default is None.

    Returns
    -------
    x : ndarray
        x coordinate (longitude if crs epsg4326).
    y : ndarray
        y coordinate (latitude if crs epsg4326).
    raster : ndarray
        Dem altitude array.
    transform : affine.Affine
        Raster transform to crs coordinates
    crs : Any crs type accepted by pyproj
        crs of the returned points.

    Notes
    -----
    Due to multidem current implementation, only datum == "ellipsoidal"
    is accepted at the moment

    """
    assert datum == "ellipsoidal", "Multidem\
        only supports crs for ellipsoidal datum"
    # geometry of the query
    lons = [P[0] for P in geometry]
    lats = [P[1] for P in geometry]
    bounds = (min(lons), min(lats), max(lons), max(lats))
    # query for dem
    raster, transform, crs = multidem.crop(bounds,
                                           source=source,
                                           datum=datum)
    if outfile:
        # save dem
        multidem.write_crop_to_file(raster, transform, crs, outfile)

    # get dem points in crs
    col, row = np.meshgrid(
        np.arange(raster.shape[1]), np.arange(raster.shape[0]))
    col = col.ravel()
    row = row.ravel()

    # Add 0.5 for pixel is area
    col = col + 0.5
    row = row + 0.5
    # to earth coordinates
    x, y = transform * (col, row)
    # reshape
    x = x.reshape(raster.shape)
    y = y.reshape(raster.shape)
    return x, y, raster, transform, crs


def orbital_registration(row_primary, col_primary, secondary_model,
                         x, y, raster, crs):
    """Compute registration matrix between primary and secondary model.

    Parameters
    ----------
    row_primary : ndarray
        Row projection of the x, y, raster points in the primary burst.
    col_primary : ndarray
        Col projection of the x, y, raster points in the primary burst..
    secondary_model : eos.sar.model.SensorModel
        Sensor model for the secondary image.
    x : ndarray
        x coordinate of 3D point on DEM (longitude if crs epsg4326).
    y : ndarray
        y coordinate of 3D point on DEM (latitude if crs epsg4326).
    raster : ndarray
        DEM altitude array of 3D point.
    crs : Any crs type accepted by pyproj
        crs of the dem points.

    Returns
    -------
    matrix : ndarray
        Affine registration matrix from primary_model coordinates
        to secondary_model coordinates.

    """
    # project in the secondary image
    # some points will exceed the burst bounds,
    # however this does not harm registration
    row_secondary, col_secondary, _ = secondary_model.projection(
        x.ravel(), y.ravel(), raster.ravel(), crs=crs)

    # fit affine matrix
    primary_pts = np.column_stack([row_primary.ravel(), col_primary.ravel()])
    secondary_pts = np.column_stack([row_secondary, col_secondary])

    # affine matrix from primary to secondary burst
    matrix = affine_transformation(primary_pts, secondary_pts)

    return matrix


def apply_affine(src_array, matrix, destination_array_shape):
    """Resamples an image with the provided matrix using Lanczos interpolation.

    Parameters
    ----------
    src_array : ndarray
        Image to be resampled, single band, complex64 support available.
    matrix : 3x3 ndarray
        Inverse transform matrix from destination to source frame.
    destination_array_shape : tuple
        (h, w) of destination image frame.

    Returns
    -------
    resampled : ndarray
        Resampled image.

    """
    # parameters for warpAffine
    dsize = destination_array_shape[::-1]
    flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    M = np.zeros((2, 3))
    # swap x and y for opencv
    M[0, 0] = matrix[1, 1]
    M[0, 1] = matrix[1, 0]
    M[0, 2] = matrix[1, 2]
    M[1, 0] = matrix[0, 1]
    M[1, 1] = matrix[0, 0]
    M[1, 2] = matrix[0, 2]

    if src_array.dtype == np.complex64:
        h, w = src_array.shape
        img = src_array.copy(order='C')  # ensures contiguous data and C order
        img = img.view(dtype=np.float32)  # now data is float
        img = img.reshape([h, w, 2])  # now data is float2

        resampled = cv2.warpAffine(img, M, dsize, flags=flags, borderValue=np.nan)

        h, w = destination_array_shape
        resampled.reshape((h, w*2))
        resampled = resampled.view(dtype=np.complex64).squeeze()

    else:
        resampled = cv2.warpAffine(src_array, M, dsize, flags=flags, borderValue=np.nan)

    return resampled


class ComplexResample(abc.ABC):
    """ComplexResample is an abstract class that defines the expected method\
        of any complex resampling mechanism. It is expected that this abstract\
        will be implemented for each SAR satellite,\
        and for each satellite mode."""

    dst_shape: tuple
    matrix: np.ndarray

    @abc.abstractmethod
    def deramp(self, src_array):
        # The deramping should work on the regular src grid
        # and estimate a phase for each point in this grid
        pass

    @abc.abstractmethod
    def reramp(self, dst_array):
        # The reramping should work on the irregular matrix*dst_grid
        # and should yield a phase for each point in this grid
        pass

    def resample(self, src_array):
        """.

        Parameters
        ----------
        src_array : ndarray
            src image.

        Returns
        -------
        ndarray
            Resampled complex image.
        """
        # deramp
        deramped = self.deramp(src_array)

        # resample
        dst_array = apply_affine(deramped, self.matrix, self.dst_shape)

        # reramp
        return self.reramp(dst_array)

def change_resamp_mat_orig(row_dst, col_dst, row_src, col_src, A):
    """
    Adapts a resampling matrix to new origins at the source and the destination. 

    Parameters
    ----------
    row_src : float
        row at the new source origin w.r.t the old origin.
    col_src : float
        col at the new source origin w.r.t the old origin.
    row_dst : float
        row at the new destination origin w.r.t the old origin.
    col_dst : float
        col at the new destination origin w.r.t the old origin.
    A : (3,3) ndarray
        Inverse resampling matrix A.dst_coords = src_coords at the old origins.

    Returns
    -------
    (3,3) ndarray
        Adapted affine matrix

    """
    T_dst_inv = np.eye(3)
    T_dst_inv[0,2] = row_dst
    T_dst_inv[1,2] = col_dst
    T_src = np.eye(3)
    T_src[0,2] = - row_src
    T_src[1,2] = - col_src 
    return T_src.dot(A.dot(T_dst_inv))
