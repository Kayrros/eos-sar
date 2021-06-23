import numpy as np
import math 

def warp_roi(roi, matrix): 
    """
    Warp a roi using the registration matrix. 

    Parameters
    ----------
    roi : tuple
        (col, row, w, h).
    matrix: ndarray (3,3)
        warping matrix to apply
    Returns
    -------
    out_roi : tuple
        (col, row, w, h) warped region of interest.

    """
    col_in, row_in, w_in, h_in = roi
    col_max = col_in + w_in - 1
    row_max = row_in + h_in - 1 
    # get the boundary points of the input roi 
    bound_points = np.array([[row_in, col_in, 1], 
                             [row_in, col_max, 1], 
                             [row_max, col_max, 1], 
                             [row_max, col_in, 1]]).T
    
    # warp points using the matrix
    rows_out, cols_out = matrix.dot(bound_points)[:2]
    
    # take the integer bounding box
    row_min = math.floor(min(rows_out))
    row_max = math.ceil(max(rows_out))
    col_min = math.floor(min(cols_out))
    col_max = math.ceil(max(cols_out))
    
    # construct output roi
    h_out = row_max - row_min + 1
    w_out = col_max - col_min + 1
    out_roi = (col_min, row_min, w_out, h_out)

    return out_roi

def add_margin(roi, margin=0): 
    """
    Add a margin in pixels on the boundary of a roi. 

    Parameters
    ----------
    roi : tuple
        (col, row, w, h).
    margin : int, optional
        Margin in pixels to add to the roi. The default is 0.

    Returns
    -------
    out_roi : tuple
        (col, row, w, h) the expanded roi.

    """
    margin = int(margin)
    col, row, w, h = roi 
    out_roi = (col - margin, row - margin, w + 2 * margin, h + 2 * margin) 
    return out_roi

def make_valid_roi(parent_shape, child_roi): 
    """
    If the child roi is not within the boundaries of the parent image dimension, 
    modify it to satisfy the condition. 

    Parameters
    ----------
    parent_shape : tuple
        (h, w) shape of the parent image.
    child_roi : tuple
        (col, row, w, h) the region of interest in the parent image coordinates.

    Returns
    -------
    adapted_roi : tuple
        (col, row, w, h) region of interest that lies within the parent shape.

    """""
    h_p, w_p = parent_shape
    col_c, row_c, w_c, h_c = child_roi
    
    # take min, max with image boundary 
    col_min = max(col_c, 0) 
    col_max = min(col_c + w_c , w_p) 
    row_min = max(0, row_c) 
    row_max = min(row_c + h_c , h_p)
    
    # reconstruct roi
    adapted_roi = (col_min, row_min, col_max - col_min, row_max - row_min) 
    
    return adapted_roi

def warp_valid_rois(in_roi, input_parent_shape, output_parent_shape,  
                    matrix, margin=0): 
    """
    Warp an input roi while making sure it is valid to an output roi, add margin
    and make sure it is valid. 

    Parameters
    ----------
    in_roi : tuple
        (col, row, w, h) of the input roi in the input image.
    input_parent_shape : tuple
        (h, w) of the input image that contains the input roi.
    output_parent_shape : tuple
        (h, w) of the output image that will contain the output roi.
    matrix : ndarray (3,3)
        Matrix that will be used to warp from input parent frame
        to output parent  frame.
    margin : int, optional
        Margin in pixels to padd the bounding box of the warped roi. 
        The default is 0.

    Returns
    -------
    out_valid_roi : tuple
        (col, row, w, h) Validated against the dimensions of the output image
        and padded bounding box of the warped roi.

    """
    
    # assert input roi within parent boundaries
    in_valid_roi = make_valid_roi(input_parent_shape , in_roi)
    
    # transform roi, we get the bounding box 
    out_roi = warp_roi(in_valid_roi, matrix=matrix  )
    
    # add a margin in pixels in all directions
    out_margin_roi = add_margin(out_roi, margin=margin)
    
    # make valid output roi within parent boundaries
    out_valid_roi = make_valid_roi(output_parent_shape, out_margin_roi)
    
    return out_valid_roi

def translate_roi(roi, col, row): 
    """
    Translate a region of interest. 

    Parameters
    ----------
    roi : tupple
        (col, row, w, h) region of interest.
    col : int
        column translation.
    row : int
        row translation.

    Returns
    -------
    out_roi: tuple
        (col, row, w, h) translated roi

    """
    out_roi = (roi[0] + col, 
               roi[1] + row, 
               roi[2], 
               roi[3])
    return out_roi
        