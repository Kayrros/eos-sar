from __future__ import annotations

import numpy as np
import math

from eos.sar import utils


class Roi:

    col: int
    row: int
    w: int
    h: int

    def __init__(self, col, row, w, h):
        self.set_from_roi(col, row, w, h)

    def __repr__(self) -> str:
        return f"Roi(col={self.col}, row={self.row}, w={self.w}, h={self.h})"

    def copy(self) -> Roi:
        return Roi.from_roi_tuple(self.to_roi())

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Roi):
            return NotImplemented
        return self.to_roi() == o.to_roi()

    def set_from_roi(self, col: int, row: int, w: int, h: int) -> None:
        self.col = col
        self.row = row
        self.w = w
        self.h = h

    def set_from_roi_tuple(self, roi: tuple[int, int, int, int]) -> None:
        """
        Parameters
        ----------
        roi : tuple
            (col, row, w, h).
        """
        self.set_from_roi(*roi)

    def set_from_bounds_tuple(self, bounds: tuple[int, int, int, int]) -> None:
        self.set_from_roi_tuple(Roi.bounds_to_roi(bounds))

    @staticmethod
    def from_roi_tuple(roi: tuple[int, int, int, int]) -> Roi:
        """
        Parameters
        ----------
        roi : tuple
            (col, row, w, h).
        """
        return Roi(*roi)

    @staticmethod
    def from_bounds_tuple(bounds: tuple[int, int, int, int]) -> Roi:
        """
        Parameters
        ----------
        bounds: tuple.
            (col_min, row_min, col_max, row_max)
             col_max and row_max are included in the image.
        """
        return Roi(*Roi.bounds_to_roi(bounds))

    def obj_from_roi_tuple(self, roi: tuple[int, int, int, int], inplace: bool = False) -> Roi:
        """
        Parameters
        ----------
        roi : tuple
            (col, row, w, h).
        """
        if inplace:
            self.set_from_roi_tuple(roi)
            return self
        else:
            return Roi.from_roi_tuple(roi)

    def obj_from_bounds_tuple(self, bounds: tuple[int, int, int, int], inplace: bool = False) -> Roi:
        """
        Parameters
        ----------
        bounds: tuple.
            (col_min, row_min, col_max, row_max)
             col_max and row_max are included in the image.
        """
        if inplace:
            self.set_from_bounds_tuple(bounds)
            return self
        else:
            return Roi.from_bounds_tuple(bounds)

    @staticmethod
    def roi_to_bounds(roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Convert roi representation to bound representation

        Parameters
        ----------
        roi : tuple
            (col, row, w, h).

        Returns
        -------
        bounds: tuple.
            (col_min, row_min, col_max, row_max)
            col_max and row_max are included in the image.
        """
        col, row, w, h = roi
        return (col, row, col + w - 1, row + h - 1)

    @staticmethod
    def bounds_to_roi(bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Convert bounds to roi representation.

        Parameters
        ----------
        bounds: tuple.
            (col_min, row_min, col_max, row_max)
             col_max and row_max are included in the image.
        Returns
        -------
        roi : tuple
            (col, row, w, h).

        """
        col, row, col_max, row_max = bounds
        w = col_max - col + 1
        h = row_max - row + 1
        return (col, row, w, h)

    @staticmethod
    def points_to_bbox(rows, cols) -> tuple[int, int, int, int]:
        """
        Derive bounds from a set of points

        Parameters
        ----------
        rows : float 1darray
            Row coordinates.
        cols : float 1darray
            Column coordinates.

        Returns
        -------
        bbox_bounds : tuple
            (col_min, row_min, col_max, row_max).

        """
        # take the integer bounding box
        row_min = math.floor(min(rows))
        row_max = math.ceil(max(rows))
        col_min = math.floor(min(cols))
        col_max = math.ceil(max(cols))
        bbox_bounds = (col_min, row_min, col_max, row_max)
        return bbox_bounds

    def to_roi(self) -> tuple[int, int, int, int]:
        return (self.col, self.row, self.w, self.h)

    def to_bounds(self) -> tuple[int, int, int, int]:
        return Roi.roi_to_bounds(self.to_roi())

    def get_shape(self) -> tuple[int, int]:
        return (self.h, self.w)

    def get_origin(self) -> tuple[int, int]:
        return self.col, self.row

    def to_bounding_points(self, homogeneous: bool = False):
        """
        Convert to its bounding points representation.

        Parameters
        ----------
        homogeneous : bool, optional
            If True, homogeneous coordinates points are returned,
            i.e. a additional entry equal to 1 is added . The default is False.

        Returns
        -------
        points : ndarray (N x 4) N=2 if homogeneous = False, N=3 otherwise
            The bounding points. Each column is a point (row, col).

        """
        col_min, row_min, col_max, row_max = self.to_bounds()

        # get the boundary points of the input roi
        points = np.array([[row_min, row_min, row_max, row_max],
                           [col_min, col_max, col_max, col_min]])
        if homogeneous:
            points = np.vstack((points, np.ones(4)))
        return points

    def warp(self, matrix, inplace: bool = False) -> Roi:
        # input bounding points
        bound_points = self.to_bounding_points(homogeneous=True)
        # warp points using the matrix
        rows_out, cols_out = matrix.dot(bound_points)[:2]
        # output bounding points
        out_bounds = Roi.points_to_bbox(rows_out, cols_out)
        # reset or get new Roi instance
        return self.obj_from_bounds_tuple(out_bounds,
                                          inplace=inplace)

    def add_margin(self, margin: int = 0, inplace: bool = False) -> Roi:
        """
        Add a margin in pixels on the boundary of a roi.

        Parameters
        ----------
        margin : int, optional
            Margin in pixels to add to the roi. The default is 0.

        Returns
        -------
        out_roi :

        """
        margin = int(margin)
        col, row, w, h = self.to_roi()
        out_roi = (col - margin, row - margin, w + 2 * margin, h + 2 * margin)
        return self.obj_from_roi_tuple(out_roi, inplace=inplace)

    def add_custom_margin(self,
                          custom_margin: tuple[tuple[int, int], tuple[int, int]],
                          inplace=False) -> tuple[Roi, Roi]:
        """
        Add custom margin for all directions of a roi.

        Parameters
        ----------
        custom_margin : tuple[tuple]
            ((up, down), (left, right)).
        inplace : bool, optional
            If True, perform the modification inplace. The default is False.

        Returns
        -------
        out_roi: Roi
            Roi with added margin.
        roi_in_padded_output: Roi
            Roi location in padded output

        """
        col, row, w, h = self.to_roi()
        (up, down), (left, right) = custom_margin
        out_roi = (col - left, row - up, w + left + right, h + up + down)
        return self.obj_from_roi_tuple(out_roi, inplace=inplace), Roi(left, up, w, h)

    def assert_valid(self, parent_shape: tuple[int, int]) -> None:
        h_parent, w_parent = parent_shape
        col_child_min, row_child_min, col_child_max, row_child_max = self.to_bounds()
        msg = "Roi outside of parent"
        assert (col_child_max < w_parent), msg
        assert (row_child_max < h_parent), msg
        assert (col_child_min >= 0), msg
        assert (row_child_min >= 0), msg

    def make_valid(self, parent_shape: tuple[int, int], inplace: bool = False) -> Roi:
        """
        If the child roi is not within the boundaries of the parent image dimension,
        modify it to satisfy the condition.

        Parameters
        ----------
        parent_shape : tuple
            (h, w) shape of the parent image.

        Returns
        -------
        adapted_roi : Roi
            region of interest that lies within the parent shape.

        """""
        h, w = parent_shape
        parent_roi = Roi(0, 0, w, h)
        return self.clip(parent_roi, inplace=inplace)

    def clip(self, parent_roi: Roi, inplace: bool = False) -> Roi:
        """
        If the child roi is not within the boundaries of the parent image dimension,
        modify it to satisfy the condition.

        Parameters
        ----------
        parent_roi : Roi

        Returns
        -------
        adapted_roi : Roi
            region of interest that lies within the parent shape.

        """""
        col_parent_min, row_parent_min, col_parent_max, row_parent_max = parent_roi.to_bounds()
        col_child_min, row_child_min, col_child_max, row_child_max = self.to_bounds()

        # take min, max with image boundary
        col_min = max(col_child_min, col_parent_min)
        col_max = min(col_child_max, col_parent_max)
        row_min = max(row_child_min, row_parent_min)
        row_max = min(row_child_max, row_parent_max)

        if (row_max < row_min) or (col_max < col_min):
            return self.obj_from_roi_tuple((0, 0, 0, 0), inplace=inplace)

        else:
            out_bounds = (col_min, row_min, col_max, row_max)
            # reset or get new Roi instance
            return self.obj_from_bounds_tuple(out_bounds, inplace=inplace)

    def intersects_roi(self, other_roi: Roi) -> bool:
        '''
        Check whether other_roi and self intersect.

        Parameters
        ----------
        other_roi : Roi

        Returns
        -------
        are_intersecting : bool
            True if the two ROIs are intersecting, False otherwise.
        '''
        clipped = self.clip(other_roi)
        return clipped.w > 0 and clipped.h > 0

    def warp_valid_roi(self,
                       input_parent_shape: tuple[int, int],
                       output_parent_shape: tuple[int, int],
                       matrix,
                       margin: int = 0,
                       inplace: bool = False) -> Roi:
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
        out_valid_roi : Roi
            Roi validated against the dimensions of the output image
            and padded bounding box of the warped roi.
        """

        # assert input roi within parent boundaries
        # if inplace=True, roi_obj is the same as self
        roi_obj = self.make_valid(parent_shape=input_parent_shape,
                                  inplace=inplace)

        # transform roi
        roi_obj.warp(matrix=matrix, inplace=True)

        # add a margin in pixels in all directions
        roi_obj.add_margin(margin=margin, inplace=True)

        # make valid output roi within parent boundaries
        roi_obj.make_valid(parent_shape=output_parent_shape,
                           inplace=True)
        return roi_obj

    def translate_roi(self, col: int, row: int, inplace: bool = False) -> Roi:
        """
        Translate a region of interest.

        Parameters
        ----------
        col : int
            column translation.
        row : int
            row translation.

        Returns
        -------

        """
        c, r, w, h = self.to_roi()
        out_roi = (c + col, r + row, w, h)
        return self.obj_from_roi_tuple(out_roi, inplace=inplace)

    def crop_array(self, arr):
        col_min, row_min, col_max, row_max = self.to_bounds()
        arr_cropped = arr[row_min:row_max + 1, col_min:col_max + 1]
        return arr_cropped

    def get_meshgrid(self):
        col, row, w, h = self.to_roi()
        cols_grid, rows_grid = np.meshgrid(np.arange(col, col + w),
                                           np.arange(row, row + h))
        return cols_grid, rows_grid

    def contains(self, cols, rows):
        """
        Compute mask on points that are within the roi.

        Parameters
        ----------
        cols : array
            Colmuns.
        rows : array
            Rows.

        Returns
        -------
        mask : array(boolean)
            Mask of points in the roi.

        """
        col_min, row_min, col_max, row_max = self.to_bounds()

        # get a mask on the points that are within the roi
        mask = np.logical_and(utils.arr_in_interval(cols, col_min, col_max),
                              utils.arr_in_interval(rows, row_min, row_max)
                              )
        return mask
