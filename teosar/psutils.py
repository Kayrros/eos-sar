from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import scipy

def wrap(phi: NDArray[float]) -> NDArray[float]:
    return (phi + np.pi)%(2 * np.pi) - np.pi


@dataclass(frozen=True)
class Window:
    col: int
    row: int
    w: int
    h: int

    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.w)

    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.h)

    def get_slices(self) -> tuple[slice]:
        return self.row_slice(), self.col_slice()

    def get_mask(self, parent_shape: tuple[int]) -> NDArray[bool]:
        mask = np.zeros(parent_shape, dtype=bool)
        mask[self.get_slices()] = True
        return mask

    def add_margin(self, margin_h, margin_w)-> Window:
        edge_col = self.col - margin_w
        edge_row = self.row - margin_h
        nh = 2 * margin_h + self.h
        nw = 2 * margin_w + self.w
        return Window(edge_col, edge_row, nw, nh)

    def get_corners(self ,*, closed=False) -> list[list[float]]:
        col_min = self.col
        col_max = self.col + self.w - 1
        row_min = self.row
        row_max = self.row + self.h - 1
        corners = [[col_min, row_min], [col_max, row_min], [col_max, row_max], [col_min, row_max]]
        if closed:
            # close the loop
            corners.append(corners[0])
        return corners

    def make_valid(self, parent_shape: tuple[int]) -> Window:
        p_h, p_w = parent_shape
        col = max(self.col, 0)
        row = max(self.row, 0)
        col_max = min(self.col + self.w - 1, p_w - 1)
        row_max = min(self.row + self.h - 1, p_h - 1)
        return Window(col, row, col_max - col + 1, row_max - row + 1)


def sparse_data_to_raster(sparse_data, row_ps, col_ps, parent_shape: tuple[int, int]):
        data_full = np.full(parent_shape, np.nan, dtype=sparse_data.dtype)
        data_full[row_ps, col_ps] = sparse_data
        return data_full

def dense_mask_to_sparse(mask):
    """
    coo enables to iterate other the points that
    pass the mask like this:
    for (x, y) in zip(mask.col, mask.row)
    """
    return scipy.sparse.coo_array(mask)
