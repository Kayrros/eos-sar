"""Coordinates conversion between image (row/column denoted as `row` \
    and `col`) and sar (azimuth time and range denoted as `azt` and `rng`)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eos.sar import const
from eos.sar.srgr import SRGRConverter

Arrayf32 = NDArray[np.float32]


@dataclass(frozen=True)
class SLCCoordinate:
    first_row_time: float
    first_col_time: float
    azimuth_frequency: float
    range_frequency: float

    def to_azt(self, row: ArrayLike) -> Arrayf32:
        row = np.asarray(row)
        azt = row / self.azimuth_frequency + self.first_row_time
        return azt

    def to_rng(self, col: ArrayLike, azt: Optional[ArrayLike] = None) -> Arrayf32:
        col = np.asarray(col)
        rng = (
            (col / self.range_frequency + self.first_col_time)
            * const.LIGHT_SPEED_M_PER_SEC
            / 2
        )
        return rng

    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        azt = self.to_azt(row)
        rng = self.to_rng(col)
        return azt, rng

    def to_row(self, azt: ArrayLike) -> Arrayf32:
        azt = np.asarray(azt)
        row = (azt - self.first_row_time) * self.azimuth_frequency
        return row

    def to_col(self, rng: ArrayLike, azt: Optional[ArrayLike] = None) -> Arrayf32:
        rng = np.asarray(rng)
        col = (
            2 * rng / const.LIGHT_SPEED_M_PER_SEC - self.first_col_time
        ) * self.range_frequency
        return col

    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        row = self.to_row(azt)
        col = self.to_col(rng)
        return row, col


@dataclass(frozen=True)
class GRDCoordinate:
    # NOTE: the function signature is slightly different than in SLCCoordinate
    # because the azt is required for the to_col, not optional

    first_row_time: float
    azimuth_time_interval: float
    range_pixel_spacing: float
    srgr: SRGRConverter

    def to_azt(self, row: ArrayLike) -> Arrayf32:
        row = np.asarray(row)
        azt = row * self.azimuth_time_interval + self.first_row_time
        return azt

    def to_rng(self, col: ArrayLike, azt: ArrayLike) -> Arrayf32:
        col = np.asarray(col)
        gr = col * self.range_pixel_spacing
        rng = self.srgr.gr_to_rng(gr, azt)
        return rng

    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        azt = self.to_azt(row)
        rng = self.to_rng(col, azt)
        return azt, rng

    def to_row(self, azt: ArrayLike) -> Arrayf32:
        azt = np.asarray(azt)
        row = (azt - self.first_row_time) / self.azimuth_time_interval
        return row

    def to_col(self, rng: ArrayLike, azt: ArrayLike) -> Arrayf32:
        gr = self.srgr.rng_to_gr(rng, azt)
        col = gr / self.range_pixel_spacing
        return col

    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf32, Arrayf32]:
        row = self.to_row(azt)
        col = self.to_col(rng, azt)
        return row, col
