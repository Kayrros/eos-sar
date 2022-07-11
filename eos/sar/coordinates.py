"""Coordinates conversion between image (row/column denoted as `row` \
    and `col`) and sar (azimuth time and range denoted as `azt` and `rng`)."""

from eos.sar import const
from eos.sar.srgr import SRGRConverter


class SLCCoordinateMixin:

    first_row_time: float
    first_col_time: float
    azimuth_frequency: float
    range_frequency: float

    def to_azt(self, row):
        azt = row / self.azimuth_frequency + self.first_row_time
        return azt

    def to_rng(self, col, azt=None):
        rng = (col / self.range_frequency + self.first_col_time) * \
            const.LIGHT_SPEED_M_PER_SEC / 2
        return rng

    def to_azt_rng(self, row, col):
        azt = self.to_azt(row)
        rng = self.to_rng(col)
        return azt, rng

    def to_row(self, azt):
        row = (azt - self.first_row_time) * self.azimuth_frequency
        return row

    def to_col(self, rng, azt=None):
        col = (2 * rng / const.LIGHT_SPEED_M_PER_SEC - self.first_col_time) * \
            self.range_frequency
        return col

    def to_row_col(self, azt, rng):
        row = self.to_row(azt)
        col = self.to_col(rng)
        return row, col


class GRDCoordinateMixin:

    first_row_time: float
    azimuth_time_interval: float
    range_pixel_spacing: float
    srgr: SRGRConverter

    def to_azt(self, row):
        azt = row * self.azimuth_time_interval + self.first_row_time
        return azt

    def to_rng(self, col, azt):
        gr = col * self.range_pixel_spacing
        rng = self.srgr.gr_to_rng(gr, azt)
        return rng

    def to_azt_rng(self, row, col):
        azt = self.to_azt(row)
        rng = self.to_rng(col, azt)
        return azt, rng

    def to_row(self, azt):
        row = (azt - self.first_row_time) / self.azimuth_time_interval
        return row

    def to_col(self, rng, azt):
        gr = self.srgr.rng_to_gr(rng, azt)
        col = gr / self.range_pixel_spacing
        return col

    def to_row_col(self, azt, rng):
        row = self.to_row(azt)
        col = self.to_col(rng, azt)
        return row, col
