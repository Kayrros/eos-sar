"""Coordinates conversion between image (row/column denoted as `row` \
    and `col`) and sar (azimuth time and range denoted as `azt` and `rng`)."""

from eos.sar import const
from eos.sar.srgr import SRGRConverter


class SLCCoordinateMixin:

    first_row_time: float
    first_col_time: float
    azimuth_frequency: float
    range_frequency: float

    def to_azt_rng(self, row, col):
        azt = row / self.azimuth_frequency + self.first_row_time
        rng = (col / self.range_frequency + self.first_col_time) * \
            const.LIGHT_SPEED_M_PER_SEC / 2
        return azt, rng

    def to_row_col(self, azt, rng):
        row = (azt - self.first_row_time) * self.azimuth_frequency
        col = (2 * rng / const.LIGHT_SPEED_M_PER_SEC - self.first_col_time) * \
            self.range_frequency
        return row, col


class GRDCoordinateMixin:

    first_row_time: float
    azimuth_time_interval: float
    range_pixel_spacing: float
    srgr: SRGRConverter

    def to_azt_rng(self, row, col):
        azt = row * self.azimuth_time_interval + self.first_row_time
        gr = col * self.range_pixel_spacing
        rng = self.srgr.gr_to_rng(gr, azt)
        return azt, rng

    def to_row_col(self, azt, rng):
        row = (azt - self.first_row_time) / self.azimuth_time_interval
        gr = self.srgr.rng_to_gr(rng, azt)
        col = gr / self.range_pixel_spacing
        return row, col
