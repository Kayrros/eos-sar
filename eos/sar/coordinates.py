"""Coordinates conversion between image (row/column denoted as `row` \
    and `col`) and sar (azimuth time and range denoted as `azt` and `rng`)."""

from eos.sar import const


class CoordinateMixin:
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
