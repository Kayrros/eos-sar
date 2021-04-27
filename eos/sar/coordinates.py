"""
Coordinates conversion between image (row/column denoted as `r` and `c`) and
sar (azimuth time and range denoted as `t` and `g`).
"""
from eos.sar import const


class CoordinateMixin:
    def to_azt_rng(self, r, c):
        azt = r / self.azimuth_frequency + self.first_row_time
        rng = (c / self.range_frequency + self.first_col_time) * \
              const.LIGHT_SPEED_M_PER_SEC / 2
        return azt, rng

    def to_row_col(self, azt, rng):
        r = 0  # ...
        c = 0  # ...
        return r, c
