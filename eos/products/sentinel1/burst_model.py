import numpy as np
import pyproj
from eos.sar import model, range_doppler, const, coordinates


class Sentinel1BurstModel(coordinates.CoordinateMixin, model.SensorModel):
    def __init__(self,
                 range_frequency,
                 azimuth_frequency,
                 slant_range_time,
                 bistatic_correction=True,
                 apd_correction=True,
                 max_iterations=20,
                 tolerance=0.01):
        """Sentinel1BurstModel"""
        self.range_frequency = range_frequency
        self.azimuth_frequency = azimuth_frequency
        self.slant_range_time = slant_range_time
        # etc.

        self.bistatic_correction = bistatic_correction
        self.apd_correction = apd_correction

        self.geocentric_tolerance = tolerance

        self.localization_tolerance = ... # need to translate
        self.projection_tolerance = ... # can we find a rule of thumb to convert tolerance in meters to tolerance in s?

    def projection(self, xs, ys, alts, crs='epsg:4326', vert_crs=None):
        """write the function"""
        pass

    def localization(self, cols, rows, alts, crs='epsg:4326', vert_crs=None):
        """ write the function"""
        pass
