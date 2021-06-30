"""Base class for all Sensor Models."""

import abc
from eos.sar.orbit import Orbit

class SensorModel(abc.ABC):
    """SensorModel is an abstract class that defines the expected method of\
    any eos sensor model. It is expected that this abstract will be \
    implemented for each SAR satellite."""

    azimuth_frequency: float
    range_frequency: float
    approx_geom: list
    w: int # width of image
    h: int # height of image
    orbit: Orbit
    wavelength: float 
    
    @abc.abstractmethod
    def to_azt_rng(self, row, col):
        pass

    @abc.abstractmethod
    def to_row_col(self, azt, rng):
        pass

    @abc.abstractmethod
    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None):
        pass

    @abc.abstractmethod
    def localization(self, row, col, alt, crs='epsg:4326', vert_crs=None):
        pass
