import abc


class SensorModel(abc.ABC):
    """SenSorModel is an abstract class that defines the expected method of
    any eos sensor model. It is expected that this abstract will be implemented
    for each SAR satellite.
    """
    azimuth_frequency: float
    range_frequency: float
    approx_geom: list 
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
