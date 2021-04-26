from eos.sar import range_doppler
from eos.sar import const

def col_to_rng(col, first_col_time, range_frequency):
    """
    Parameters
    ----------
    col : ndarray or scalar
        col coordinate in image.
    first_col_time: scalar
        Two way slant range time of the first column in the image.
    range_frequency: scalar
        Two way slant range time sampling frequency.

    Returns
    -------
    ndarray or scalar
        One way range expressed in meters
    """
    Two_way_time = col/range_frequency + first_col_time
    return Two_way_time*const.LIGHT_SPEED_M_PER_SEC/2


def rng_to_col(rng, first_col_time, range_frequency):
    """
    Parameters
    ----------
    rng : ndarray or scalar
        One way range expressed in meters.
    first_col_time: scalar
        Two way slant range time of the first column in the image.
    range_frequency: scalar
        Two way slant range time sampling frequency.

    Returns
    -------
    ndarray or scalar
        col coordinate in image

    """
    Two_way_time = 2*rng/const.LIGHT_SPEED_M_PER_SEC
    return (Two_way_time  - first_col_time)*range_frequency


def row_to_ta(row, first_row_time, azimuth_frequency):
    """

    Parameters
    ----------
    row : ndarray or scalar
        row coordinate in image.
    first_row_time: scalar
        Azimuth timestamp of the first row in the image.
    azimuth_frequency: scalar
        Sampling frequency in azimuth.
    
    Returns
    -------
    ndarray or scalar
        Azimuth timestamp.

    """
    return row / azimuth_frequency + first_row_time


def ta_to_row(ta, first_row_time, azimuth_frequency):
    """

    Parameters
    ----------
    ta : ndarray or scalar
        Azimuth timestamp.
    first_row_time: scalar
        Azimuth timestamp of the first row in the image.
    azimuth_frequency: scalar
        Sampling frequency in azimuth.

    Returns
    -------
    ndarray or scalar
        row coordinate in image.

    """
    return (ta - first_row_time)*azimuth_frequency

class SensorModel:
    """Base class for all SAR sensors
    """
    def __init__(self, state_vectors, azimuth_frequency, range_frequency,
                 first_row_time, first_col_time):
        """Constructor
    
        Parameters
        ----------
        state_vectors : List of dicts
            List of state vectors (time, position, velocity).
        azimuth_frequency: scalar
            Sampling frequency in azimuth.
        range_frequency : scalar
            Two way slant range time sampling frequency.
        first_row_time: scalar
            Azimuth timestamp of the first row in the image.
        first_col_time: scalar
            Two way slant range time of the first column in the image.

        """
        self.orbit = range_doppler.Orbit(state_vectors)
        self.azimuth_frequency = azimuth_frequency
        self.range_frequency = range_frequency
        self.first_row_time = first_row_time
        self.first_col_time = first_col_time
        
    def to_ta(self, row):
        """To azimuth time"""
        return row_to_ta(row, self.first_row_time, self.azimuth_frequency)
    
    def to_row(self, ta): 
        """To row coordinate"""
        return ta_to_row(ta, self.first_row_time, self.azimuth_frequency)
    
    def to_rng(self, col):
        """To range in meters"""
        return col_to_rng(col, self.first_col_time, self.range_frequency)
    
    def to_col(self, rng): 
        """To col coordinate"""
        return rng_to_col(rng, self.first_col_time, self.range_frequency)
            

