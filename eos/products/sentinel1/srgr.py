import numpy as np

import eos.sar


def _evaluate(azt, x, times, coeffs, origins):
    """
    (from s1m)

    Compute ground range from slant range or the opposite, and the azimuth time.

    This implements the conversion described in table 6-91, page 6-95 of
    https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification

    Args:
        x (array): ground or slant range, in meters
        azt (array): azimuth time POSIX timestamp

    Returns:
        gr (scalar or array): ground or slant range, in meters
    """
    # for each element of azt, find the two closest elements in the times
    # list:
    #   - d is the distance matrix, with shape (len(azt), len(times)),
    #   - a and b are lists of length len(azt), where a[k] is the index of the
    #     closest sample to azt[k], and b[k] is the index of the second
    #     closest sample to azt[k].
    #   - ta and tb are lists of length len(azt) containing the closest and
    #     second closest times to azt
    d = np.abs(azt[:, np.newaxis] - times[np.newaxis, :])
    a, b = np.argpartition(d, 1).T[:2]
    ta = times[a]
    tb = times[b]

    # linear interpolation of the slant/ground range origin
    ra = origins[a]
    rb = origins[b]
    s = np.abs((azt - ta) / (tb - ta))
    r0 = (1 - s) * ra + s * rb

    # linear interpolation of the polynomial coefficients to convert slant
    # range to ground range (or the opposite):
    #  - pa and pb are arrays of shape (len(azt), n), where n is the number of srgr/grsr coefficients
    #  - s is a list of length len(azt), which we convert to a 2D array of
    #    shape (len(azt), ) for multiplication broadcasting: when writing
    #    s * pa, we want to multiply each column of pa by s, elementwise.
    pa = coeffs[a]
    pb = coeffs[b]
    s = s[:, np.newaxis]
    p = (1 - s) * pa + s * pb

    # revert the polynomial's coefficients to get them in decreasing powers
    p = np.fliplr(p)

    # evaluate every polynom on the corresponding slant/ground range value
    y = np.polyval(p.T, x - r0)

    return y


class Sentinel1SRGRConverter(eos.sar.srgr.SRGRConverter):

    def __init__(self, times, srgr_coeffs, grsr_coeffs, sr0, gr0):
        super().__init__()
        self.times = np.asarray(times)
        self.srgr_coeffs = np.asarray(srgr_coeffs)
        self.grsr_coeffs = np.asarray(grsr_coeffs)
        self.sr0 = np.asarray(sr0)
        self.gr0 = np.asarray(gr0)

    def gr_to_rng(self, gr, azt):
        gr = np.atleast_1d(gr)
        azt = np.atleast_1d(azt)

        if any(azt < self.times[0]) or any(azt > self.times[-1]):
            raise ValueError("Azimuth time not included in GRD image bounds")

        rng = _evaluate(azt, gr, self.times, self.grsr_coeffs, self.gr0)

        # support for scalar input
        if len(rng) == 1:
            return rng[0]
        return rng

    def rng_to_gr(self, rng, azt):
        rng = np.atleast_1d(rng)
        azt = np.atleast_1d(azt)
        if any(azt < self.times[0]) or any(azt > self.times[-1]):
            raise ValueError("Azimuth time not included in GRD image bounds")

        gr = _evaluate(azt, rng, self.times, self.srgr_coeffs, self.sr0)

        # support for scalar input
        if len(gr) == 1:
            return gr[0]
        return gr
