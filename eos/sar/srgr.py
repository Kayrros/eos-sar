import numpy as np


import datetime


def string_to_timestamp(s):
    """
    Convert a string representing a date and time to a float number.

    Args:
        s (str): string representing a date and time.

    Returns:
        float: POSIX timestamp.
    """
    try:
        dt = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        dt = datetime.datetime.strptime(s, "%Y%m%dT%H%M%S")
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()


class SRGRConverter:

    def __init__(self, srgr):
        self.srgr = srgr

    def gr_to_rng(self, rng, azt):
        raise NotImplementedError

    def rng_to_gr(self, rng, azt):
        """
        Compute ground range from slant range and azimuth time.

        This implements the conversion described in table 6-91, page 6-95 of
        https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification

        from s1m

        Args:
            rng (array): slant range, in meters
            azt (array): azimuth time POSIX timestamp

        Returns:
            gr (scalar or array): ground range, in meters
        """
        rng = np.atleast_1d(rng)
        azt = np.atleast_1d(azt)
        samples = np.asarray([string_to_timestamp(s["azimuthTime"]) for s
                              in self.srgr])
        if not (all(azt >= samples[0]) and all(azt <= samples[-1])):
            raise ValueError("Azimuth time not included in GRD image bounds")

        # for each element of azt, find the two closest elements in the samples
        # list:
        #   - d is the distance matrix, with shape (len(azt), len(samples)),
        #   - a and b are lists of length len(azt), where a[k] is the index of the
        #     closest sample to azt[k], and b[k] is the index of the second
        #     closest sample to azt[k].
        #   - ta and tb are lists of length len(azt) containing the closest and
        #     second closest samples to azt
        d = np.abs(azt[:, np.newaxis] - samples[np.newaxis, :])
        a, b = np.argpartition(d, 1).T[:2]
        ta = np.array([samples[i] for i in a])
        tb = np.array([samples[i] for i in b])

        # linear interpolation of the slant range origin
        ra = np.array([float(self.srgr[k]["sr0"]) for k in a])
        rb = np.array([float(self.srgr[k]["sr0"]) for k in b])
        s = np.abs((azt - ta) / (tb - ta))
        r0 = (1 - s) * ra + s * rb

        # linear interpolation of the polynomial coefficients to convert slant
        # range to ground range:
        #  - pa and pb are arrays of shape (len(azt), n), where n is the number of srgr coefficients
        #  - s is a list of length len(azt), which we convert to a 2D array of
        #    shape (len(azt), ) for multiplication broadcasting: when writing
        #    s * pa, we want to multiply each column of pa by s, elementwise.
        pa = np.array([list(map(float, self.srgr[k]["srgrCoefficients"]["#text"].split())) for k in a])
        pb = np.array([list(map(float, self.srgr[k]["srgrCoefficients"]["#text"].split())) for k in b])
        s = s[:, np.newaxis]
        p = (1 - s) * pa + s * pb

        # revert the polynomial's coefficients to get them in decreasing powers
        p = np.fliplr(p)

        # evaluate every polynom on the corresponding slant range value
        gr = np.array([np.polyval(pp, x) for pp, x in zip(p, rng - r0)])
        return gr.squeeze()
