from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import override

import eos.sar
from eos.sar.srgr import Arrayf32


@dataclass(frozen=True)
class Sentinel1GRDSRGRMetadata:
    times: list[float]
    srgr_coeffs: list[list[float]]
    grsr_coeffs: list[list[float]]
    sr0: list[float]
    gr0: list[float]

    def __getitem__(self, name: str) -> Any:
        import warnings

        warnings.warn(
            "Indexing a Sentinel1GRDSRGRMetadata is deprecated (they no longer are dict).",
            DeprecationWarning,
        )
        return self.__dict__[name]

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Sentinel1GRDSRGRMetadata:
        return Sentinel1GRDSRGRMetadata(**d)


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
    s = (azt - ta) / (tb - ta)
    r0 = ra + s * (rb - ra)

    # linear interpolation of the polynomial coefficients to convert slant
    # range to ground range (or the opposite):
    #  - pa and pb are arrays of shape (len(azt), n), where n is the number of srgr/grsr coefficients
    #  - s is a list of length len(azt), which we convert to a 2D array of
    #    shape (len(azt), ) for multiplication broadcasting: when writing
    #    s * pa, we want to multiply each column of pa by s, elementwise.
    pa = coeffs[a]
    pb = coeffs[b]
    s = s[:, np.newaxis]
    p = pa + s * (pb - pa)

    # revert the polynomial's coefficients to get them in decreasing powers
    p = np.fliplr(p)

    # evaluate every polynom on the corresponding slant/ground range value
    y = np.polyval(p.T, x - r0)

    return y


class Sentinel1SRGRConverter(eos.sar.srgr.SRGRConverter):
    def __init__(self, srgr_meta: Sentinel1GRDSRGRMetadata):
        super().__init__()
        self.times = np.asarray(srgr_meta.times)
        self.srgr_coeffs = np.asarray(srgr_meta.srgr_coeffs)
        self.grsr_coeffs = np.asarray(srgr_meta.grsr_coeffs)
        self.sr0 = np.asarray(srgr_meta.sr0)
        self.gr0 = np.asarray(srgr_meta.gr0)

    @override
    def gr_to_rng(self, gr: ArrayLike, azt: ArrayLike) -> Arrayf32:
        gr = np.atleast_1d(gr)
        azt = np.atleast_1d(azt)

        rng = _evaluate(azt, gr, self.times, self.grsr_coeffs, self.gr0)

        # support for scalar input
        if len(rng) == 1:
            return rng[0]
        return rng

    @override
    def rng_to_gr(self, rng: ArrayLike, azt: ArrayLike) -> Arrayf32:
        rng = np.atleast_1d(rng)
        azt = np.atleast_1d(azt)

        gr = _evaluate(azt, rng, self.times, self.srgr_coeffs, self.sr0)

        # support for scalar input
        if len(gr) == 1:
            return gr[0]
        return gr
