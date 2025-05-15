import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeVar, Union

import numpy as np
import pyproj
from numpy.typing import NDArray
from typing_extensions import Self, override

from eos.sar import const, geoconfig


def _all_len_eq(vals):
    """Check that all arrays in vals have the same length  \
        and if the length is one, keep that in mind."""
    lengths = [len(val) for val in vals if val is not None]
    if len(lengths) > 1:
        assert all(len_curr == lengths[0] for len_curr in lengths[1:])


def _add_val(val, d_val):
    if d_val is None:
        return val
    else:
        assert len(val) == len(d_val), "vals and diff vals have different lengths"
        return val + d_val


@dataclass(frozen=True)
class GeoPoints:
    """Geo Point for correction estimation"""

    gx: NDArray[np.float64]
    """ X geocentric coord. """
    gy: NDArray[np.float64]
    """ Y geocentric coord. """
    gz: NDArray[np.float64]
    """ Z geocentric coord. """

    def __post_init__(self):
        _all_len_eq([self.gx, self.gy, self.gz])

    def get_geo(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the geocentric coordinates.

        Returns
        -------
        gx: float or 1darray
            X geocentric coord.
        gy : float or 1darray
            Y geocentric coord.
        gz : float or 1darray
            Z geocentric coord.

        """
        return self.gx, self.gy, self.gz

    def get_lon_lat_alt(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the longitude, latitude and altitude.

        Returns
        -------
        lon : 1darray
            Longitude.
        lat : 1darray
            Latitude.
        alt : 1darray
            Ellipsoid altitude.

        """
        transformer = pyproj.Transformer.from_crs(
            "epsg:4978", "epsg:4979", always_xy=True
        )
        lon, lat, alt = transformer.transform(self.gx, self.gy, self.gz)

        return lon, lat, alt

    def _add_geo_as_tuple(self, dgx=None, dgy=None, dgz=None):
        """add geo and return it as tuple."""

        new_gx = _add_val(self.gx, dgx)
        new_gy = _add_val(self.gy, dgy)
        new_gz = _add_val(self.gz, dgz)

        return new_gx, new_gy, new_gz

    def add_geo(self, dgx=None, dgy=None, dgz=None) -> Self:
        """
        Add a shift to geocentric coordinates. If shift is kept as None, no shift
        is applied in the specified axis direction.

        Parameters
        ----------
        dgx : float or 1d array, optional
            Shift to add to X geocentric coord. The default is None.
        dgy : float or 1d array, optional
            Shift to add to Y geocentric coord. The default is None.
        dgz : float or 1d array, optional
            Shift to add to Z geocentric coord. The default is None.

        Returns
        -------
        GeoPoints
            New shifted GeoPoints.

        """
        gx, gy, gz = self._add_geo_as_tuple(dgx, dgy, dgz)
        return dataclasses.replace(self, gx=gx, gy=gy, gz=gz)


@dataclass(frozen=True)
class ImagePoints:
    """Image Points class"""

    azt: NDArray[np.float64]
    rng: NDArray[np.float64]

    def __post_init__(self):
        _all_len_eq([self.azt, self.rng])

    def get_azt_rng(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the azimuth and the range.

        Returns
        -------
        azt : float or 1darray
            Azimuth time.
        rng : float or 1darray
            Range distance (meters).

        """
        return self.azt, self.rng

    def _add_azt_rng_as_tuple(self, dazt=None, drng=None):
        """add azt rng and return it as tuple."""
        new_azt = _add_val(self.azt, dazt)
        new_rng = _add_val(self.rng, drng)

        return new_azt, new_rng

    def add_azt_rng(self, dazt=None, drng=None) -> Self:
        """
        Add the coordinate shifts. If shift is None, don't add anything in this
        dimension.

        Parameters
        ----------
        dazt : float or 1darray, optional
            Shift on azimuth. The default is None.
        drng : float or 1darray, optional
            Shift on range. The default is None.

        Returns
        -------
        ImagePoints
            New Shifted ImagePoints.

        """
        azt, rng = self._add_azt_rng_as_tuple(dazt, drng)
        return dataclasses.replace(self, azt=azt, rng=rng)


@dataclass(frozen=True)
class GeoImagePoints(
    ImagePoints, GeoPoints
):  # order kind of matter for the constructor
    def __post_init__(self):
        _all_len_eq([self.azt, self.rng, self.gx, self.gy, self.gz])

    def get_cos_i(self, orbit) -> NDArray[np.float64]:
        """
        Compute the cosine incidence.

        Parameters
        ----------
        orbit : eos.sar.orbit.Orbit
            Orbit instance.

        Returns
        -------
        cos_i : float or 1darray
            Cosine incidence.

        """
        sat = orbit.evaluate(self.azt)
        cos_i, _ = geoconfig.compute_cosi_rng(
            np.column_stack([self.gx, self.gy, self.gz]), sat
        )

        return cos_i


_GeoP = TypeVar("_GeoP", bound=GeoPoints)
_ImageP = TypeVar("_ImageP", bound=ImagePoints)


def invert_or_None(shift=None):
    """Invert a float or array if not None, else return None."""
    if shift is None:
        return None
    else:
        return -shift


@dataclass(frozen=True)
class GeoCorrection:
    """Correction on x, y, z geocentric coordinates"""

    dgx: Optional[Any]
    dgy: Optional[Any]
    dgz: Optional[Any]

    def apply(self, geo_pt: _GeoP, inverse=False) -> _GeoP:
        """
        Apply GeoCorrection on a GeoPoint.

        Parameters
        ----------
        geo_pt : GeoPoints
            GeoPoints to be corrected.
        inverse : Boolean, optional
            If True, the inverse correction (shift vector) is applied. The default is False.

        Returns
        -------
        GeoPoints
            New shifted GeoPoints instance.

        """

        if inverse:
            return geo_pt.add_geo(
                invert_or_None(self.dgx),
                invert_or_None(self.dgy),
                invert_or_None(self.dgz),
            )

        return geo_pt.add_geo(self.dgx, self.dgy, self.dgz)


@dataclass(frozen=True)
class ImageCorrection:
    """Correction on azt, rng image coordinates"""

    dazt: Optional[Any]
    drng: Optional[Any]

    def apply(self, im_pt: _ImageP, inverse=False) -> _ImageP:
        """
        Apply the Coordinate Correction.

        Parameters
        ----------
        im_pt : GeoImagePoints
            ImagePoints to be corrected.
        inverse : Boolean, optional
            If inverse, apply the inverse correction(inverse shift vector). The default is False.

        Returns
        -------
        ImagePoints
            New shifted GeoImagePoints instance.

        """
        if inverse:
            return im_pt.add_azt_rng(
                invert_or_None(self.dazt), invert_or_None(self.drng)
            )

        return im_pt.add_azt_rng(self.dazt, self.drng)


class GeoCorrectionEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate(self, pt: GeoImagePoints) -> GeoCorrection: ...


class ImageCorrectionEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate(self, pt: GeoImagePoints) -> ImageCorrection:
        """
        Estimate the corrections dazt, drng.

        Parameters
        ----------
        pt : GeoImagePoints
            GeoImagePoints on which to compute the corrections.

        Returns
        -------
        ImageCorrection:
            The estimated corrections to be applied on the points.
        """


@dataclass(frozen=True)
class Corrector:
    """Corrector class containing multiple corrections."""

    estimators: Sequence[Union[ImageCorrectionEstimator, GeoCorrectionEstimator]] = (
        dataclasses.field(default_factory=list)
    )
    """ Each element is a correction. The default is []. """

    def empty(self):
        """Check if corrector has 0 corrections."""
        return len(self.estimators) == 0

    def estimate(
        self, geo_im_pt: GeoImagePoints
    ) -> list[Union[GeoCorrection, ImageCorrection]]:
        """
        All corrections are estimated on initial (uncorrected) Points.

        Parameters
        ----------
        geo_im_pt : GeoImagePoints
            Initial uncorrected Points.

        Returns
        -------
        list of ImageCorrection | GeoCorrection

        """
        return [estimator.estimate(geo_im_pt) for estimator in self.estimators]

    def apply(
        self,
        corrections: Sequence[Union[GeoCorrection, ImageCorrection]],
        geo_im_pt: GeoImagePoints,
        inverse=False,
    ) -> GeoImagePoints:
        """
        All corrections previously estimated are applied sequentially according
        to their list order.

        Parameters
        ----------
        corrections:
            Sequence of estimated corrections to apply.
        geo_im_pt : GeoImagePoints
            Uncorrected GeoImagePoints.
        inverse : Boolean, optional
            If True, apply the inverse shift vector. The default is False.

        Returns
        -------
        geo_im_pt : GeoImagePoints
            Corrected GeoImagePoints.

        """
        for correc in corrections:
            geo_im_pt = correc.apply(geo_im_pt, inverse)
        return geo_im_pt

    def estimate_and_apply(
        self, geo_im_pt: GeoImagePoints, inverse=False
    ) -> GeoImagePoints:
        """
        Estimate and apply the corrections.

        geo_im_pt : GeoImagePoints
            Uncorrected GeoImagePoints.
        inverse : Boolean, optional
            If True, apply the inverse shift vector. The default is False.

        Returns
        -------
        geo_im_pt : GeoImagePoints
            Corrected GeoImagePoints.
        """
        corrections = self.estimate(geo_im_pt)
        return self.apply(corrections, geo_im_pt, inverse)


@dataclass(frozen=True)
class RngAztShift(ImageCorrectionEstimator):
    """Applies the same shift on all points in image coordinates."""

    rng_shift: float
    azt_shift: float

    @override
    def estimate(self, pt: GeoImagePoints) -> ImageCorrection:
        """Makes shift arrays with the same lenght as points"""
        num_points = len(pt.azt)
        drng = np.full((num_points,), self.rng_shift, dtype=float)
        dazt = np.full((num_points,), self.azt_shift, dtype=float)
        return ImageCorrection(drng=drng, dazt=dazt)


class SLCPxShiftCorrection(RngAztShift):
    """Applies the same shift on all points in image coordinates."""

    def __init__(
        self,
        azimuth_frequency: float,
        range_frequency: float,
        col_shift: float,
        row_shift: float,
    ):
        azt_shift = row_shift / azimuth_frequency
        rng_shift = col_shift / range_frequency * const.LIGHT_SPEED_M_PER_SEC / 2
        super().__init__(rng_shift, azt_shift)
