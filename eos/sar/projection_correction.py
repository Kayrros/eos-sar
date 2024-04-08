import abc
from typing import Sequence

import numpy as np
import pyproj

from eos.sar import geoconfig


class Points:
    """base class for a set of Points"""

    singleton: bool

    def _force_array(self, val):
        """convert val to 1d array if not already the case."""
        return np.atleast_1d(val)

    def _all_len_eq(self, vals):
        """Check that all arrays in vals have the same length  \
            and if the lenght is one, keep that in mind."""
        lengths = [len(val) for val in vals if val is not None]
        if len(lengths) > 1:
            assert all(len_curr == lengths[0] for len_curr in lengths[1:])
        self.singleton = lengths[0] == 1

    def _get_vals(self, vals, squeeze):
        """Get the arrays in vals, optionnaly squeezing if singleton."""
        if squeeze and self.singleton:
            return [val[0] for val in vals]
        else:
            return vals

    def _get_val(self, val, squeeze):
        """Get val, optionnaly squeezing if singleton."""
        return self._get_vals([val], squeeze)[0]

    def _add_val(self, val, d_val):
        if d_val is None:
            return val
        else:
            assert len(val) == len(d_val), "vals and diff vals have different lengths"
            return val + d_val


class GeoPoints(Points):
    """Geo Point for correction estimation"""

    def __init__(self, gx, gy, gz):
        """
        Constructor.

        Parameters
        ----------
        gx: float or 1darray
            X geocentric coord.
        gy : float or 1darray
            Y geocentric coord.
        gz : float or 1darray
            Z geocentric coord.

        Returns
        -------
        None.

        """
        self.gx = self._force_array(gx)
        self.gy = self._force_array(gy)
        self.gz = self._force_array(gz)
        self._all_len_eq([self.gx, self.gy, self.gz])

    def get_geo(self, squeeze=False):
        """
        Get the geocentric coordinates.

        Parameters
        ----------
        squeeze : Boolean, optional
            If True, and if only one element in GeoPoints, return float instead of array.
            The default is False.

        Returns
        -------
        gx: float or 1darray
            X geocentric coord.
        gy : float or 1darray
            Y geocentric coord.
        gz : float or 1darray
            Z geocentric coord.

        """
        gx, gy, gz = self._get_vals([self.gx, self.gy, self.gz], squeeze)
        return gx, gy, gz

    def get_lon_lat_alt(self, squeeze=False):
        """
        Get the longitude, latitude and altitude.

        Parameters
        ----------
        squeeze : Boolean, optional
            If True, and if only one element in GeoPoints, return float instead of array.
            The default is False.

        Returns
        -------
        lon : float or 1darray, optional
            Longitude.
        lat : float or 1darray, optional
            Latitude.
        alt : float or 1darray, optional
            Ellipsoid altitude.

        """
        transformer = pyproj.Transformer.from_crs(
            "epsg:4978", "epsg:4979", always_xy=True
        )
        lon, lat, alt = transformer.transform(self.gx, self.gy, self.gz)

        lon, lat, alt = self._get_vals([lon, lat, alt], squeeze)
        return lon, lat, alt

    def _add_geo_as_tuple(self, dgx=None, dgy=None, dgz=None):
        """add geo and return it as tuple."""

        new_gx = self._add_val(self.gx, dgx)
        new_gy = self._add_val(self.gy, dgy)
        new_gz = self._add_val(self.gz, dgz)

        return new_gx, new_gy, new_gz

    def add_geo(self, dgx=None, dgy=None, dgz=None):
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
        return GeoPoints(*self._add_geo_as_tuple(dgx, dgy, dgz))


class ImagePoints(Points):
    """Image Points class"""

    def __init__(self, azt, rng):
        """
        Constructor.

        Parameters
        ----------
        azt : float or 1darray
            Azimuth time.
        rng : float or 1darray

        Returns
        -------
        None.

        """
        self.azt = self._force_array(azt)
        self.rng = self._force_array(rng)
        self._all_len_eq([self.azt, self.rng])

    def get_azt_rng(self, squeeze=False):
        """
        Get the azimuth and the range.

        Parameters
        ----------
        squeeze : Boolean, optional
            If True, and if only one element in ImagePoints, return float instead of array.
            The default is False.

        Returns
        -------
        azt : float or 1darray
            Azimuth time.
        rng : float or 1darray
            Range distance (meters).

        """
        azt, rng = self._get_vals([self.azt, self.rng], squeeze)
        return azt, rng

    def _add_azt_rng_as_tuple(self, dazt=None, drng=None):
        """add azt rng and return it as tuple."""
        new_azt = self._add_val(self.azt, dazt)
        new_rng = self._add_val(self.rng, drng)

        return new_azt, new_rng

    def add_azt_rng(self, dazt=None, drng=None):
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
        return ImagePoints(*self._add_azt_rng_as_tuple(dazt, drng))


class GeoImagePoints(GeoPoints, ImagePoints):
    def __init__(self, gx, gy, gz, azt, rng):
        """
        Constructor.

        Parameters
        ----------
        gx: float or 1darray
            X geocentric coord.
        gy : float or 1darray
            Y geocentric coord.
        gz : float or 1darray
            Z geocentric coord.
        azt : float or 1darray
            Azimuth time.
        rng : float or 1darray
            Range distance (meters).

        Returns
        -------
        None.

        """
        GeoPoints.__init__(self, gx, gy, gz)
        ImagePoints.__init__(self, azt, rng)
        # other lengths checked in base classes
        self._all_len_eq([self.gx, self.azt])

    def get_cos_i(self, orbit, squeeze=False):
        """
        Compute the cosine incidence.

        Parameters
        ----------
        orbit : eos.sar.orbit.Orbit
            Orbit instance.
         squeeze : Boolean, optional
             If True, and if only one element in the Geo Point, return float instead of array.
             The default is False.

        Returns
        -------
        cos_i : float or 1darray
            Cosine incidence.

        """
        sat = orbit.evaluate(self.azt)
        cos_i, _ = geoconfig.compute_cosi_rng(
            np.column_stack([self.gx, self.gy, self.gz]), sat
        )

        return self._get_val(cos_i, squeeze)

    def add_geo(self, dgx=None, dgy=None, dgz=None):
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
        GeoImagePoints
            New shifted GeoImagePoints.

        """
        gx, gy, gz = self._add_geo_as_tuple(dgx, dgy, dgz)
        return GeoImagePoints(gx, gy, gz, self.azt, self.rng)

    def add_azt_rng(self, dazt=None, drng=None):
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
        GeoImagePoint
            New Shifted GeoImagePoints.

        """
        return GeoImagePoints(
            self.gx, self.gy, self.gz, *self._add_azt_rng_as_tuple(dazt, drng)
        )


def invert_or_None(shift=None):
    """Invert a float or array if not None, else return None."""
    if shift is None:
        return None
    else:
        return -shift


class GeoCorrection(abc.ABC):
    """Correction on x, y, z geocentric coordinates"""

    def __init__(self):
        """Constructor, set shifts to None."""
        self.dgx = None
        self.dgy = None
        self.dgz = None

    @abc.abstractmethod
    def estimate(self, pt: Points):
        """Here self.dg[x, y, z] will be set to other than None."""
        pass

    def apply(self, geo_pt: GeoPoints, inverse=False):
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


class ImageCorrection(abc.ABC):
    """Correction on azt, rng image coordinates"""

    def __init__(self):
        """Constructor, set shifts to None."""
        self.dazt = None
        self.drng = None

    @abc.abstractmethod
    def estimate(self, pt: GeoImagePoints):
        """Here self.dazt, self.drng will be estimated."""
        pass

    def apply(self, im_pt: GeoImagePoints, inverse=False):
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
        # here Coord correction will be applied

        if inverse:
            return im_pt.add_azt_rng(
                invert_or_None(self.dazt), invert_or_None(self.drng)
            )

        return im_pt.add_azt_rng(self.dazt, self.drng)


class Corrector:
    """Corrector class containing multiple corrections."""

    def __init__(self, corrections: Sequence[ImageCorrection] = []):
        """
        Constructor.

        Parameters
        ----------
        corrections : list, optional
            Each element is a correction. The default is [].

        Returns
        -------
        None.

        """
        self.corrections = corrections

    def empty(self):
        """Check if corrector has 0 corrections."""
        return len(self.corrections) == 0

    def estimate(self, geo_im_pt: GeoImagePoints):
        """
        All corrections are estimated on initial (uncorrected) Points.

        Parameters
        ----------
        geo_im_pt : GeoImagePoints
            Initial uncorrected Points.

        Returns
        -------
        None.

        """
        for correc in self.corrections:
            correc.estimate(geo_im_pt)

    def apply(self, geo_im_pt: GeoImagePoints, inverse=False):
        """
        All corrections previously estimated are applied sequentially according
        to their list order.

        Parameters
        ----------
        geo_im_pt : GeoImagePoints
            Uncorrected GeoImagePoints.
        inverse : Boolean, optional
            If True, apply the inverse shift vector. The default is False.

        Returns
        -------
        geo_im_pt : GeoImagePoints
            Corrected GeoImagePoints.

        """
        for correc in self.corrections:
            geo_im_pt = correc.apply(geo_im_pt, inverse)
        return geo_im_pt

    def estimate_and_apply(self, geo_im_pt: GeoImagePoints, inverse=False):
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
        self.estimate(geo_im_pt)
        return self.apply(geo_im_pt, inverse)
