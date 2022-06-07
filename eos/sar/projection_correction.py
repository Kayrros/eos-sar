import abc
import numpy as np

from eos.sar import geoconfig


class ControlPoint:

    ''' Control point for correction estimation '''

    def __init__(self, gx, gy, gz, lon=None, lat=None, alt=None
                 ):
        """
        Constructor for the control point.

        Parameters
        ----------
        gx: float or 1darray
            X geocentric coord.
        gy : float or 1darray
            Y geocentric coord.
        gz : float or 1darray
            Z geocentric coord.
        lon : float or 1darray, optional
            Longitude. The default is None.
        lat : float or 1darray, optional
            Latitude. The default is None.
        alt : float or 1darray, optional
            Ellipsoid altitude. The default is None.

        Returns
        -------
        None.

        """
        self.gx = self._force_array(gx)
        self.gy = self._force_array(gy)
        self.gz = self._force_array(gz)
        self.lon = self._force_array(lon)
        self.lat = self._force_array(lat)
        self.alt = self._force_array(alt)
        self._all_len_eq([self.gx, self.gy, self.lon, self.lat, self.alt])

    def _force_array(self, val):
        """convert val to 1d array if not already the case."""
        if val is not None:
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

    def get_geo(self, squeeze=False):
        """
        Get the geocentric coordinates.

        Parameters
        ----------
        squeeze : Boolean, optional
            If True, and if only one element in the Control Point, return float instead of array.
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
            If True, and if only one element in the Control Point, return float instead of array.
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
        if self.lon is None or self.lat is None or self.alt is None:
            import pyproj
            transformer = pyproj.Transformer.from_crs(
                'epsg:4978', 'epsg:4979', always_xy=True)
            lon, lat, alt = transformer.transform(self.gx, self.gy, self.gz)

        self.lon = lon
        self.lat = lat
        self.alt = alt

        lon, lat, alt = self._get_vals([self.lon, self.lat, self.alt], squeeze)
        return lon, lat, alt

    def _add_to_arrays(self, list_arrays, list_delta_arrays):
        """Add a list of of delta_arrays to a list of arrays.
        If a delta array is None, the array is kept as is."""
        list_result = []
        for arr, darr in zip(list_arrays, list_delta_arrays):
            if darr is not None:
                list_result.append(arr + darr)
            else:
                list_result.append(arr)
        return list_result

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
        ControlPoint
            New shifted ControlPoint.

        """
        if dgx is None and dgy is None and dgz is None:
            print("Warning: all differential vectors are None, nothing to be changed")
            return self

        res = self._add_to_arrays([self.gx, self.gy, self.gz],
                                  [dgx, dgy, dgz])

        return ControlPoint(*res)


class CorrectionControlPoint(ControlPoint):

    def __init__(self, gx, gy, gz, azt, rng, lon=None, lat=None, alt=None):
        """
        Correction Control Point Constructor.

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
        lon : float or 1darray, optional
            Longitude. The default is None.
        lat : float or 1darray, optional
            Latitude. The default is None.
        alt : float or 1darray, optional
            Ellipsoid altitude. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(gx, gy, gz, lon, lat, alt)

        self.azt = self._force_array(azt)
        self.rng = self._force_array(rng)
        # other lengths checked in super
        self._all_len_eq([self.gx, self.azt, self.rng])

    def get_azt_rng(self, squeeze=False):
        """
        Get the azimuth and the range.

        Parameters
        ----------
        squeeze : Boolean, optional
            If True, and if only one element in the Control Point, return float instead of array.
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

    def get_cos_i(self, orbit, squeeze=False):
        """
        Compute the cosine incidence.

        Parameters
        ----------
        orbit : eos.sar.orbit.Orbit
            Orbit instance.
         squeeze : Boolean, optional
             If True, and if only one element in the Control Point, return float instead of array.
             The default is False.

        Returns
        -------
        cos_i : float or 1darray
            Cosine incidence on CorrectionControlPoint.

        """
        sat = orbit.evaluate(self.azt)
        cos_i, _ = geoconfig.compute_cosi_rng(
            np.column_stack([self.gx, self.gy, self.gz]), sat)

        return self._get_val(cos_i, squeeze)

    def add_coords(self, dazt=None, drng=None):
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
        Shifted CorrectionControlPoint.

        """
        if dazt is None and drng is None:
            print("Warning: all differential vectors are None, nothing to be changed")
            return self

        res = self._add_to_arrays([self.azt, self.rng], [dazt, drng])

        return CorrectionControlPoint(self.gx, self.gy, self.gz,
                                      *res, self.lon, self.lat, self.alt,
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
    def estimate(self, cp: ControlPoint):
        """Here self.dg[x, y, z] will be set to other than None."""
        pass

    def apply(self, cp: ControlPoint, inverse=False):
        """
        Apply GeoCorrection on a ControlPoint.

        Parameters
        ----------
        cp : ControlPoint
            Control Point to be corrected.
        inverse : Boolean, optional
            If True, the inverse correction (shift vector) is applied. The default is False.

        Returns
        -------
        ControlPoint
            New shifted ControlPoint instance.

        """

        if inverse:
            return cp.add_geo(invert_or_None(self.dgx),
                              invert_or_None(self.dgy),
                              invert_or_None(self.dgz))

        return cp.add_geo(self.dgx, self.dgy, self.dgz)


class CoordCorrection(abc.ABC):

    """Correction on azt, rng image coordinates"""

    def __init__(self):
        """Constructor, set shifts to None."""
        self.dazt = None
        self.drng = None

    @abc.abstractmethod
    def estimate(self, ccp: CorrectionControlPoint):
        """Here self.dazt, self.drng will be estimated."""
        pass

    def apply(self, ccp: CorrectionControlPoint, inverse=False):
        """
        Apply the Coordinate Correction on a ccp.

        Parameters
        ----------
        ccp : CorrectionControlPoint
            CorrectionControlPoint to be corrected.
        inverse : Boolean, optional
            If inverse, apply the inverse correction(inverse shift vector). The default is False.

        Returns
        -------
        CorrectionControlPoint
            New shifted CorrectionControlPoint instance.

        """
        # here Coord correction will be applied

        if inverse:
            return ccp.add_coords(invert_or_None(self.dazt),
                                  invert_or_None(self.drng))

        return ccp.add_coords(self.dazt, self.drng)


class Corrector:
    """Corrector class containing multiple corrections."""

    def __init__(self, corrections=[]):
        """
        Constructor.

        Parameters
        ----------
        corrections : list, optional
            Each element is a correction applicable on a ControlPoint. The default is [].

        Returns
        -------
        None.

        """
        self.corrections = corrections

    def not_empty(self):
        """Check if corrector has at least one correction."""
        return len(self.corrections) > 0

    def estimate(self, cp: ControlPoint):
        """
        All corrections are estimated on initial (uncorrected) ControlPoint.

        Parameters
        ----------
        cp : ControlPoint
            Initial uncorrected ControlPoint.

        Returns
        -------
        None.

        """
        for correc in self.corrections:
            correc.estimate(cp)

    def apply(self, cp: ControlPoint, inverse=False):
        """
        All corrections previously estimated are applied sequentially according
        to their list order.

        Parameters
        ----------
        cp : ControlPoint
            Uncorrected ControlPoint.
        inverse : Boolean, optional
            If True, apply the inverse shift vector. The default is False.

        Returns
        -------
        cp : ControlPoint
            Corrected ControlPoint.

        """
        for correc in self.corrections:
            cp = correc.apply(cp, inverse)
        return cp

    def estimate_and_apply(self, cp: ControlPoint, inverse=False):
        """
        Estimate and apply the corrections.

        cp : ControlPoint
            Uncorrected ControlPoint.
        inverse : Boolean, optional
            If True, apply the inverse shift vector. The default is False.

        Returns
        -------
        cp : ControlPoint
            Corrected ControlPoint.

        """
        self.estimate(cp)
        return self.apply(cp, inverse)
