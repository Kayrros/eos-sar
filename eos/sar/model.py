"""Base class for all Sensor Models."""

import abc
import logging
from typing import Optional

import numpy as np

from eos.sar.orbit import Orbit
from eos.sar import utils
from eos.sar.roi import Roi as Roi
import eos.dem

logger = logging.getLogger(__name__)


class SensorModel(abc.ABC):
    """SensorModel is an abstract class that defines the expected method of\
    any eos sensor model. It is expected that this abstract will be \
    implemented for each SAR satellite."""

    approx_geom: list
    w: int  # width of image
    h: int  # height of image
    orbit: Orbit
    wavelength: float

    @abc.abstractmethod
    def to_azt_rng(self, row, col):
        pass

    @abc.abstractmethod
    def to_row_col(self, azt, rng):
        pass

    @abc.abstractmethod
    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None, azt_init=None, as_azt_rng=False):
        pass

    @abc.abstractmethod
    def localization(self, row, col, alt, crs='epsg:4326', vert_crs=None,
                     x_init=None, y_init=None, z_init=None):
        pass

    def localize_without_alt(self, row, col, max_iter=5, eps=1.0,
                             alt_min=-1000, alt_max=9000, num_alt=100,
                             verbosity=False, *, elev):
        """
        Localize a pixel in the image to the 3D point without needing the
        the altitude. A set of altitude values are tested recursively.

        Parameters
        ----------
        row : ndarray
            row in the image.
        col : ndarray
            col in the image.
        max_iter : int, optional
            maximum number of recursions to find the height. The default is 5.
        eps : float, optional
            Precision on the height to stop the iterations. The default is 1.
        alt_min : float, optional
            Minimum altitude in the search space. The default is -1000.
        alt_max : float, optional
            Maximum altitude in the search space. The default is 9000.
        num_alt : int, optional
            Number of altitudes to test in each iteration. The default is 100.
        verbosity : bool, optional
            If True, print messages about the optimization. The default is False.
        elev: function
            Function elev(lon, lat) ---> alt w.r.t. ellipsoidal datum.
            lon, lat are in wgs84 crs.
            Should be vectorized (able to handle a query of arrays).

        Returns
        -------
        lon : ndarray
            longitude of point (wgs84).
        lat : ndarray
            latitude of point (wgs84).
        alt_opt : ndarray
            altitude of point (wgs84).
        masks : dict
            Masks on the returned points to indicate the status of the optim.
            mask["zeros"] indicates points that have exactly converged (error of 0)
            mask["converged"] indicates points that converged within eps tolerance.
            mask["invalid"] indicates the points for which no solution was found,
            probably because the actual altitude is outside the given range.


        """
        # recursively sample point on LOS curve and shrink the search space
        alt_min, alt_max, alt_diff1, alt_diff2, masks = recursive_shrink_interval(
            self, row, col, alt_min, alt_max, num_alt,
            max_iter=max_iter, eps=eps, verbosity=verbosity, elev=elev)

        # do a last linear interpolation
        alt_opt = alt_min - alt_diff1 * \
            (alt_max - alt_min) / (alt_diff2 - alt_diff1 + 1e-32)
        lon, lat, alt_opt = self.localization(row, col, alt_opt)
        return lon, lat, alt_opt, masks

    def fetch_dem(self, dem_source: eos.dem.DEMSource, roi: Optional[Roi] = None,
                  alt_min: float = -1000, alt_max: float = 9000) -> eos.dem.DEM:
        geometry = self.get_coarse_approx_geom(roi, alt_min=alt_min, alt_max=alt_max)
        lons = [P[0] for P in geometry]
        lats = [P[1] for P in geometry]
        bounds = (min(lons), min(lats), max(lons), max(lats))
        dem = dem_source.fetch_dem(bounds)
        return dem

    def get_coarse_approx_geom(self, roi: Optional[Roi] = None, margin: int = 1000, alt_min=-1000., alt_max=9000.):
        """
        Get the very approximate geometry in epsg:4326 of a roi in an image whose
        localization function is defined by a model. Localization is conducted
        assuming coarse elevation bounds: alt_min in near-range, alt_max in far-range.

        Parameters
        ----------
        roi : eos.sar.roi.Roi, optional
            Defines the region to localize. The default is None.
        margin : int, optional
            Margin in px to buffer the roi. The default is 1000.

        Returns
        -------
        approx_geom: list of tuples
            The 4 (lon, lat) corners of the very approximate geometry
        """
        if roi is None:
            roi = Roi(0, 0, self.w, self.h)
        if margin:
            roi = roi.add_margin(margin)

        rows, cols = roi.to_bounding_points()
        alts = [alt_min, alt_max, alt_max, alt_min]
        lons, lats, alts = self.localization(rows, cols, alts)

        approx_geom = [(lon, lat) for lon, lat in zip(lons, lats)]
        return approx_geom

    def get_approx_geom(self, roi=None, margin=0, *, elev, **kwargs):
        """
        Get the approximate geometry in epsg:4326 of a roi in an image whose
        localization function is defined by a model. Localization is conducted
        without knowledge of the altitude.

        Parameters
        ----------
        roi : eos.sar.roi.Roi, optional
            Defines the region to localize. The default is None.
        margin : int, optional
            Margin in px to buffer the roi. The default is 0.
       **kwargs: Keyword arguments for localize_without_alt function.

        Returns
        -------
        approx_geom: list of tuples
            The 4 (lon, lat) corners of the approximate geometry
        alts: ndarray
            The altitude for each point
        masks : dict
                keys: "zeros", "converged", "invalid"
                as returned by localize_without_alt
        """
        if roi is None:
            roi = Roi(0, 0, self.w, self.h)
        if margin:
            roi = roi.add_margin(margin)

        rows, cols = roi.to_bounding_points()

        lons, lats, alts, masks = self.localize_without_alt(
            rows, cols, elev=elev, **kwargs)

        approx_geom = [(lon, lat) for lon, lat in zip(lons, lats)]

        if np.any(masks["invalid"]):
            logger.warning("get_approx_geom: some points may be invalid.")
        return approx_geom, alts, masks

    def get_buffered_geom(self, dem: eos.dem.DEM, roi=None, margin=0, row_sampling=50, **kwargs):
        """
        Get the approximate geometry in epsg:4326 of a roi in an image whose
        localization function is defined by a model. Localization is conducted
        without knowledge of the altitude. This function will yield a geometry
        that will contain the roi with a higher probability than get_approx_geom. However,
        it is heavier to run.

        Parameters
        ----------
        roi : eos.sar.roi.Roi, optional
            Defines the region to localize. The default is None.
        dem: eos.sar.DEM
        margin : int, optional
            Margin in px to buffer the roi. The default is 0.
        row_sampling : int, optional
            The boundary of the roi will be sampled each row_sampling pixels for the altitude
            evaluation. Decreasing this value increases precision but slows the function.
        **kwargs: Keyword arguments for localize_without_alt function.

        Returns
        -------
        buffered_geom: list of tuples
            The 4 (lon, lat) corners of the geometry
        """
        if roi is None:
            roi = Roi(0, 0, self.w, self.h)
        if margin:
            roi = roi.add_margin(margin)

        col, row, w, h = roi.to_roi()
        _rows = np.arange(row, row + h, row_sampling)
        if _rows[-1] != row + h - 1:
            _rows = np.append(_rows, row + h - 1)

        # deal with the left boundary
        lons_left, lats_left, alts, masks = self.localize_without_alt(
            _rows, col * np.ones_like(_rows), elev=dem.elevation, **kwargs)

        if np.any(masks["invalid"]):
            logger.warning("get_buffered_geom: some points may be invalid.")

        min_alt = np.amin(alts)

        # deal with the right boundary
        lons_right, lats_right, alts, masks = self.localize_without_alt(
            _rows, (col + w - 1) * np.ones_like(_rows), elev=dem.elevation, **kwargs)

        if np.any(masks["invalid"]):
            logger.warning("get_buffered_geom: some points may be invalid.")

        max_alt = np.amax(alts)

        rows, cols = roi.to_bounding_points()

        lons, lats, _ = self.localization(rows, cols, [min_alt, max_alt, max_alt, min_alt],
                                          x_init=[lons_left[0], lons_right[0], lons_right[-1], lons_left[-1]],
                                          y_init=[lats_left[0], lats_right[0], lats_right[-1], lats_left[-1]],
                                          z_init=[min_alt, max_alt, max_alt, min_alt])

        buffered_geom = [(lon, lat) for lon, lat in zip(lons, lats)]

        return buffered_geom


def localized_vs_dem(sensor_model, row, col, alt, elev):
    """
    Computes the error between localized point at a certain height and the dem.

    Parameters
    ----------
    sensor_model : SensorModel instance
        Model to project or localize points.
    row : ndarray
        row in image.
    col : ndarray
        col in image.
    alt : ndarray
        Altitude to test vs dem.
    elev: function
        Function elev(lon, lat) ---> alt w.r.t. ellipsoidal datum.
        lon, lat are in wgs84 crs.
        Should be vectorized (able to handle a query of arrays).

    Returns
    -------
    ndarray
        alt - dem at localized point Loc(row, col, alt) .

    """
    lon, lat, _ = sensor_model.localization(row, col, alt)
    return alt - elev(lon, lat)


def shrink_interval(sensor_model, rows, cols, alts_min, alts_max, num_alt,
                    *, elev):
    """
    Shrink a search interval by num_alt

    Parameters
    ----------
    sensor_model : SensorModel instance
        model to perform projection and localization.
    rows : ndarray
        row in the image.
    cols : ndarray
        col in the image.
    alts_min : ndarray
        lower bound of the search space.
    alts_max : ndarray
        upper bound of the search space.
    num_alt : int
        number of altitudes to test.
    elev: function
        Function elev(lon, lat) ---> alt w.r.t. ellipsoidal datum.
        lon, lat are in wgs84 crs.
        Should be vectorized (able to handle a query of arrays).

    Returns
    -------
    _alts_min : ndarray
        New lower bound of the search space.
    _alts_max : ndarray
        New upper bound of the search space.
    alts_diff1 : ndarray
        Difference of alts_min with the dem .
    alts_diff2 : ndarray
        Difference of alts_max with the dem .
    masks : dict
        masks["zeros"] indicates points where _alts_min = _alts_max = dem.
        masks["valid"] indicates points where the search space was shrinked
        masks["invalid"] indicates points where the true solution lies outside
        of the search space.

    """
    # TODO to make this call faster, localization seems to have a bottelneck
    # related to the pyproj transformer

    # rows (N,)
    # cols (N,)
    # alts_min (N,)
    # alts_max (N,)
    # num_alt float

    # Take num_alt steps
    potential_alt = np.linspace(
        alts_min, alts_max, num_alt, axis=1)  # N x num_alt

    # take actual localized points at different heights
    # Check height diff w.r.t. dem
    alt_diff = localized_vs_dem(sensor_model,
                                utils.hrepeat(rows, num_alt).ravel(),
                                utils.hrepeat(cols, num_alt).ravel(),
                                potential_alt.ravel(),
                                elev)

    alt_diff = alt_diff.reshape(potential_alt.shape)  # N x num_alt

    # check if any of the potential alts
    # yielded points exactly on the dem
    zero_id = utils.first_nonzero(alt_diff == 0, axis=1)
    zero_mask = zero_id != -1

    # Check for sign change in alt_diff
    id_best = utils.first_nonzero(np.diff(np.sign(alt_diff)), axis=1)

    # points that don't have a zero crossing anywhere
    # this can only happen if your alt_min, alt_max range doesn't intersect
    # the dem
    invalid_mask = id_best == -1

    # points remaining where we can shrink the interval
    valid_mask = np.logical_not(np.logical_or(invalid_mask, zero_mask))

    # ids useful for shrinking the alt_min alt_max extent
    lower_ids = np.zeros(rows.shape, dtype=int)
    upper_ids = np.zeros(rows.shape, dtype=int)
    for j in range(len(rows)):
        if valid_mask[j]:
            lower_ids[j] = id_best[j]
            upper_ids[j] = id_best[j] + 1
        elif zero_mask[j]:
            lower_ids[j] = zero_id[j]
            upper_ids[j] = zero_id[j]
        else:
            lower_ids[j] = 0
            upper_ids[j] = -1

    # fill the desired quantities
    line_ids = np.arange(len(rows))
    _alts_min = potential_alt[line_ids, lower_ids]
    _alts_max = potential_alt[line_ids, upper_ids]
    alts_diff1 = alt_diff[line_ids, lower_ids]
    alts_diff2 = alt_diff[line_ids, upper_ids]
    masks = {"zeros": zero_mask, "valid": valid_mask,
             "invalid": invalid_mask}
    return _alts_min, _alts_max, alts_diff1, alts_diff2, masks


def recursive_shrink_interval(sensor_model, row, col, alt_min, alt_max,
                              num_alt, max_iter=10, eps=1e-1,
                              verbosity=False, *, elev):
    """
    Iteratively shrink the search interval for the altitude of a point.

    Parameters
    ----------
    sensor_model : SensorModel instance
        model to perform projection and localization.
    row : ndarray
        row in the image.
    col : ndarray
        col in the image.
    alt_min : ndarray
        lower bound of the search space.
    alt_max : ndarray
        upper bound of the search space.
    num_alt : int
        number of altitudes to test.
    max_iter : int, optional
        maximum number of iterations. The default is 10.
    eps : float, optional
        Width of the search interval needed to stop the iterations.
        The default is 1e-1.
    verbosity : bool, optional
        if True, print optim messages. The default is False.
    elev: function
        Function elev(lon, lat) ---> alt w.r.t. ellipsoidal datum.
        lon, lat are in wgs84 crs.
        Should be vectorized (able to handle a query of arrays).

    Returns
    -------
    alts_min : ndarray
        Lower bound of the search interval.
    alts_max : ndarray
        Upper bound of the search interval.
    alts_diff1 : ndarray
        alts_min - dem.
    alts_diff2 : ndarray
        alts_max - dem.
    masks : dict
        Masks on the returned points to indicate the status of the optim.
        mask["zeros"] indicates points that have exactly converged (error of 0)
        mask["converged"] indicates points that converged within eps tolerance.
        mask["invalid"] indicates the points for which no solution was found,
        probably because the actual altitude is outside the given range.

    """
    row = np.atleast_1d(row)
    col = np.atleast_1d(col)
    alts_min = np.ones(row.shape, dtype=float) * alt_min
    alts_max = np.ones(row.shape, dtype=float) * alt_max
    iterate_mask = np.ones(row.shape, dtype=bool)
    zero_mask = np.ones(row.shape, dtype=bool)
    invalid_mask = np.ones(row.shape, dtype=bool)
    converged_mask = np.ones(row.shape, dtype=bool)
    alts_diff1 = np.zeros(row.shape, dtype=float)
    alts_diff2 = np.zeros(row.shape, dtype=float)

    for j in range(max_iter):
        am1, am2, ad1, ad2, masks = shrink_interval(
            sensor_model,
            row[iterate_mask],
            col[iterate_mask],
            alts_min[iterate_mask],
            alts_max[iterate_mask],
            num_alt, elev=elev)
        # update masks
        zero_mask[iterate_mask] = masks["zeros"]
        invalid_mask[iterate_mask] = masks["invalid"]
        # stop iterating on some data
        _converged_mask = np.zeros(masks["valid"].shape, dtype=bool)
        _converged_mask[masks["valid"]] = (
            am2[masks["valid"]] - am1[masks["valid"]]) < eps
        converged_mask[iterate_mask] = _converged_mask
        _converged_mask[masks["invalid"]] = True
        _converged_mask[masks["zeros"]] = True
        # update the alts_min, max
        alts_min[iterate_mask] = am1
        alts_max[iterate_mask] = am2
        # update alts diff
        alts_diff1[iterate_mask] = ad1
        alts_diff2[iterate_mask] = ad2
        # update iterate mask
        iterate_mask[iterate_mask] = np.logical_not(_converged_mask)
        # if all converged, stop iterations
        if not np.any(iterate_mask):
            if verbosity:
                logger.info("Stopped after {} iterations on all points".format(j + 1))
            break

    # check if a scalar needs to be returned
    if len(row) == 1:
        alts_min, alts_max, alts_diff1, alts_diff2, zero_mask,\
            invalid_mask, converged_mask = alts_min[0], alts_max[0], alts_diff1[0],\
            alts_diff2[0], zero_mask[0], invalid_mask[0], converged_mask[0]

    masks = {"zeros": zero_mask, "invalid": invalid_mask,
             "converged": converged_mask}

    return alts_min, alts_max, alts_diff1, alts_diff2, masks
