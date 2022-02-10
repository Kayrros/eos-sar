"""Base class for all Sensor Models."""

import abc
import numpy as np
from eos.sar.orbit import Orbit
from eos.sar import utils, roi
import eos.dem

# TODO: change the functions so that they take a DemSource as parameter
elevation = eos.dem.get_any_source().elevation


class SensorModel(abc.ABC):
    """SensorModel is an abstract class that defines the expected method of\
    any eos sensor model. It is expected that this abstract will be \
    implemented for each SAR satellite."""

    azimuth_frequency: float
    range_frequency: float
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
    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None, azt_init=None):
        pass

    @abc.abstractmethod
    def localization(self, row, col, alt, crs='epsg:4326', vert_crs=None,
                     x_init=None, y_init=None, z_init=None):
        pass

    def _apd_correction(self, alt, cos_i):
        """
        Compute atmospheric path delay correction. Range shift dependent on\
            the altitude and the incidence angle.

        Parameters
        ----------
        alt : float or array
            Altitude above the wgs84 ellipsoid.
        cos_i : float or array
            Cosine of incidence angle between the LOS and the normal.
            A spherical approximation can be used to compute it.

        Returns
        -------
        drng : float or array
            Atmospheric path delay, i.e. single path additional range induced by
            the passage of the EM wave through the atmosphere.

        """
        drng = (alt * alt / 8.55e7 - alt / 3411.0 + 2.41) / cos_i
        return drng

    def localize_without_alt(self, row, col, max_iter=5, eps=1,
                             alt_min=-1000, alt_max=9000, num_alt=100,
                             verbosity=False, elev=elevation):
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

    def get_approx_geom(self, _roi=None, margin=0, **kwargs):
        """
        Get the approximate geometry in epsg:4326 of a roi in an image whose
        localization function is defined by a model. Localization is conducted
        without knowledge of the altitude.

        Parameters
        ----------
        _roi : eos.sar.roi.Roi, optional
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
        if _roi is None:
            _roi = roi.Roi(0, 0, self.w, self.h)
        if margin:
            _roi = _roi.add_margin(margin)

        rows, cols = _roi.to_bounding_points()

        lons, lats, alts, masks = self.localize_without_alt(
            rows, cols, **kwargs)

        approx_geom = [(lon, lat) for lon, lat in zip(lons, lats)]

        if np.any(masks["invalid"]):
            print("Warning: Some points may be invalid")
        return approx_geom, alts, masks


def localized_vs_dem(sensor_model, row, col, alt, elev=elevation):
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
                    elev=elevation):
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
                              verbosity=False, elev=elevation):
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
            num_alt, elev)
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
                print("Stopped after {} iterations on all points".format(j + 1))
            break

    # check if a scalar needs to be returned
    if len(row) == 1:
        alts_min, alts_max, alts_diff1, alts_diff2, zero_mask,\
            invalid_mask, converged_mask = alts_min[0], alts_max[0], alts_diff1[0],\
            alts_diff2[0], zero_mask[0], invalid_mask[0], converged_mask[0]

    masks = {"zeros": zero_mask, "invalid": invalid_mask,
             "converged": converged_mask}

    return alts_min, alts_max, alts_diff1, alts_diff2, masks
