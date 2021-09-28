"""Base class for all Sensor Models."""

import abc
import numpy as np
from multidem import elevation
from eos.sar.orbit import Orbit
from eos.sar import utils


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
    def projection(self, x, y, alt, crs='epsg:4326', vert_crs=None):
        pass

    @abc.abstractmethod
    def localization(self, row, col, alt, crs='epsg:4326', vert_crs=None):
        pass

    def localize_without_alt(self, row, col, max_iter=5, eps=1,
                             alt_min=-1000, alt_max=9000, num_alt=100,
                             verbosity=False):

        # recursively sample point on LOS curve and shrink the search space
        alt_min, alt_max, alt_diff1, alt_diff2, masks = recursive_shrink_interval(
            self, row, col, alt_min, alt_max, num_alt,
            max_iter=max_iter, eps=eps, verbosity=verbosity)
        
        # do a last linear interpolation
        alt_opt = alt_min - alt_diff1 * \
            (alt_max - alt_min)/(alt_diff2 - alt_diff1 + 1e-32)
        lon, lat, alt_opt = self.localization(row, col, alt_opt)
        return lon, lat, alt_opt, masks


def localized_vs_dem(sensor_model, row, col, alt):
    # TODO remove dependency on multidem elevation (io will be faster)
    lon, lat, _ = sensor_model.localization(row, col, alt)
    return alt - elevation(lon, lat)


def shrink_interval(sensor_model, rows, cols, alts_min, alts_max, num_alt,
                    ):
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
                                potential_alt.ravel())

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
                              num_alt, max_iter=10, eps=1e-1, verbosity=False):
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
            num_alt)
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
                print("Stopped after {} iterations on all points".format(j+1))
            break

    # check if a scalar needs to be returned
    if len(row) == 1:
        alts_min, alts_max, alts_diff1, alts_diff2, zero_mask,\
            invalid_mask, converged_mask = alts_min[0], alts_max[0], alts_diff1[0],\
            alts_diff2[0], zero_mask[0], invalid_mask[0], converged_mask[0]

    masks = {"zeros": zero_mask, "invalid": invalid_mask,
             "converged": converged_mask}

    return alts_min, alts_max, alts_diff1, alts_diff2, masks
