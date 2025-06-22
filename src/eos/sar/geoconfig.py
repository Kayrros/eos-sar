from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyproj
from numpy.typing import ArrayLike

from eos.sar import const, poly
from eos.sar.model import Arrayf64, CoordArrayLike, SensorModel
from eos.sar.orbit import Orbit
from eos.sar.range_doppler import iterative_projection


def compute_cosi_rng(points, sat):
    """
    Apply cosine rule to get the cosine of the incidence angle, by
    considering the triangle formed by the center of the earth( origin of geocentric coordinate system),
    the 3D point, and the satellite.

    Parameters
    ----------
    points : ndarray (n, 3)
        Points on which we wish to get the incidence.
    sat : ndarray (n, 3)
        Satellite positions corresponding the points positions.

    Returns
    -------
    cos_i : (n,)
        Cosinus of the incidence angle.
    rng : (n, )
        Geometric distance in meters from satellite to point.

    """
    # apply the cosine rule to get the incidence angle
    op = np.linalg.norm(points, axis=1)
    os = np.linalg.norm(sat, axis=1)
    rng = np.linalg.norm(sat - points, axis=1)
    cos_i = (os**2 - op**2 - rng**2) / (2 * op * rng)
    return cos_i, rng


def normalize(vec):
    """
    normalize vec so that norm(Vec) = 1
    vec (n, 3)
    """
    norm = np.linalg.norm(vec, axis=1)[:, None]
    norm[norm == 0] = 1
    return vec / norm


def compute_baseline(prim, sec, points, speed):
    """Compute the geometric baseline.


    Parameters
    ----------
    prim : ndarray (n, 3)
        Primary satellite positions.
    sec : ndarray (n, 3)
        Secondary satellite positions.
    points : ndarray
        Points positions ( on ellipsoid assumed).
    speed : ndarray
        Speed at primary position.

    Returns
    -------
    par_baseline : ndarray
        Baseline parallel (projected on LOS).
    perp_baseline : ndarray
        Baseline perpendicular (in the zero doppler plane, orthogonal to LOS).
    depth_baseline : ndarray
        Baseline along the speed component.

    """
    # assert 2D arrays
    prim = prim.reshape(-1, 3)
    sec = sec.reshape(-1, 3)
    points = points.reshape(-1, 3)
    speed = speed.reshape(-1, 3)
    baseline = sec - prim  # baseline vec
    line_of_sight = points - prim
    # construct basis
    line_of_sight = normalize(line_of_sight)
    speed = normalize(speed)
    cross = np.cross(line_of_sight, speed)
    cross = normalize(cross)
    # projection
    par_baseline = np.sum(baseline * line_of_sight, axis=1)
    perp_baseline = np.sum(baseline * cross, axis=1)
    depth_baseline = np.sum(baseline * speed, axis=1)
    return par_baseline, perp_baseline, depth_baseline


def get_grid(width, height, grid_size_col, grid_size_row, train=True):
    """Compute a meshgrid for training and testing purposes.


    Parameters
    ----------
    width : int
        width of image.
    height : int
        height of image.
    grid_size_col : int
        number of points in the width direction.
    grid_size_row : int
        number of points in the height direction.
    train : bool, optional
        If True, a training meshgrid is returned.
        Otherwise, the translated (by half a step) testing meshgrid is returned.
        The default is True.

    Returns
    -------
    col : ndarray
        Column meshgrid.
    row : ndarray
        Row meshgrid.

    """
    if train:
        cols = np.linspace(0, width - 1, grid_size_col)
        rows = np.linspace(0, height - 1, grid_size_row)
    else:
        # estimate the gap between two samples
        col_step = width / (grid_size_col - 1)
        # translate train samples by half a step
        cols = np.linspace(col_step / 2, width - 1 + col_step / 2, grid_size_col)
        # remove the last sample that goes out of the image
        cols = cols[:-1]
        # do the same for the lines
        row_step = height / (grid_size_row - 1)
        rows = np.linspace(row_step / 2, height - 1 + row_step / 2, grid_size_row)
        rows = rows[:-1]
    col, row = np.meshgrid(cols, rows)
    return col, row


def localize_on_ellipsoid(
    proj_model: SensorModel,
    rows: CoordArrayLike,
    cols: CoordArrayLike,
    alt: float = 0.0,
) -> Arrayf64:
    # localize points on ellipsoid
    alts = np.full_like(rows, fill_value=alt, dtype=np.float64)

    gx, gy, gz = proj_model.localization(rows, cols, alts, crs="epsg:4978")
    # convert to geocentric cartesian
    points_3D = np.column_stack([gx, gy, gz])
    return points_3D


def get_geom_config(
    primary_model: SensorModel,
    secondary_models: Sequence[SensorModel],
    grid_size_col: int = 20,
    grid_size_row: int = 20,
    train: bool = True,
):
    """Get the geometric configuration at a meshgrid train/test set.


    Parameters
    ----------
    primary_model : eos.sar.model.SensorModel
        Primary model w.r.t. which quantities are computed.
    secondary_models : List of eos.sar.model.SensorModel
        Secondary models list.
    grid_size_col : int, optional
        The number of points in the width direction. The default is 20.
    grid_size_row : int, optional
        The number of points in the height direction. The default is 20.
    train : bool, optional
        Train or test set. The default is True.

    Returns
    -------
    points : ndarray (N, 2)
        Image points locations (col, row).
    theta_inc : ndarray (N, )
        Incidence angle.
    perp_baseline : ndarray (n_sec, N)
        Perpendicular baseline.
    delta_r : ndarray (n_sec, N)
        Parallel baseline.

    """
    # construct grid on which we estimate each parameter
    col, row = get_grid(
        primary_model.w, primary_model.h, grid_size_col, grid_size_row, train=train
    )
    rows = row.ravel()
    cols = col.ravel()
    points = np.column_stack([cols, rows])
    theta_inc, perp_baseline, delta_r = get_geom_config_from_grid_coords(
        primary_model, secondary_models, rows, cols
    )
    return points, theta_inc, perp_baseline, delta_r


def get_geom_config_from_grid_coords(
    primary_model: SensorModel,
    secondary_models: Sequence[SensorModel],
    rows: CoordArrayLike,
    cols: CoordArrayLike,
) -> tuple[Arrayf64, Arrayf64, Arrayf64]:
    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)

    # geometric config parameters estim start
    points_3D = localize_on_ellipsoid(primary_model, rows, cols, 0.0)

    # coordinates in image to az time and range
    azt, _ = primary_model.to_azt_rng(rows, cols)
    # satellite position and speed
    sat_pos_prim = primary_model.orbit.evaluate(azt, order=0)
    sat_speed_prim = primary_model.orbit.evaluate(azt, order=1)

    cos_i, ps = compute_cosi_rng(points_3D, sat_pos_prim)

    # local incidence on ellipsoid estimation
    theta_inc = np.arccos(cos_i)

    num_points = len(rows)
    perp_baseline = np.zeros((len(secondary_models), num_points))
    delta_r = np.zeros((len(secondary_models), num_points))

    for sid, secondary_model in enumerate(secondary_models):
        # use projection to get position of closest approach on secondary orbit
        azt_sec, _, _ = secondary_model.projection(
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            crs="epsg:4978",
            as_azt_rng=True,
        )

        sat_pos_sec = secondary_model.orbit.evaluate(azt_sec, order=0)
        ps_sec = np.linalg.norm(sat_pos_sec - points_3D, axis=1)

        # Calculation of baseline for each defined pixel
        _, _perp_baseline, _ = compute_baseline(
            sat_pos_prim, sat_pos_sec, points_3D, sat_speed_prim
        )
        perp_baseline[sid] = _perp_baseline
        delta_r[sid] = ps.ravel() - ps_sec.ravel()

    return theta_inc, perp_baseline, delta_r


class GeometryPredictor:
    """
    Used to predict the geometry (baselines, incidence angles)
    """

    def __init__(
        self,
        primary_model: SensorModel,
        secondary_models: Sequence[SensorModel],
        grid_size: int = 20,
        degree: int = 7,
    ):
        """


        Parameters
        ----------
        primary_model : eos.sar.model.SensorModel
            Primary model w.r.t. which quantities are computed.
        secondary_models : List of eos.sar.model.SensorModel
            Secondary models list.
        grid_size : int, optional
            Defines sparse grid of points on which geometric quantities are
            estimated, and on which we will fit a 2D polynomial later on.
            The default is 20.
        degree : int, optional
            Degree of 2D polynomial that fits the geometric quantities as a
            function of their (row, col) position in the swath. The default is 7.
        Returns
        -------
        None.

        """
        self.len_secon = len(secondary_models)
        self.grid_size = grid_size
        self.degree = degree

        # training set to fit the geometric config
        (
            points_train,
            theta_inc_train,
            perp_baseline_train,
            delta_r_train,
        ) = get_geom_config(
            primary_model,
            secondary_models,
            grid_size_col=grid_size,
            grid_size_row=grid_size,
            train=True,
        )

        # fit a 2d polynomial of degree
        polynom = poly.polymodel(degree)
        ztrue = np.column_stack(
            [perp_baseline_train.T, delta_r_train.T, theta_inc_train]
        )

        polynom.fit_poly(points_train[:, 0], points_train[:, 1], ztrue)
        self.polynom = polynom

    def check_ids(self, ids):
        """Check that the ids are coherent with the secondary models list"""
        if ids is None:
            ids = np.arange(self.len_secon)
        else:
            ids = np.array(ids)
            ids = ids[np.logical_and(ids >= 0, ids < self.len_secon)]
        return ids

    def predict_perp_baseline(self, rows, cols, secondary_ids=None, grid_eval=False):
        """Predicts the perpendicular baseline on a set of debursted points.


        Parameters
        ----------
        rows: ndarray
            rows on which to predict
        cols: ndarray
            cols on which to predict
        secondary_ids : list of int, optional
            List of ids for which to predict the baseline.
            If None, predict everywhere.
            The default is None.
        grid_eval : bool, optional
            If set to True, the polynomial is evaluated at the cartesian
            product of rows, cols. Otherwise, the polynomial is evaluated at
            the points defined by [rows, cols].

        Returns
        -------
        ndarray
            Perpendicular baseline for each point and each image (numpts, numImgs).

        """
        secondary_ids = self.check_ids(secondary_ids)
        return self.polynom.eval_poly(cols, rows, secondary_ids, grid_eval)

    def predict_par_baseline(self, rows, cols, secondary_ids=None, grid_eval=False):
        """Compute the parallel baseline on a set of debursted points.


        Parameters
        ----------
        rows: ndarray
            rows on which to predict
        cols: ndarray
            cols on which to predict
        secondary_ids : list of int, optional
            List of ids for which to predict the baseline.
            If None, predict everywhere.
            The default is None.
        grid_eval : bool, optional
            If set to True, the polynomial is evaluated at the cartesian
            product of rows, cols. Otherwise, the polynomial is evaluated at
            the points defined by [rows, cols].


        Returns
        -------
        ndarray
            Parallel baseline for each point and each image (numpts, numImgs).

        """
        secondary_ids = self.check_ids(secondary_ids)
        return self.polynom.eval_poly(
            cols, rows, self.len_secon + np.array(secondary_ids), grid_eval
        )

    def predict_incidence(self, rows, cols, grid_eval=False):
        """Compute the incidence angle for a set of debursted points.


        Parameters
        ----------
        rows: ndarray
            rows on which to predict
        cols: ndarray
            cols on which to predict
        grid_eval : bool, optional
            If set to True, the polynomial is evaluated at the cartesian
            product of rows, cols. Otherwise, the polynomial is evaluated at
            the points defined by [rows, cols].

        Returns
        -------
        ndarray
            Incidence angle for each point on the primary reference frame.

        """
        return self.polynom.eval_poly(cols, rows, [-1], grid_eval).reshape(-1, 1)


def get_los_on_ellipsoid(
    proj_model: SensorModel,
    rows: CoordArrayLike,
    cols: CoordArrayLike,
    alt: float = 0.0,
    *,
    normalized: bool = True,
) -> tuple[Arrayf64, Arrayf64]:
    """
    Compute the LOS for a set of pixel positions on ellipsoid with alitude alt.
    This function can consume a lot of memory when len(rows) is big.
    A rough estimate is around 1 GB peak memory consumption per 2 million pixels.

    Returns
    -------
    LOS: (N, 3) NDArray[np.float64]
        vector from satellite (start) to point (end), in epsg:4978 (ECEF)
    """
    points_3D = localize_on_ellipsoid(proj_model, rows, cols, 0.0)

    # coordinates in image to az time and range
    azt, _ = proj_model.to_azt_rng(rows, cols)

    # satellite position
    sat_pos = proj_model.orbit.evaluate(azt, order=0)

    # line of sight computation
    los = points_3D - sat_pos

    if normalized:
        los = normalize(los)

    return los, points_3D


def get_los_squinted(
    points_3D: Arrayf64,
    azt: Arrayf64,
    orbit: Orbit,
    dop_centroid_freq: Arrayf64,
    wavelength: float,
    delta_azt: Optional[Arrayf64] = None,
    rng: Optional[Arrayf64] = None,
    *,
    normalized: bool = True,
) -> Arrayf64:
    """
    Computes the squinted LOS from 3D points,associated Zero Doppler times along an orbit,
    associated Doppler frequencies that determine the squint. Optionnally pass the expected
    delta_azt between sensing time and Zero-Doppler time, otherwise an estimate will be computed
    in the function.
    The returned LOS might be normalized or not.
    """
    if delta_azt is None:
        # or a less accurate estimate by supposing that the orbit is a line locally
        # and that the speed vector is constant, and drawing a right angled triangle
        v = orbit.evaluate(azt, order=1)
        sat_pos = orbit.evaluate(azt, order=0)
        if rng is None:
            rng = np.linalg.norm(points_3D - sat_pos, axis=1)
        norm_v = np.linalg.norm(v, axis=1)
        sin_squint = dop_centroid_freq * wavelength / (2 * norm_v)
        cos_squint = np.sqrt(1 - sin_squint**2)
        delta_azt = -rng * sin_squint / (cos_squint * norm_v)

    azt_sensing_rough = azt + delta_azt

    azt_sensing_exact, _, _ = iterative_projection(
        orbit,
        points_3D[:, 0],
        points_3D[:, 1],
        points_3D[:, 2],
        azt_init=azt_sensing_rough,
        half_wavelength_f_dc=wavelength / 2 * dop_centroid_freq,
    )

    pos_sat_center_beam = orbit.evaluate(azt_sensing_exact)
    los_squinted = points_3D - pos_sat_center_beam

    if normalized:
        los_squinted = normalize(los_squinted)

    return los_squinted


@dataclass(frozen=True)
class LOSPredictor:
    polynom: poly.polymodel

    @staticmethod
    def from_proj_model_grid_size(
        proj_model: SensorModel,
        grid_size_col: int,
        grid_size_row: int,
        degree: int = 7,
        alt: float = 0.0,
        *,
        normalized: bool = True,
        estimate_in_enu: bool = False,
    ) -> LOSPredictor:
        col_grid, row_grid = get_grid(
            proj_model.w,
            proj_model.h,
            grid_size_col=grid_size_col,
            grid_size_row=grid_size_row,
        )

        rows = row_grid.ravel()
        cols = col_grid.ravel()

        return LOSPredictor.from_proj_model_grid_coords(
            proj_model,
            rows,
            cols,
            degree,
            alt,
            normalized=normalized,
            estimate_in_enu=estimate_in_enu,
        )

    @staticmethod
    def from_proj_model_grid_coords(
        proj_model: SensorModel,
        rows_grid: CoordArrayLike,
        cols_grid: CoordArrayLike,
        degree: int = 7,
        alt: float = 0.0,
        *,
        normalized: bool = True,
        estimate_in_enu: bool = False,
    ) -> LOSPredictor:
        # should not consume a lot of memory if grid size is reasonable
        los, points_3D = get_los_on_ellipsoid(
            proj_model, rows_grid, cols_grid, alt=alt, normalized=normalized
        )
        if estimate_in_enu:
            los = convert_arrays_to_enu(los, points_3D, alt == 0)

        return LOSPredictor.from_los_grid_coords(los, rows_grid, cols_grid, degree)

    @staticmethod
    def from_los_grid_coords(
        los: Arrayf64,
        rows_grid: CoordArrayLike,
        cols_grid: CoordArrayLike,
        degree: int = 7,
    ) -> LOSPredictor:
        polynom = poly.polymodel(degree)
        # fit polynom
        polynom.fit_poly(cols_grid, rows_grid, los)

        return LOSPredictor(polynom)

    def predict_los(
        self, rows: CoordArrayLike, cols: CoordArrayLike, grid_eval: bool = False
    ) -> Arrayf64:
        """
        Returns
        -------
        (N, 3), where
                N=len(rows)=len(cols) if grid_eval = False
                N = len(rows) * len(cols) if grid_eval = True

        """

        return self.polynom.eval_poly(cols, rows, grid_eval=grid_eval)


def rotation_matrices(rotation_axis: ArrayLike, theta: ArrayLike) -> Arrayf64:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    # From https://stackoverflow.com/a/6802723

    Parameters
    ----------
    rotation_axis: (3,) or (N, 3) ArrayLike
        if (3,) all rotations will be w.r.t. the same axis.
        Otherwise, the number N must be broadcastable with the number of angles:
            it must match the number of angles or there must be only one angle.
    theta: float or (N,) arraylike
        Rotation angle in radians
        if float, all rotations will have the same angle. Broadcasting rules apply.

    Returns:
    --------
    rot_mats: NDArray[np.float64] (N, 3, 3)
        rotation matrices. One matrix per couple of axis and rotation angle (lenght must match),
        but broadcasting rules apply: it is possible to provide a single axis or a single angle.
    """
    rotation_axis = np.atleast_2d(np.array(rotation_axis, np.float64))
    assert rotation_axis.shape[1] == 3

    rotation_axis = normalize(rotation_axis)

    theta = np.atleast_1d(np.array(theta))
    assert len(theta.shape) == 1

    a = np.cos(theta / 2.0)

    # when one of them has 1 in shape[0], automatically broadcastable
    if rotation_axis.shape[0] != 1 and theta.shape[0] != 1:
        # otherwise, shape[0] must match
        assert rotation_axis.shape[0] == theta.shape[0], (
            "Should provide broadcastable input"
        )

    # brodcasting rules
    # rotation_axis.T is (3, 1) or (3, N)
    # theta is (1,) or (N,)

    b, c, d = -rotation_axis.T * np.sin(theta / 2.0)

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mats = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    rot_mats = np.moveaxis(rot_mats, -1, 0)

    return rot_mats


def latitude_geodetic_to_geocentric(
    lat_geod: CoordArrayLike, alt: Optional[CoordArrayLike] = None
) -> CoordArrayLike:
    """
    Convert Geodetic latitude to geocentric latitude
    See https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    Input in degree, output in degree
    """
    finv = const.EARTH_WG84_INVERSE_FLATTENING

    f = 1 / finv
    # https://en.wikipedia.org/wiki/Flattening#Identities
    e_squared = f * (2 - f)

    lat_geod_rad = np.deg2rad(lat_geod)

    if alt is None:
        # assume formula for zero altitude
        lat_geoc = np.rad2deg(np.arctan((1 - e_squared) * np.tan(lat_geod_rad)))
    else:
        # https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical
        a = const.EARTH_WGS84_AXIS_A_M
        N = a / np.sqrt(1 - e_squared * np.sin(lat_geod_rad) ** 2)
        num = N * (1 - e_squared) + alt
        denum = N + alt
        lat_geoc = np.rad2deg(np.arctan(num * np.tan(lat_geod_rad) / denum))

    return lat_geoc


def get_rot_matrices_ECEF_to_ENU(
    points_3D: Arrayf64, points_on_wgs84: bool = False
) -> Arrayf64:
    """
    Get rotation matrices from ECEF to ENU convention.

    Parameters
    ----------
    points_3D : Arrayf64
        (N, 3) in Geocentric ECEF frame.
    points_on_wgs84 : bool, optional
        Boolean indicating if the points are sampled on wgs84 ellipsoid (0 altitude).
        This knowledge simplifies the geodetic latitude formula.
        If the user does not have this info, set it to False, since it is possible to get the same
        result while ignoring the flag anyway.
        The default is False.

    Returns
    -------
    rot_matrices : Arrayf64
        (N, 3, 3) rotation matrices from ECEF to ENU.
    """
    src_crs = pyproj.crs.CRS.from_epsg(4978)
    dst_crs = pyproj.crs.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lons, lats, alts = transformer.transform(
        points_3D[:, 0], points_3D[:, 1], points_3D[:, 2]
    )

    # ECEF reference frame
    # i1 = np.array([1,0,0]) # X-axis of the Earth
    # j1 = np.array([0,1,0]) # Y-axis of the Earth
    k1 = np.array([0, 0, 1])  # Z-axis of the Earth

    # (N, 3, 3) shape, where N number of points
    Rlon_k1 = rotation_matrices(k1, np.deg2rad(lons))
    del lons
    # (N, 3) shape for each axis
    # one way to get j2 is to rotate j1
    # j2 = np.dot(Rlon_k1, j1)
    # but this is equivalent to taking the 2nd column of each rotation matrix
    j2 = Rlon_k1[:, :, 1]

    lats = latitude_geodetic_to_geocentric(lats, None if points_on_wgs84 else alts)
    # (N, 3, 3) shape, where N number of points
    Rlat_j2 = rotation_matrices(j2, np.deg2rad(-lats))  # /!\ change sign of latitude
    del lats
    del j2

    # R_UEN_to_ECEF
    rot_matrices = np.matmul(Rlat_j2, Rlon_k1)

    del Rlon_k1
    del Rlat_j2

    # R_ECEF_to_UEN
    # The inverse of an orthogonal matrix is its transpose
    rot_matrices = np.moveaxis(rot_matrices, 2, 1)

    # R_ECEF_to_ENU
    # apply permutation to lines
    rot_matrices = rot_matrices[:, [1, 2, 0], :]

    return rot_matrices


def rotate_arrays(arrays: Arrayf64, rot_matrices: Arrayf64) -> Arrayf64:
    """
    Rotate arrays with rotation matrices.

    Parameters
    ----------
    arrays : Arrayf64
        (N, 3) N arrays representing vectors in 3D space.
    rot_matrices : Arrayf64
        (N, 3, 3) rotation matrices.

    Returns
    -------
    Arrayf64
        (N, 3) rotated arrays.

    """
    return np.matmul(rot_matrices, arrays[..., None])[..., 0]


def convert_arrays_to_enu(
    arrays: Arrayf64, points_3D: Arrayf64, points_on_wgs84: bool = False
) -> Arrayf64:
    """
    Convert arrays from ECEF to ENU convention.

    Parameters
    ----------
    arrays: Arrayf64
        (N, 3) N arrays representing vectors in ECEF frame.
    points_3D : Arrayf64
        (N, 3) points associated to arrays in ECEF frame.
    points_on_wgs84 : bool, optional
        Boolean indicating if the points are sampled on wgs84 ellipsoid (0 altitude).
        This knowledge simplifies the geodetic latitude formula.
        If the user does not have this info, set it to False, since it is possible to get the same
        result while ignoring the flag anyway.
        The default is False.

    Returns
    -------
    arrays_enu: Arrayf64
        (N, 3) arrays in ENU frame.
    """
    rot_matrices = get_rot_matrices_ECEF_to_ENU(
        points_3D, points_on_wgs84=points_on_wgs84
    )
    arrays_enu = rotate_arrays(arrays, rot_matrices)

    return arrays_enu
