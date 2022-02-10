import numpy as np
import pyproj
from eos.sar import poly


def normalize(vec):
    '''
    normalize vec so that norm(Vec) = 1
    vec (n, 3)
    '''
    norm = np.linalg.norm(vec, axis=1).reshape(-1, 1)
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
    line_of_sight = (points - prim)
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


def get_geom_config(primary_model, secondary_models, grid_size_col=20,
                    grid_size_row=20, train=True):
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
    perp_baseline : ndarray (N, )
        Perpendicular baseline.
    delta_r : ndarray (N, )
        Parallel baseline.

    """
    # construct grid on which we estimate each parameter
    col, row = get_grid(primary_model.w, primary_model.h, grid_size_col,
                        grid_size_row, train=train)
    rows = row.ravel()
    cols = col.ravel()
    points = np.column_stack([cols, rows])
    num_points = len(rows)

    # coordinates in image to az time and range (for later)
    azt, _ = primary_model.to_azt_rng(rows, cols)

    # geometric config parameters estim start
    # localize points on ellipsoid
    alts = [0 for i in range(num_points)]
    lons, lats, _ = primary_model.localization(
        rows, cols, alts)

    # convert to geocentric cartesian
    to_gxyz = pyproj.Transformer.from_crs(
        'epsg:4326', 'epsg:4978', always_xy=True)
    gx, gy, gz = to_gxyz.transform(lons, lats, alts)
    points_3D = np.column_stack([gx, gy, gz])

    # satellite position and speed
    sat_pos_prim = primary_model.orbit.evaluate(azt, order=0)
    sat_speed_prim = primary_model.orbit.evaluate(azt, order=1)

    # distances with earth center
    op = np.linalg.norm(points_3D, axis=1)
    _os = np.linalg.norm(sat_pos_prim, axis=1)
    ps = np.linalg.norm(sat_pos_prim - points_3D, axis=1)

    # local incidence on ellipsoid estimation
    theta_inc = np.arccos((_os**2 - op**2 - ps**2) / (2 * op * ps))

    perp_baseline = np.zeros((len(secondary_models), num_points))
    delta_r = np.zeros((len(secondary_models), num_points))

    for sid, secondary_model in enumerate(secondary_models):
        # use projection to get position of closest approach on secondary orbit
        rows_sec, cols_sec, _ = secondary_model.projection(lons, lats, alts)
        azt_sec, _ = secondary_model.to_azt_rng(rows_sec, cols_sec)

        sat_pos_sec = secondary_model.orbit.evaluate(azt_sec, order=0)
        ps_sec = np.linalg.norm(sat_pos_sec - points_3D, axis=1)

        # Calculation of baseline for each defined pixel
        _, _perp_baseline, _ = compute_baseline(
            sat_pos_prim, sat_pos_sec, points_3D, sat_speed_prim)
        perp_baseline[sid] = _perp_baseline
        delta_r[sid] = ps.ravel() - ps_sec.ravel()

    return points, theta_inc, perp_baseline, delta_r


class GeometryPredictor:
    '''
    Used to predict the geometry (baselines, incidence angles)
    '''

    def __init__(self, primary_model, secondary_models, grid_size=20, degree=7):
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
        points_train, theta_inc_train, perp_baseline_train, delta_r_train = get_geom_config(
            primary_model, secondary_models, grid_size_col=grid_size,
            grid_size_row=grid_size, train=True)

        # fit a 2d polynomial of degree
        polynom = poly.polymodel(degree)
        ztrue = np.column_stack(
            [perp_baseline_train.T, delta_r_train.T, theta_inc_train])

        polynom.fit_poly(points_train[:, 0], points_train[:, 1], ztrue)
        self.polynom = polynom

    def check_ids(self, ids):
        """ Check that the ids are coherent with the secondary models list
        """
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
        return self.polynom.eval_poly(cols, rows,
                                      secondary_ids,
                                      grid_eval)

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
        return self.polynom.eval_poly(cols, rows,
                                      self.len_secon + np.array(secondary_ids),
                                      grid_eval)

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
        return self.polynom.eval_poly(cols, rows, [-1],
                                      grid_eval).reshape(-1, 1)
