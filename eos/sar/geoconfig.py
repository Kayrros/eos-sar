import numpy as np
import pyproj
from eos.sar import poly


def normalize(Vec):
    '''
    normalize vec so that norm(Vec) = 1
    Vec (n, 3)
    '''
    norm = np.linalg.norm(Vec, axis=1).reshape(-1, 1)
    norm[norm == 0] = 1
    return Vec/norm


def compute_baseline(Prim, Sec, P, speed):
    """Compute the geometric baseline. 


    Parameters
    ----------
    Prim : ndarray (n, 3)
        Primary satellite positions.
    Sec : ndarray (n, 3)
        Secondary satellite positions.
    P : ndarray
        Points positions ( on ellipsoid assumed).
    speed : ndarray
        Speed at primary position.

    Returns
    -------
    Bpar : ndarray
        Baseline parallel (projected on LOS).
    Bperp : ndarray
        Baseline perpendicular (in the zero doppler plane, orthogonal to LOS).
    Bdepth : ndarray
        Baseline along the speed component.

    """
    # assert 2D arrays
    Prim = Prim.reshape(-1, 3)
    Sec = Sec.reshape(-1, 3)
    P = P.reshape(-1, 3)
    speed = speed.reshape(-1, 3)
    B = Sec - Prim  # baseline vec
    LOS = (P - Prim)
    # construct basis
    LOS = normalize(LOS)
    speed = normalize(speed)
    cross = np.cross(LOS, speed)
    cross = normalize(cross)
    # projection
    Bpar = np.sum(B * LOS, axis=1)
    Bperp = np.sum(B * cross, axis=1)
    Bdepth = np.sum(B * speed, axis=1)
    return Bpar, Bperp, Bdepth


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
    Col : ndarray
        Column meshgrid.
    Row : ndarray
        Row meshgrid.

    """
    if train:
        cols = np.linspace(0, width-1, grid_size_col)
        rows = np.linspace(0, height-1, grid_size_row)
    else:
        # estimate the gap between two samples
        col_step = width/(grid_size_col - 1)
        # translate train samples by half a step
        cols = np.linspace(col_step/2, width-1 + col_step/2, grid_size_col)
        # remove the last sample that goes out of the image
        cols = cols[:-1]
        # do the same for the lines
        row_step = height/(grid_size_row - 1)
        rows = np.linspace(row_step/2, height - 1 + row_step/2, grid_size_row)
        rows = rows[:-1]
    Col, Row = np.meshgrid(cols, rows)
    return Col, Row


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
    Bperp : ndarray (N, )
        Perpendicular baseline.
    delta_r : ndarray (N, )
        Parallel baseline.

    """
    # construct grid on which we estimate each parameter
    Col, Row = get_grid(primary_model.w, primary_model.h, grid_size_col,
                        grid_size_row, train=train)
    rows = Row.ravel()
    cols = Col.ravel()
    points = np.column_stack([cols, rows])
    num_points = len(rows)

    # coordinates in image to az time and range (for later)
    azt, rng = primary_model.to_azt_rng(rows, cols)

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

    Bperp = np.zeros((len(secondary_models), num_points))
    delta_r = np.zeros((len(secondary_models), num_points))

    for sid, secondary_model in enumerate(secondary_models):
        # use projection to get position of closest approach on secondary orbit
        rows_sec, cols_sec, _ = secondary_model.projection(lons, lats, alts)
        azt_sec, rng_sec = secondary_model.to_azt_rng(rows_sec, cols_sec)

        sat_pos_sec = secondary_model.orbit.evaluate(azt_sec, order=0)

        # Calculation of baseline for each defined pixel
        _, _Bperp, _ = compute_baseline(
            sat_pos_prim, sat_pos_sec, points_3D, sat_speed_prim)
        Bperp[sid] = _Bperp
        delta_r[sid] = ps.ravel() - rng_sec.ravel()

    return points, theta_inc, Bperp, delta_r


class geometryPredictor:
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
        points_Train, theta_inc_Train, Bperp_Train, delta_r_Train = get_geom_config(
            primary_model, secondary_models, grid_size_col=grid_size,
            grid_size_row=grid_size, train=True)

        # fit a 2d polynomial of degree
        polynom = poly.polymodel(degree)
        ztrue = np.column_stack(
            [Bperp_Train.T,  delta_r_Train.T, theta_inc_Train])

        polynom.fit_poly(points_Train[:, 0], points_Train[:, 1], ztrue)
        self.polynom = polynom

    def checkIds(self, Ids):
        """ Check that the ids are coherent with the secondary models list
        """
        if Ids is None:
            Ids = np.arange(self.len_secon)
        else:
            Ids = np.array(Ids)
            Ids = Ids[np.logical_and(Ids >= 0, Ids < self.len_secon)]
        return Ids

    def predict_Bperp(self, rows, cols, secondary_ids=None, grid_eval=False):
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
        secondary_ids = self.checkIds(secondary_ids)
        return self.polynom.eval_poly(cols, rows, 
                                      secondary_ids,
                                      grid_eval)

    def predict_Bpar(self, rows, cols, secondary_ids=None, grid_eval=False):
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
        secondary_ids = self.checkIds(secondary_ids)
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
