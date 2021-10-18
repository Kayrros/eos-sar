import numpy as np
from eos.sar.geoconfig import GeometryPredictor
from eos.sar import utils, roi


class TopoCorrection:
    '''
    class to predict the Topographic and the Flatearth component 
    for a set of models 
    '''

    def __init__(self, primary_model, secondary_models, grid_size=50, degree=7):
        """
        Constructor. 

        Parameters
        ----------
        primary_model : eos.sar.model.SensorModel
            primary model relative to which the corrections are computed.
        secModelList : List of eos.sar.model.SensorModel
            secondary models list.
        grid_size : int, optional
            Geometric quantities (Baseline, incidence) are computed on a meshgrid 
            of size grid_size x grid_size on the whole primary model image extent.
            A polynomial is used to fit those quantities and interpolated everywhere else.
            Increasing the grid size might improve the accuracy of the corrections.
            The default is 50.
        degree : int, optional
            Degree of 2D polynomial used for the fit of the geometric quantities. The default is 7

        Returns
        -------
        None.

        """
        # set a geometry predictor object (predicts baselines, incidence) on swath
        self.geom_pred = GeometryPredictor(primary_model, secondary_models,
                                           grid_size=grid_size,
                                           degree=degree)

        self.wavelength = primary_model.wavelength
        self.primary_model = primary_model

    def __flat_earth(self, rows, cols, grid_eval,
                     secondary_ids=None, wrapped=False):
        """internal flat_earth prediction helper func

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
        secondary_ids : list of int, optional
            List of the secondary_models on which to predict the correction. If None, 
            the prediction is done for all elements. 
            The default is None.
        wrapped : bool, optional
            If True, the phase is wraped to [-pi , pi] . The default is False.

        Returns
        -------
        phase : ndarray (numImgs, npts)
            A phase correction for each image.

        """
        par_baseline = self.geom_pred.predict_par_baseline(
            rows, cols,
            secondary_ids,
            grid_eval).T
        phase = -4 * np.pi * par_baseline / self.wavelength
        if wrapped:
            phase = utils.wrap(phase)
        return phase

    def sparse_flat_earth(self, rows, cols,
                          secondary_ids=None, wrapped=False):
        """
        Predict the flat earth phase on a sparse set of points. 

        Parameters
        ----------
        rows: ndarray
            rows on which to predict (npts, )
        cols: ndarray
            cols on which to predict (npts)
        secondary_ids : list of int, optional
            List of the secondary_models on which to predict the correction. If None, 
            the prediction is done for all elements. 
            The default is None.
        wrapped : bool, optional
            If True, the phase is wraped to [-pi , pi] . The default is False.

        Returns
        -------
        flat_earth: ndarray (nImgs, npts)
            Flat earth phase per secondary image and per point.

        """
        return self.__flat_earth(rows, cols, grid_eval=False,
                                 secondary_ids=secondary_ids,
                                 wrapped=wrapped)

    def flat_earth_image(self, primary_roi=None, secondary_ids=None,
                         wrapped=False):
        """
        Flat earth prediction on image. 

        Parameters
        ----------
        primary_roi : eos.sar.roi.Roi, optional
            If given, restrict the prediction on this region
            inside the primary image. Otherwise, predict on the whole image. 
            The default is None.
        secondary_ids : list of int, optional
            List of the secondary_models on which to predict the correction. If None, 
            the prediction is done for all elements. 
            The default is None.
        wrapped : bool, optional
            If True, the phase is wraped to [-pi , pi] . The default is False.

        Returns
        -------
        Flat_earth: ndarray (nImgs, h, w)
            Flat earth prediction per secondary image.

        """
        if primary_roi is None:
            primary_roi = roi.Roi(0, 0, self.primary_model.w, self.primary_model.h)
        col, row, w, h = primary_roi.to_roi()
        flat_earth = self.__flat_earth(np.arange(row, row+h), np.arange(col, col+w),
                                       grid_eval=True, secondary_ids=secondary_ids,
                                       wrapped=wrapped)
        return flat_earth.reshape(flat_earth.shape[0], h, w)
        
    def topo_phase_image(self, heights, primary_roi=None, secondary_ids=None,
                         wrapped=False):
        """
        Computes the topographic phase for an image.

        Parameters
        ----------
        heights : ndarray (h, w) 
            Heights of each pixel in the primary radar coordinates.
        primary_roi : eos.sar.roi.Roi
            If given, restrict the prediction on this region
            inside the primary image. Otherwise, predict on the whole image. 
            The default is None.
        secondary_ids : list of int, optional
            List of the secondary_models on which to predict the correction. If None, 
            the prediction is done for all elements. 
            The default is None.
        wrapped : bool, optional
            If True, the phase is wraped to [-pi , pi] . The default is False.

        Returns
        -------
        ndarray (nImgs, h, w) where nImgs is the number of secondary images
            Topographic phase per secondary image.

        """
        if primary_roi is None:
            primary_roi = roi.Roi(0, 0, self.primary_model.w, self.primary_model.h)
        col, row, w, h = primary_roi.to_roi()
        
        assert heights.shape == (h, w), "heights array is not consistent with primary_roi"
    
        # perp_baseline prediction
        perp_baseline = self.geom_pred.predict_perp_baseline(
            np.arange(row, row+h), np.arange(col, col+w),
            grid_eval=True,
            secondary_ids = secondary_ids).T.reshape(-1, h, w) # N, h, w
        
        # incidence prediction
        incidence = self.geom_pred.predict_incidence(
            np.arange(row, row+h), np.arange(col, col+w),
            grid_eval=True,
            ).reshape(h, w) # (h, w)
        
        _, rng = self.primary_model.to_azt_rng(0, np.arange(col, col+w)) # (w,)
        
        return self.height_to_phase(heights, rng, incidence, perp_baseline, wrapped)
   
    def height_to_phase(self, heights, rng, incidence, perp_baseline, wrapped=False):
        """
        Convert height values to topographic phase. 

        Parameters
        ----------
        heights : ndarray
            height above wgs84 ellipsoid.
        rng : ndarray
            distance between point and sensor in meters.
        incidence : ndarray
            Local incidence angle for the point on the ellispoid.
        perp_baseline : ndarray
            Perpendicular baseline for the point on the ellipsoid.
        wrapped : bool, optional
            If True, return a wrapped phase. The default is False.

        Returns
        -------
        phase : ndarray
            Topographic phase.

        """
        phase = - 4 * np.pi / self.wavelength
        phase *=   heights / (rng * np.sin(incidence))
        phase = phase * perp_baseline
        if wrapped:
            phase = utils.wrap(phase)
        return phase     
                
    def sparse_topo_phase(self, heights, rows, cols,
                          secondary_ids=None, wrapped=False):
        """
        Topographic phase on a set of sparse points (not on a regular grid).

        Parameters
        ----------
        heights : ndarray
            heights of the points.
        rows : ndarray
            row position of the points.
        cols : ndarray
            col position of the points.
        secondary_ids : list of int, optional
            List of the secondary_models on which to predict the correction. If None, 
            the prediction is done for all elements. 
            The default is None.
        wrapped : bool, optional
            If True, the phase is wraped to [-pi , pi] . The default is False.

        Returns
        -------
        ndarray
            Topographic phase predicted on the sparse points.

        """
        # perp_baseline prediction
        perp_baseline = self.geom_pred.predict_perp_baseline(
            rows, cols,
            grid_eval=False,
            secondary_ids = secondary_ids).T # nImgs, npts
        
        # incidence prediction
        incidence = self.geom_pred.predict_incidence(
            rows, cols,
            grid_eval=False,
            ).ravel() # (npts, )
        
        _, rng = self.primary_model.to_azt_rng(0, cols) # (npts,)
        
        return self.height_to_phase(heights, rng, incidence, perp_baseline, wrapped)