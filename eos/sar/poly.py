import numpy as np


class polymodel:
    """2D polynomial model"""

    def __init__(self, degree):
        """
        Constructor

        Parameters
        ----------
        degree : int
            degree of polynomial.

        Returns
        -------
        None.

        """
        self.num_coeffs = int((degree + 1) * (degree + 2) / 2)
        self.d = degree
        pow_x = np.zeros((self.num_coeffs,), dtype=int)
        pow_y = np.zeros((self.num_coeffs,), dtype=int)
        k = 0
        for i in range(self.d + 1):
            for j in range(i + 1):
                pow_x[k] = i - j
                pow_y[k] = j
                k += 1
        self.pow_x = pow_x
        self.pow_y = pow_y

    def _get_powers(self, vec, deg):
        """Computes vec^pow for pow:0->deg


        Parameters
        ----------
        vec : ndarray
            vector for which we which to get the powers.
        deg : int
            Max degree of power.

        Returns
        -------
        powers : ndarray (len(vec), deg + 1)
            Each column is a power of the vector.

        """
        powers = np.zeros((len(vec), deg + 1))
        powers[:, 0] = np.ones((len(vec),))
        for deg in range(1, deg + 1):
            powers[:, deg] = powers[:, deg - 1] * vec.ravel()
        return powers

    def _design_mat(self, x, y):
        """Computes the design matrix containing all the powers of the polynomial
        x^(i-j) y^(j) for i 0 -> deg, j 0 -> i

        Parameters
        ----------
        x : ndarray
            x coordinate.
        y : ndarray
            y coordinate.

        Returns
        -------
        design_mat : ndarray (numpts, numPolycoeffs)
            Design matrix to predict the polynomial (only need to multiply it by the
                                                     poly coeffs).

        """
        xpowers = self._get_powers(x, self.d)
        ypowers = self._get_powers(y, self.d)
        design_mat = np.zeros((len(x), self.num_coeffs))
        for i in range(self.num_coeffs):
            design_mat[:, i] = xpowers[:, self.pow_x[i]] * \
                ypowers[:, self.pow_y[i]]
        return design_mat

    def _normalization(self, vec):
        '''
        normalize between -2 & 2
        '''
        a = np.amin(vec, axis=0)
        b = np.amax(vec, axis=0)
        off = (b + a) / 2
        scale = (b - a) / 4
        return off, scale

    def set_normalization(self, x, y, z):
        '''Initialize the normalization offset and scale;
        '''
        self.xoff, self.xscale = self._normalization(x)
        self.yoff, self.yscale = self._normalization(y)
        self.zoff, self.zscale = self._normalization(z)

    def _normalize(self, vec, off, scale):
        '''normalize the vector with offset and scale '''
        return (vec - off) / scale

    def _unnormalize(self, vec, off, scale, indices=None):
        '''un-normalize vector with offset and scale '''
        if indices is None:
            indices = np.arange(scale.shape[1])
        return vec * scale[indices] + off[indices]

    def fit_poly(self, x, y, z):
        """Fit a 2D polynomial that predicts z from x, y.


        Parameters
        ----------
        x : ndarray (n, )
            x coordinate.
        y : ndarray (n, )
            y coordinate.
        z : (n, zdim)
            Predicted quantity, can be multidimentionnal (zdim) , this
            way, we predict many quantities at the same time.

        Returns
        -------
        None.

        """
        assert(len(x) > self.num_coeffs), 'unsufficient points for fit'
        self.set_normalization(x, y, z)
        x = self._normalize(x, self.xoff, self.xscale)
        y = self._normalize(y, self.yoff, self.yscale)
        z = self._normalize(z, self.zoff, self.zscale)
        design_mat = self._design_mat(x, y)
        U, s, Vt = np.linalg.svd(design_mat, full_matrices=False)
        idx = s > 1e-15  # same default value as scipy.linalg.pinv
        s_nnz = s[idx][:, np.newaxis]
        UTz = np.dot(U.T, z)
        d = np.zeros((s.size, 1), dtype=design_mat.dtype)
        d[idx] = 1 / s_nnz
        d_UT_z = d * UTz
        coeffs = np.dot(Vt.T, d_UT_z)
        self.coeffs = coeffs

    def get_reshaped_coeffs(self):
        """
        Reshape the coeffs matrix to make it compatible with np.polynomial

        Returns
        -------
        c : ndarray (degree + 1, degree + 1, number of predictions)
            Coefficients for the polynomials compatible with np.polynomial.

        """
        c = np.zeros((self.d + 1, self.d + 1, self.coeffs.shape[1]))
        # loop on predicted quantities
        for pred_id in range(c.shape[2]):
            # loop on number of coeffs per polynomial
            for k in range(self.num_coeffs):
                c[self.pow_x[k], self.pow_y[k], pred_id] = self.coeffs[k, pred_id]
        return c

    def eval_poly(self, x, y, zindices=None, grid_eval=False):
        """Evaluate the prediction at the provided x, y coordinates.


        Parameters
        ----------
        x : ndarray (n, )
            x coordinate where we need to evaluate the polynomial.
        y : ndarray (n', )
            y coordinate where we need to evaluate the polynomial.
            if grid_eval=False, n = n'
        zindices : ndarray, optional
            Indices of the columns of z on which we would like a prediction
            if we performed a fit on a multidimensionnal z. If None,
            the prediction is performed on all the columns.
            The default is None.
        grid_eval : bool, optional
            If set to True, the polynomial is evaluated at the cartesian
            product of x and y. Otherwise, the polynomial is evaluated at
            the points defined by [x, y].

        Returns
        -------
        ndarray (nresult, len(zindices))
            The prediction.
            if grid_eval=False, nresult = n ( the dimension of x or y)
            if grid_eval=True, nresutl = n x n'

        """
        assert hasattr(self, 'coeffs'), 'not fitted yet'
        if zindices is None:
            zindices = np.arange(self.coeffs.shape[1])
        x = self._normalize(x, self.xoff, self.xscale)
        y = self._normalize(y, self.yoff, self.yscale)

        c = self.get_reshaped_coeffs()[:, :, zindices]

        if grid_eval:
            znormed = np.polynomial.polynomial.polygrid2d(x, y, c)
            # shape is now (num_pred, len(x), len(y))
            znormed = np.transpose(znormed, (2, 1, 0)).reshape((len(y) * len(x),
                                                                c.shape[2]))
        else:
            znormed = np.polynomial.polynomial.polyval2d(x, y, c).T

        return self._unnormalize(znormed, self.zoff, self.zscale, zindices)
