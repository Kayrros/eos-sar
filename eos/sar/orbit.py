from eos.sar import cheb


class Orbit:
    """Orbit object encapsulating the position variation with time,
    as well as the possibility to get the nth derivative (for speed and acceleration for ex)
    """

    def __init__(self, state_vectors, degree=11):
        """
        Constructor
        Parameters
        ----------
        state_vectors: list of dict
                        List of state vectors (time, position, velocity)
        degree: int
                Degree of the polynomial
        """
        self.sv = state_vectors
        self.degree = degree
        self.fit()

    def fit(self):
        """
        Fit the orbit representation on the samples
        """
        self.coeffs = []
        coeffs, self.cheb_domain = cheb.build_cheb_interp(self.sv, self.degree)
        self.coeffs.append(coeffs)
        # Also store the speed/acc coefficients
        for i in range(2):
            self.coeffs.append(cheb.get_diff_coeffs(
                self.coeffs[-1], self.cheb_domain, der=1))

    def evaluate(self, t, order=0):
        """Evaluate the nth order derivative of the position of satellite
            along the orbit at time t
        Parameters
        ----------
        t: 1darray (n, )
           Time on which to evaluate
        order: int
            Order of the derivative, default is 0
            for order = 0, the position of the satellite is returned
        Returns:
        -------
        (n, 3) numpy.ndarray
            Position of satellite for each azimuth time provided
        """
        assert order >= 0, "order must be greater or equal to zero"
        if order < 3:
            coeff = self.coeffs[order]
        else:
            coeff = cheb.get_diff_coeffs(
                self.coeffs[0], self.cheb_domain, der=order)
        return cheb.evaluate_cheb_interp(t, coeff, self.cheb_domain)
