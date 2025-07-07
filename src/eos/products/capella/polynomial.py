from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.polynomial.chebyshev as T
import numpy.polynomial.legendre as L
import numpy.polynomial.polynomial as P
from numpy.typing import NDArray

from eos.products.capella.metadata import CapellaPolynomialMeta


@dataclass(frozen=True)
class CapellaPolynomial1D:
    poly_type: Literal["standard", "chebyshev", "legendre"]
    coefficients: NDArray[np.float64]

    def __post_init__(self):
        assert self.poly_type in ["standard", "chebyshev", "legendre"]
        # assert 1D array
        assert len(self.coefficients.shape) == 1

    @classmethod
    def from_poly_meta(cls, poly_meta: CapellaPolynomialMeta) -> CapellaPolynomial1D:
        coefs = np.array(poly_meta.coefficients)

        return CapellaPolynomial1D(poly_meta.poly_type, coefs)

    def evaluate(self, x):
        if self.poly_type == "standard":
            return P.polyval(x, self.coefficients)
        elif self.poly_type == "chebyshev":
            return T.chebval(x, self.coefficients)
        elif self.poly_type == "legendre":
            return L.legval(x, self.coefficients)


@dataclass(frozen=True)
class CapellaPolynomial2D:
    poly_type: Literal["standard", "chebyshev", "legendre"]
    coefficients: NDArray[np.float64]

    def __post_init__(self):
        assert self.poly_type in ["standard", "chebyshev", "legendre"]
        # assert 2D array
        assert len(self.coefficients.shape) == 2

    @classmethod
    def from_poly_meta(cls, poly_meta: CapellaPolynomialMeta) -> CapellaPolynomial2D:
        coefs = np.array(poly_meta.coefficients)
        return CapellaPolynomial2D(poly_meta.poly_type, coefs)

    def evaluate(self, x, y):
        if self.poly_type == "standard":
            return P.polyval2d(x, y, self.coefficients)
        elif self.poly_type == "chebyshev":
            return T.chebval2d(x, y, self.coefficients)
        elif self.poly_type == "legendre":
            return L.legval2d(x, y, self.coefficients)

    def evaluate_grid(self, x, y):
        if self.poly_type == "standard":
            return P.polygrid2d(x, y, self.coefficients)
        elif self.poly_type == "chebyshev":
            return T.chebgrid2d(x, y, self.coefficients)
        elif self.poly_type == "legendre":
            return L.leggrid2d(x, y, self.coefficients)
