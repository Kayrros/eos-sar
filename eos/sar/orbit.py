"""Encapsulation of ephemerides position and speed interpolation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eos.sar import cheb


@dataclass(frozen=True)
class StateVector:
    time: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]

    def __getitem__(self, name: str) -> Any:
        warnings.warn(
            "Indexing a StateVector is deprecated (they no longer are dict).",
            DeprecationWarning,
        )
        return self.__dict__[name]

    def to_dict(self) -> dict[str, Any]:
        return dict(
            time=self.time,
            position=self.position,
            velocity=self.velocity,
        )

    @staticmethod
    def from_dict(dict: dict[str, Any]) -> StateVector:
        return StateVector(
            time=dict["time"],
            position=dict["position"],
            velocity=dict["velocity"],
        )


@dataclass
class Orbit:
    """Orbit object encapsulating the position variation with time,
    as well as the possibility to get the nth derivative (for speed
    and acceleration for ex)."""

    sv: list[StateVector]
    """ List of state vectors (time, position, velocity) """
    degree: int = 11
    """ Degree of the polynomial """
    coeffs: list[NDArray[np.float64]] = field(init=False)
    cheb_domain: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        # this is a temporary workaround, since the interface of Orbit.__init__ changed
        # from accepting a list[dict] to a list[StateVector]
        if self.sv and isinstance(self.sv[0], dict):
            self.sv = [StateVector.from_dict(s) for s in self.sv]  # type: ignore
            warnings.warn(
                "The Orbit constructor will no longer accept a list of dict for the sv parameter. "
                "Use StateVector.from_dict or Orbit.from_dict(dict(state_vectors=sv)) instead.",
                DeprecationWarning,
            )
        self.fit()

    def fit(self):
        """Fit the orbit representation on the samples."""
        self.coeffs = []
        coeffs, self.cheb_domain = cheb.build_cheb_interp(self.sv, self.degree)
        self.coeffs.append(coeffs)
        # Also store the speed/acc coefficients
        for i in range(2):
            self.coeffs.append(
                cheb.get_diff_coeffs(self.coeffs[-1], self.cheb_domain, der=1)
            )

    def evaluate(self, azt, order=0):
        """Evaluate the nth order derivative of the position of satellite
        along the orbit at time azt.

        Parameters
        ----------
        azt: 1darray (n, )
           Azimuth time on which to evaluate
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
            coeff = cheb.get_diff_coeffs(self.coeffs[0], self.cheb_domain, der=order)
        return cheb.evaluate_cheb_interp(azt, coeff, self.cheb_domain)

    def to_dict(self) -> dict[str, Any]:
        metadata = dict(
            state_vectors=[s.to_dict() for s in self.sv],
            degree=self.degree,
        )
        return metadata

    @staticmethod
    def from_dict(dict: dict[str, Any]) -> Orbit:
        sv = [StateVector.from_dict(s) for s in dict["state_vectors"]]
        degree = dict.get("degree", 11)
        return Orbit(sv=sv, degree=degree)
