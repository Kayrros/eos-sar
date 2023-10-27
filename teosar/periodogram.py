from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from scipy.optimize import minimize
import tqdm
from teosar import psutils

@dataclass(frozen=True)
class Grid:
    values: NDArray
    parent_model: Model

    def get_values(self) -> NDArray:
        return self.values

    def get_last_dim_size(self):
        return self.values.shape[-1]

class BaseModel(ABC):

    def get_dimension(self):
        return len(self.get_theta_list())

    @abstractmethod
    def get_theta_list(self):
        pass

    def get_theta_from_indices(self, indices: list[int]):
        assert len(indices) == self.get_dimension(), "Should provide one index per element in theta_list"
        return [theta[idx] for idx, theta in zip(indices, self.get_theta_list())]

@dataclass
class Model(BaseModel):
    """Model class

    Parameters
    ----------
    const:
        Nconst
    feat:
        Nfeat x Nobs
    theta_list:
        d elems size Nj j=1...d

    """
    const: list[float]
    feat: NDArray[np.float64]
    theta_list: list[list[float]]

    def predict_grid(self):
        '''
        Predicts the model for a meshgrid defined by each value taken in theta_list
        and each value of the feature observation.

        Returns
        -------
        Grid.

        '''
        raise NotImplementedError

    def predict_single(self, theta_single : list[float]):
        '''
        Predicts the model for a single value per theta

        Parameters
        ----------
        theta_single: list[float]
                    d float elements

        Returns
        -------
        prediction: (Nobs,)
        '''
        raise NotImplementedError

    def get_theta_list(self):
        return self.theta_list


class SeasonalModel(Model):
    '''
    Implements C.S.\sin(\frac{2\pi}{T}(T_i - T_0)),
    where T_i is the feature i=1...Nobservations,
    Cv is a known constant (for conversion to phase for example),
    S, T, T_0 are the unkown parameters of the sinusoidal signal
    '''
    def __init__(self, C: float, Ti: NDArray[np.float64], S, T, T0):
        super().__init__([C], Ti[None, :], [S, T, T0])

    def predict_grid(self):
        S, T, T0 = self.theta_list
        Ti = self.feat[0]
        res = - 2 * np.pi * np.subtract.outer(T0, Ti) # len(T0) x Nobs
        res = np.sin(np.multiply.outer(1/T, res)) # len(T) x len(T0) x Nobs
        res = self.const[0] * np.multiply.outer(S, res) # len(S) x len(T) * len(T0) x Nobs
        grid = Grid(res, self)
        return grid

    def predict_single(self, theta_single : list[float]):
        assert len(theta_single) == 3, "Expected 3 values as theta"
        S, T, T0 = theta_single
        Ti = self.feat[0]
        return np.sin(2 * np.pi * (Ti - T0) / T) * S * self.const[0]

class LinearTermModel(Model):
    '''
    1 constant, 1 feature, 1 unkown
    Implement const * feat * theta
    '''
    def __init__(self, const:float,
                 feat: NDArray[np.float64], # Nobs
                 theta: list[float]
                 ):
        super().__init__([const], feat[None, :], [theta])

    def predict_grid(self):
        # len(theta) x Nobs
        res = self.const[0] * np.multiply.outer(self.theta_list[0], self.feat[0])
        grid = Grid(res, self)
        return grid

    def predict_single(self, theta_single : list[float]):
        assert len(theta_single) == 1, "Expected single element in list"

        return self.const[0] * theta_single[0] * self.feat[0]

def expand_dims(array, pos, d, total_dims):
    dims_to_expand = [a for a in range(pos)] + [a for a in range(pos + d, total_dims)]
    return np.expand_dims(array, tuple(dims_to_expand))

class CompoundModel(BaseModel):
    def __init__(self, models: list[Model]):
        self.models = models
        self.dims = [mod.get_dimension() for mod in self.models]
        self.begin_positions = np.cumsum([0] + self.dims[:-1])
        self.d = np.sum(self.dims)

    def predict_grid(self):
        pred = 0
        for mod, pos, d in zip(self.models, self.begin_positions, self.dims):
            pred = pred + expand_dims(mod.predict_grid().get_values(), pos, d, self.d)
        return Grid(pred, self)

    def predict_single(self, theta_single: list[float]):
        assert len(theta_single) == self.d, f"Expected {self.d} elements in theta_single"
        pred = 0
        for mod, pos, d in zip(self.models, self.begin_positions, self.dims):
            pred = pred + mod.predict_single(theta_single[pos:pos + d])
        return pred

    # override base method
    def get_dimension(self):
        return self.d

    def get_theta_list(self):
        theta_list = []
        for mod in self.models:
            theta_list += mod.theta_list
        return theta_list

@dataclass
class ExhaustiveGamma:
    abs_gamma: NDArray[np.float32]
    max_indices: tuple[int]
    max_theta: list[float]
    max_gamma_abs: float
    max_gamma_angle: float

    def get_bounds(self, theta_list: list[list[float]]) -> list[tuple[float]]:
        # bounds
        bounds = []
        assert len(self.max_indices) == len(theta_list), "theta_list must have same size as max_indices"
        for m, theta in zip(self.max_indices, theta_list):
            l= theta[m - 1] if m > 0 else None
            if m > 0:
                l = theta[m - 1]
            else:
                l = None
            if m < len(theta) - 1:
                u = theta[m + 1]
            else:
                u = None

            bounds.append((l, u))

        return bounds

@dataclass
class Periodogram:
    phi_wrapped: list[float] # N obs
    weights: Optional[list[float]]=None # N obs

    def __init__(self,  phi_wrapped: list[float],
                 weights: Optional[list[float]]=None):

        nan_mask = np.isnan(phi_wrapped)
        assert not np.all(nan_mask), "All values are nan, can't do periodogram"

        # might contain nans
        # replace nan with arbitrary value, weight=à will eliminate this from sum
        self.phi_wrapped = np.array(np.nan_to_num(phi_wrapped))

        if weights is not None:
            weights = np.array(weights)
        else:
            weights = np.ones((len(self.phi_wrapped), ), )

        # put nan value weights to zero
        weights[nan_mask] = 0
        # normalize weights
        weights /= np.sum(weights)
        self.weights = weights

    def exhaustive_gamma(self, grid: Grid) -> ExhaustiveGamma:
        assert grid.get_last_dim_size() == len(self.phi_wrapped), "Grid last dimension size should match phase array size"

        tmp = np.exp( 1j * ( - grid.get_values() + self.phi_wrapped))

        cmpx_gamma = np.sum(tmp * self.weights, axis=-1)

        del tmp
        abs_gamma = np.abs(cmpx_gamma)
        max_indices = np.unravel_index(np.argmax(abs_gamma), abs_gamma.shape)
        max_gamma_angle = np.angle(cmpx_gamma[max_indices])
        del cmpx_gamma
        max_theta = grid.parent_model.get_theta_from_indices(max_indices)
        max_gamma_abs = abs_gamma[max_indices]

        return ExhaustiveGamma(abs_gamma, max_indices, max_theta,
                               max_gamma_abs, max_gamma_angle)

    def get_gamma_single(self, model: Model, theta_single: list[float]):
        pred_per_obs = model.predict_single(theta_single)
        tmp = np.exp(1j * (self.phi_wrapped - pred_per_obs))
        gamma = np.sum(tmp * self.weights)
        return gamma


    def refinement(self, model: Model, exhaustive_gamma: ExhaustiveGamma, no_failure=False):
        estimated = exhaustive_gamma.max_theta
        bounds = exhaustive_gamma.get_bounds(model.get_theta_list())

        # refinement
        def to_minimize(x):
            return -np.abs(self.get_gamma_single(model, x))

        res = minimize(to_minimize, estimated, method='L-BFGS-B', options={'disp': False},
                       bounds=bounds)
        if res.success or no_failure:
            return res.x, - res.fun
        else:
            raise ArithmeticError("Did not converge")

def get_test_vals(max_val, half_n_samples):
        test_vals = np.linspace(0, max_val, half_n_samples)
        return np.hstack([-test_vals[:0:-1], test_vals])

# here specialize some cases for ease of use
def get_planar_model(xx, yy, max_slopes=(5e-2, 0.2), min_half_samples=10):
    """ for fitting spatially affine phase"""
    n_ps = len(xx)
    def get_slope(max_slope):
        half_n_samples = max(min_half_samples, int(np.sqrt(n_ps)/4)) # 25 for 10 000
        return get_test_vals(max_slope, half_n_samples)
    slope_x = get_slope(max_slopes[0])
    slope_y = get_slope(max_slopes[1])

    model = CompoundModel(
        [LinearTermModel(1, yy, slope_y),
         LinearTermModel(1, xx, slope_x)])

    return model

@dataclass(frozen=True)
class PlanarPeriodogramResult:
    exhaustive_gamma: ExhaustiveGamma
    slopes: list[float, float]
    bias: float
    gamma_opt: float

def planar_periodogram(model: CompoundModel, phi_wrapped_ts: list[list[float]],
                       no_failure=False) -> list[PlanarPeriodogramResult]:

    grid = model.predict_grid()
    results = []
    for i in tqdm.trange(len(phi_wrapped_ts)):
        period = Periodogram(phi_wrapped_ts[i])
        exhaustive_gamma = period.exhaustive_gamma(grid)
        slopes, gamma_opt = period.refinement(model, exhaustive_gamma, no_failure=no_failure)
        cmpx_gamma_opt = period.get_gamma_single(model, slopes)
        bias = np.angle(cmpx_gamma_opt)
        diff = abs(gamma_opt - np.abs(cmpx_gamma_opt))
        threshold = 0.01
        assert diff  < threshold, f"something wrong with gamma opt, {diff} greater than {threshold} "
        results.append(PlanarPeriodogramResult(exhaustive_gamma, list(slopes), bias,
                                               gamma_opt))

    return results

def compensate_planar_phase(phi_ts, xx, yy, slopes, biases):

    model = CompoundModel(
        [LinearTermModel(1, yy, [0]),
         LinearTermModel(1, xx, [0])])

    phi_compensated = np.zeros_like(phi_ts)
    for i in tqdm.trange(len(phi_ts)):
        prediction = model.predict_single(slopes[i]) + biases[i]
        phi_compensated[i] = psutils.wrap(phi_ts[i] - prediction)

    return phi_compensated
