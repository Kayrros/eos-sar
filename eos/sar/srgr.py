import abc

import numpy as np
from numpy.typing import ArrayLike, NDArray

Arrayf64 = NDArray[np.float64]


class SRGRConverter(abc.ABC):
    @abc.abstractmethod
    def gr_to_rng(self, gr: ArrayLike, azt: ArrayLike) -> Arrayf64: ...

    @abc.abstractmethod
    def rng_to_gr(self, rng: ArrayLike, azt: ArrayLike) -> Arrayf64: ...
