import abc

import numpy as np
from numpy.typing import ArrayLike, NDArray

Arrayf32 = NDArray[np.float32]


class SRGRConverter(abc.ABC):
    @abc.abstractmethod
    def gr_to_rng(self, gr: ArrayLike, azt: ArrayLike) -> Arrayf32:
        ...

    @abc.abstractmethod
    def rng_to_gr(self, gr: ArrayLike, azt: ArrayLike) -> Arrayf32:
        ...
