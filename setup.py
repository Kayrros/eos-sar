import numpy as np
import setuptools
from Cython.Build import cythonize

ext_module = cythonize(
    ("eos/products/sentinel1/_calibration.pyx", "eos/sar/simulator.pyx")
)
for m in ext_module:
    m.include_dirs = [np.get_include()]

setuptools.setup(ext_modules=ext_module)
