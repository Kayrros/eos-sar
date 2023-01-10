import setuptools
from setuptools import find_packages

from Cython.Build import cythonize
import numpy as np

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

ext_module = cythonize(('eos/products/sentinel1/_calibration.pyx', 'eos/sar/simulator.pyx'))
for m in ext_module:
    m.include_dirs = [np.get_include()]

setuptools.setup(
    name='kayrros-eos-sar',
    version='0.13.1',
    description='',
    long_description=long_description,
    author='Kayrros',
    url='https://git.dev-kayrros.ovh/products/satellite-tools/rs-tlbx/eos-sar/',
    packages=find_packages(exclude=['tests']),
    package_dir={'': '.'},
    package_data={},
    ext_modules=ext_module,
    setup_requires=["cython", "numpy"],
    install_requires=required,
    include_package_data=True,
    entry_points='',
    extras_require={
        'test': ['pytest', 's1m']
    },
    classifiers=[]
)
