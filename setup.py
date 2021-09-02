import setuptools
from setuptools import find_packages

from Cython.Build import cythonize
import numpy as np

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='kayrros-eos-sar',
    version='0.4.0',
    description='',
    long_description=long_description,
    author='Kayrros',
    url='https://git.dev-kayrros.ovh/products/satellite-tools/rs-tlbx/eos-sar/',
    packages=find_packages(),
    package_dir={'': '.'},
    package_data={},

    ext_modules=cythonize('eos/products/sentinel1/_calibration.pyx'),
    include_dirs=[np.get_include()],

    install_requires=required,
    include_package_data=True,
    entry_points='',
    extras_require={
        'test': ['pytest', 's1m']
    },
    classifiers=[]
)
