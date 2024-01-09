import numpy as np
import setuptools
from Cython.Build import cythonize
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

ext_module = cythonize(
    ("eos/products/sentinel1/_calibration.pyx", "eos/sar/simulator.pyx")
)
for m in ext_module:
    m.include_dirs = [np.get_include()]

setuptools.setup(
    name="kayrros-eos-sar",
    version="0.24.0",
    description="",
    long_description=long_description,
    author="Kayrros",
    url="https://git.dev-kayrros.ovh/products/satellite-tools/rs-tlbx/eos-sar/",
    packages=find_packages(exclude=["tests"]),
    package_dir={"": "."},
    package_data={"eos": ["py.typed"], "teosar": ["py.typed"]},
    ext_modules=ext_module,
    setup_requires=["cython", "numpy"],
    install_requires=required,
    include_package_data=True,
    entry_points="",
    extras_require={
        "test": ["pytest", "s1m"],
        # dependencies that teosar requires
        "teosar": ["tqdm", "tifffile", "tensorflow-cpu", "tensorflow_probability"],
        # dependencies for usage at Kayrros
        "kayrros": ["kayrros-phoenix[source-s3,plugin-burster]", "kayrros-bursterio"],
    },
    classifiers=[],
)
