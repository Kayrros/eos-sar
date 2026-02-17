# eos-sar

This package provides access to some generic SAR (Synthetic Aperture Radar) processing algorithms.

Currently supported sensors: 

- Sentinel-1 SLC in IW mode 

- For the stripmap acquisition mode, some support for: 
  
  - Cosmo-SkyMed
  
  - TerraSAR-X
  
  - Capella 

- NISAR (initial support, work in progress) 

## Requirements & Installation

To install the package in editable mode, you can run:

    uv sync

or

    pip install -e . --group dev

Some processing pipelines assume that a library providing acess to a DEM (Digital Elevation Model) source is provided. The recommended source is `eos.dem.DEMStitcherSource` (with `pip install dem-stitcher`) that uses GLO30 and GLO90. You can also install [srtm4](https://github.com/centreborelli/srtm4) as an SRTM source.

If you wish to install srtm4, make sure to install with the "crop" extra dependencies:

    pip install srtm4["crop"]

Both these libraries are included by default in the dev dependencies.

If you wish to use another DEM source, make sure to inherit from the template `eos.dem.DEMSource` and to provide functions for cropping/querying a DEM.

The package also has some extras, such as: 

- `teosar-light` to run the `teosar/tsinsar.py` module that creates a time series of coregistered Sentinel-1 crops with flattened phases, ready for interferometry. 

- `teosar` which as some support for Persistent Scatterer Interferometry, through `teosar/ferreti_2001.py` for instance. Note that this part of the codebase is more research oriented and not heavily tested. !! This extra has a dependency on pyopencl.[ PyOpenCL cannot run code without an OpenCL device driver](https://documen.tician.de/pyopencl/misc.html#enabling-access-to-cpus-and-gpus-via-py-opencl) (for CPU or GPU). If you want this extra (teosar) to properly run, ensure you have installed an OpenCL driver.

## Usage

Check the usage folder for a tutorial. The tutorial corresponds to performing a Sentinel-1 interferogram (among other things) on data spanning an earthquake taking place at [January 7 2022: M 6.6 - 113 km SW of Jinchang, China](https://sarviews-hazards.alaska.edu/Event/e2dfcb22-e1a4-43d8-a17e-c6b175849463).

For the tutorial, you need a [CDSE](https://dataspace.copernicus.eu/) account. You also need to generate [AWS secrets](https://eodata-s3keysmanager.dataspace.copernicus.eu/). 
Then, set them as environment variables, by creating a .env file for instance: 

```
CDSE_ACCESS_KEY_ID = <value>
CDSE_SECRET_ACCESS_KEY = <value>
CDSE_USERNAME = <value>
CDSE_PASSWORD = <value>
```

Then, you can run the file `usage/tutorial.ipynb`.

```
uv run --env-file .env --with jupyter --with matplotlib jupyter lab
```

The main features shown in the tutorial are listed below:

- Physical sensor model for projection and localization in a Sentinel-1 image.
- Reading/ Calibration of S-1 data.
- Registration/ resampling/ debursting of a secondary image onto a primary image. 
  The processing can be restricted to a region of interest.
- Line of sight computation
- Interferogram formation, orbital and topographic phase estimation and removal, coherence estimation.
- Orthorectification

## Contributing

### Tests

To run the tests, we use `pytest`:

    uv run --all-extras pytest -n auto -v -m "not cdse" .
    uv run --env-file .env --all-extras pytest -v -m "cdse" .

Ideally, you would put your CDSE credentials in the .env file (see section above), so that the tests that read data from CDSE can run. Otherwise, the tests will be skipped. Note that the tests reading from CDSE are run separately in the commands above, on a single worker, to avoid issues related to rate limiting. Also, those tests are marked as "flaky", i.e., they are retried if/when they fail (due to rate limiting).

### Code formatting

The CI validates the code against pep8 rules and formatting, as configured in `pyproject.toml`.

You can check your code locally before commiting using pre-commit or using:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Avoid making commits that only format the code; instead, amend commits or rebase the changes against the relevant commit.

You can also use the pre-commit. 

```
source .venv/bin/activate # the .venv needs to be activated
uvx pre-commit install # you can do this once
git add file.py 
git commit -m "message here" # pre-commit runs, might fail, no commit
# In case the pre-commit failed because of formatting
# --> retry
# In case the pre-commit failed because of typing (mypy)
# --> fix problems then retry
git add file.py
git commit -m "message here" # should work now
```

### Making a release

1. generate the changelog: `uv run --no-project --with git-cliff git cliff --unreleased` and update `CHANGELOG.md` manually
2. update the version in `pyproject.toml` (try to respect semantic versioning)
3. run `uv lock` to update uv.lock
4. commit (message="x.y.z") and tag the commit (tag="x.y.z")
5. push with the tag (`git push --tags`)

### Tips for external contributors

Make sure to have pyproj data: `pyproj sync -v --file us_nga_egm96_15`
