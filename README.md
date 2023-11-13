# eos-sar

This package provides access to some generic SAR (Synthetic Aperture Radar) processing algorithms.

Currently, algorithms specific to **Sentinel1 SLC** in **IW** mode have been implemented.

### Requirements & Installation
To install the package in editable mode, you can run:

	pip install -e .

Some processing pipelines assume that a library providing acess to a DEM (Digital Elevation Model) source is provided. The recommended source is `eos.dem.DEMStitcherSource` (with `pip install dem-stitcher`) that uses GLO30 and GLO90. You can also install `multidem` (Kayrros package) or [srtm4](https://github.com/centreborelli/srtm4) as SRTM sources.

If you wish to install srtm4, make sure to install with the "crop" extra dependencies: 

	pip install srtm4["crop"]

 If you wish to use another dem source, make sure to inherit from the template `eos.dem.DEMSource` and to provide functions for cropping/querying a dem.

Automatic download of Sentinel-1 SLC data is not provided in this package yet. You can use [ASF](https://search.asf.alaska.edu/#/) to find and download the data.

As for the precise or restituted orbit files, automatic querying and download for a product is not provided in this package yet.

### Usage

Check the usage folder for a tutorial. The tutorial corresponds to performing an interferogram (among other things) on data spanning an earthquake taking place at [January 7 2022: M 6.6 - 113 km SW of Jinchang, China](https://sarviews-hazards.alaska.edu/Event/e2dfcb22-e1a4-43d8-a17e-c6b175849463).

Before running the tutorial, the necessary data must first be downloaded. There are also additional dependencies. So you can simply run this in a shell:

	pip install jupyter matplotlib # install dependencies
	cd eos-sar
	python tools/download_from_asf.py https://s1qc.asf.alaska.edu/aux_resorb/S1A_OPER_AUX_RESORB_OPOD_20211230T024411_V20211229T224022_20211230T015752.EOF usage/tutorial/data/orb
	python tools/download_from_asf.py https://s1qc.asf.alaska.edu/aux_resorb/S1A_OPER_AUX_RESORB_OPOD_20220111T024731_V20220110T224022_20220111T015752.EOF usage/tutorial/data/orb
	python tools/download_from_asf.py https://datapool.asf.alaska.edu/SLC/SA/S1A_IW_SLC__1SDV_20211229T231926_20211229T231953_041230_04E66A_3DBE.zip usage/tutorial/data/safes --unzip
	python tools/download_from_asf.py https://datapool.asf.alaska.edu/SLC/SA/S1A_IW_SLC__1SDV_20220110T231926_20220110T231953_041405_04EC57_103E.zip usage/tutorial/data/safes --unzip
	jupyter notebook # launch notebook

   *eos-sar/usage/tutorial/data* folder will be created containing the safes and the associated orbit files. The two products will be downloaded and unzipped in the directory. The two corresponding orbits will also be downloaded.


Then, you can check the file `usage/tutorial.ipynb`.

The features shown in the tutorial are listed below:

- Physical sensor model for projection and localization in a Sentinel-1 image.
- Reading/ Calibration of S-1 data.
- Registration/ resampling/ debursting of a secondary image onto a primary image. The processing can be restricted to a region of interest.
- Interferogram formation, orbital and topographic phase estimation and removal, coherence estimation.

### Tests

Some tests currently use Kayrros cloud storage, which means that certain credentials must be set up for these tests.  Currently, only local tests will run and others will fail if you don't have these credentials. To run the tests locally, we use `pytest`:

	pip install pytest
	pytest .

For the future, all tests should be able to run locally, but test data should be downloaded beforehand. We plan to use data from ASF, similarly to the tutorial data, that can be downloaded by running in a shell:

	cd eos-sar
	python tools/download_from_asf.py https://s1qc.asf.alaska.edu/aux_poeorb/S1B_OPER_AUX_POEORB_OPOD_20210330T214422_V20190701T225942_20190703T005942.EOF tests/data/pair/orb
	python tools/download_from_asf.py https://s1qc.asf.alaska.edu/aux_poeorb/S1A_OPER_AUX_POEORB_OPOD_20210330T235242_V20190707T225942_20190709T005942.EOF tests/data/pair/orb
	python tools/download_from_asf.py https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190702T032447_20190702T032514_016949_01FE47_69C5.zip tests/data/pair/safes --unzip
	python tools/download_from_asf.py https://datapool.asf.alaska.edu/SLC/SA/S1A_IW_SLC__1SDV_20190708T032532_20190708T032559_028020_032A14_33CA.zip tests/data/pair/safes --unzip


### Code formatting

The CI validates the code against pep8 rules and formatting, as configured in `pyproject.toml`.

You can check your code locally before commiting using:
```bash
pip install ruff==0.1.5
ruff check . --fix; ruff format .
```

Avoid making commits that only format the code; instead, amend commits or rebase the changes against the relevant commit.
