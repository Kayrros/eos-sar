# eos-sar

This package provides access to some generic SAR (Synthetic Aperture Radar) processing algorithms. 

Currently, algorithms specific to **Sentinel1 SLC** in **IW** mode have been implemented. 

### Requirements & Installation
To install the package, first install `cython` and `numpy`. You can run: 
	
	pip install cython numpy
	pip install -e .

Some processing pipelines assume that a library providing acess to a DEM (Digital Elevation Model) source is provided. Therefore, make sure that you either install `multidem` (Kayrros package) or [srtm4](https://github.com/centreborelli/srtm4), which is an open source alternative providing acess to SRTM90 data.  If you wish to use another dem source, make sure to inherit from the template `eos.dem.DEMSource` and to provide functions for cropping/querying a dem.

Automatic download of Sentinel-1 SLC data is not provided in this package. You can use [ASF](https://search.asf.alaska.edu/#/) to find and download the data.

As for the precise or restituted orbit files, automatic querying and download for a product is not provided in this package. Another Kayrros package for handling orbit files may be made public. 

### Usage

Check the usage folder for a tutorial. The tutorial corresponds to performing an interferogram (among other things) on data spanning an earthquake taking place at [January 7 2022: M 6.6 - 113 km SW of Jinchang, China](https://sarviews-hazards.alaska.edu/Event/e2dfcb22-e1a4-43d8-a17e-c6b175849463).

Before running the tutorial, the necessary data must first be downloaded, so you can simply run this in a shell: 
	
	cd usage
	mkdir tutorial
	cd tutorial
	python ../download_pair.py

A *data* folder will be created containing the safes and the associated orbit files. 

Then, you can check the file `tutorial.ipynb` or `tutorial.py`. The file is divided in code cells (similar to Matlab).

The features shown in the tutorial are listed below: 

- Physical sensor model for projection and localization in a Sentinel-1 image. 
- Reading/ Calibration of S-1 data.
- Registration/ resampling/ debursting of a secondary image onto a primary image. The processing can be restricted to a region of interest.
- Interferogram formation, orbital and topographic phase estimation and removal, coherence estimation.

### Tests 

Some tests currently use Kayrros cloud storage, which means that certain credentials must be set up for these tests.  Currently, only local tests will run and others will fail if you don't have these credentials. To run the tests locally, we use `pytest`: 

	pip install pytest 
	pytest .
