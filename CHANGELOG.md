# [0.42.0](https://github.com/Kayrros/eos-sar/compare/0.41.0..0.42.0) -- 2025-12-16

## Added
- nisar: metadata: Add NISAR RSLC metadata
- test: Add tests for NISAR metadata
- test: Add tests for NISAR projection model
- nisar: proj_model: Add NISAR projection model
- io: Add open_netcdf_osio

- tests: Add tests for Capella deramping/cropping/InSAR
- usage/capella: Add capella stripmap insar usage
- usage:capella: orthorectification / bperp / custom DEM:
    * Compute orthorectified interferograms. Calculate perpendicular baseline. Show how to use high resolution DEM stored locally.
- capella/slc_cropper: Add workflow for cropping/aligning a stack of SLC capella imgs
- capella/slc_cropper: Add translation information in the crop metadata
- capella: Add sigma nought calibration
- resampler: Add CapellaResample obj for deramping and reramping
- capella/doppler_info: Add CapellaDoppler class for freq dop centroid computation
- capella/metadata: Add state vectors origin metadata
- capella/metadata: Add check for ZeroDoppler geometry
- capella/metadata: Add parsing of Doppler Centroid frequency metadata

- ortho: add `on_multilooked_raster`

## Changed
- teosar:inout: Enhance save_img to support CRS / transform

## Fixed
- geoconfig: Fix Bperp for left looking SAR


# [0.41.0](https://github.com/Kayrros/eos-sar/compare/0.40.0..0.41.0) -- 2025-08-06

## Added
- grd cropper: `process` now returns a CropMetadata object, which allows to retrieve the LOS angles of the center of the crop
- tests: Add tests for validity of localize_without_alt
- test_dem: Add some tests for dem interpolation

## Changed
- model: Change get_approx_geom and get_buffered_geom behavior:
    * This is a consequence of the change of behavior of localize without alt,
    * which now is more prone to returning invalid results. All functions
    * using it must check the validity mask and decide to raise an exception
    * or not on a case by case basis.
- dem: Move some interpolation code into DEM.interpolate_array
- localize_without_alt and DEM.elevation: take into account points falling outside of the DEM extent thus avoiding costly DEM padding

## Fixed
- dem: Fix interpolation near right and lower border


# [0.40.0](https://github.com/Kayrros/eos-sar/compare/0.39.0..0.40.0) -- 2025-06-22

## Added
- usage:teosar: Add los computation example for teosar results
- tests: Add los test after stitching
- tutorial: Add los computation
    breaking change in Sentinel1AssemblyCropper.crop which now returns
    the array and the resampler as well
- feat: Add conversion from ECEF to ENU
    breaking change in eos.sar.geoconfig.get_los_on_ellipsoid
    which now returns los, points_3D
- sentinel1:los: Add functions to compute squinted/ZeroDopplerLos on mosaic
    change in behaviour of eos.sar.utils.stitch_array which now can stich
    arrays with arbitrary number of channels as long as their height and width
    respects the writing rois.
- geoconfig: Add function for squinted LOS
- range_doppler: Add support for non zero Doppler in projection
    change in eos.sar.range_doppler in iterative_projection and get_E_dE
    which now can take as input (optionally) the doppler centroid
    scaled by half the wavelength. This is necessary to estimate
    the time along the orbit that corresponds to a non zero Doppler frequency
- chore: add compiled .c to gitignore
- geoconfig: Add code to predict los with tests
- tests: Add test for geoconfig

## Changed
- ortho/tuto: comments about `resolution` units w.r.t the crs:
    * [skip ci]
- deps: lower the version requirement of ortools
- geoconfig: Compute geomconfig from grid coords

## Fixed
- test: Relax constraint on unstable test
- fix: Fix pre-commit for src layout


# [0.39.0](https://github.com/Kayrros/eos-sar/compare/0.38.0..0.39.0) -- 2025-05-15

## Added
- Capella: add metadata parsing for SLC and GRD
- Capella: add geometric model for SLC

## Changed
- repo: move to src-layout packaging
- dependencies: exclude scipy versions with slow linprog
- io: add rasterio_session_kwargs as input to open_image


# [0.38.0](https://github.com/Kayrros/eos-sar/compare/0.37.0..0.38.0) -- 2025-05-07

## Added
- readme: add info about releasing a new version
- repo: add changelog
- ci: add stage to test lower dependencies
- sentinel1/catalog: Add COG GRD support
- usage: Add usage example for timeseries
- tsinsar: add ProductProviderBase
- tsinsar: Add possibility to switch from phx to cdse
- setup.py: Add lower bound on tfp so that extra makes sense

## Changed

- pyproject: enable python 3.13
- readme: tweaks for uv
- ci: use uv
- repo: bump ruff and reformat
- repo: improve dev experience using uv
- sentinel1/catalog: Update the catalog odata request
- Update metadata.py (#169)

## Fixed
- tsinsar: Fix orbit type annotation

## Removed
- readme: remove old comments
- workflow: Remove usage of deprecated tifffile.imsave


# [0.37.0](https://github.com/Kayrros/eos-sar/compare/0.35.0..0.37.0) -- 2025-04-25

## Added
- metadata: Add relative orbit number formula for S1C
- s1/grd_cropper: add compression
- s1/product: add ProductInfo.get_properties()

## Changed



# [0.35.0](https://github.com/Kayrros/eos-sar/compare/0.34.0..0.35.0) -- 2025-03-19

## Added
- dem-stitcher: add parameter `tiles_cache_dir` and set gdal EMPTY_DIR:
    * GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR reduces the number of requests
    * performed when fetching tiles
- dem: add tests for MyDEMSource margin parameter
- tests/dem: add test for MyDEMSource
- dem: add MyDEMSource and DEM.get_extent
- dem: add fill_nan and parameters to SRTM4Source and DEMStitcherSource
- gitignore: add DS_store

## Changed

- teosar: don't require tensorflow and pyopencl (-light variant)
- Modify the workflow: disconnect secondary pipelines from primary (#157):
    * Co-authored-by: Roland Akiki <r.akiki@kayrros.com>
- dem_to_radar: Mask out nan values in radarcoding
- dem: check that MyDEMSource crs is 4326
- model/localize_without_alt: compute alt_min/alt_max from dem when none

## Fixed
- s1/grd-cropper: fix out-of-product cropping


# [0.34.0](https://github.com/Kayrros/eos-sar/compare/0.33.0..0.34.0) -- 2025-01-16

## Changed

- s1 model: set azt_init as the center of the model
- s1 proj_model: use the model helper
- model: use typevar to solve typing issues
- correction: work only with ndarrays
- corrections: make instances immutable

## Fixed
- model: fix type to f64 and fix projection/localization return type

## Removed
- s1 coord correction: remove __dict__ interface


# [0.33.0](https://github.com/Kayrros/eos-sar/compare/0.32.0..0.33.0) -- 2025-01-02

## Changed

- ortho: convert to dataclass and allow user instanciation
- reqs: bump ruff
- reqs: bump mypy

## Removed
- ortho: remove Orthorectifier.apply_stack


# [0.32.0](https://github.com/Kayrros/eos-sar/compare/0.31.0..0.32.0) -- 2024-12-19

## Added
- ortho: add explanations and typing

## Changed

- ortho/from_roi: use get_buffered_geom instead of get_approx_geom
- grd_cropper: better estimation of the Roi:
    * and reduce area of the fetched dem
- ortho: improve DEM subsetting:
    * Using get_approx_geom was prone to issues, in particular for DEM
    * containing nans but not only. In eos.sar.ortho we know the destination
    * AOI, we don't have to rely on sensor geometry to subset the DEM.
- s1/grd_cropper: improve invalid alignment exception message

## Fixed
- ortho: fix typos and unnecessary call
- tests/grd_cropper: assert that the rasters do not contain nans
- s1/grd_cropper: recompute the bbox on BboxDestinationGeometry:
    * This fixes cases with some nodata on the requested bbox

## Removed
- grd_cropper: remove unused return value


# [0.31.0](https://github.com/Kayrros/eos-sar/compare/0.30.1..0.31.0) -- 2024-12-04

## Added
- grd_cropper: support product assembly and CDSE products
- tests: add `cdse_s3_session` fixture
- ferreti_2001: Added CL+TF periodo for atmospheric estimation
- usage: Add small periodogram example
- periodogram: Add class that performs CL + TF periodogram
- teosar/ferreti: Add helper to call periodogram_cl
- teosar/periodogram: Add alternative OpenCL periodogram code:
    * The code enables to create your own periodogram functions,
    * and test in parallel on GPU or CPU for a lot of PS many
    * input values in order to maximize the periodogram.
    * 
    * This implementation currently only supports the simple
    * periodogram with 2 variables to optimize, but the core
    * of the code supports more, and it should be simple to
    * add support for more configurations.

## Changed

- teosar:workflow: Catch BursterException
- ferreti_2001: Run a loop if ncpu=1
- teosar:workflow: Catch Secondary Exceptions:
    * We expect the secondary pipeline to run on many products, thus, instead of interrupting the program if a problem occurs for a product, we catch the exception  in the execute function and issue a Warning. This choice is debatable and lies on the assumption that a few products might raise exceptions, but we might want the result on the other products anyway.
- periodgram_par:tf: Suppress some warnings
- ferreti_2001: Simplify import of PeriodogramPar/CL
- teosar: Expose Periodogram in init
- teosar: Simplify tf and add siphash
- ferreti_2001: Rename func gamma_inference
- periodogram_cl: Rename class PeriodogramCL
- teosar/ferreti: Overhaul of the tensorflow path:
    * Use a first pass with exhaustive search in order
    * to improve the quality of the results. Use the
    * new opencl path for speed.
    * 
    * Rewrite the tensorflow code to treat each element
    * individually rather than in group.
    * 
    * This accelerates compilation, computation and
    * limits resource usage.
- ci: install setuptools to package for 3.12

## Fixed
- s1 product: fix extract_ipf for GRD COG products:
    * Indeed it was picking up the version of the "GRD COGififier" and not the S1 processor,
    * causing problem in the calibration noise removal.
- snap: Fix reader when 2 products are read
- fix:periodogram.cl: Fix bug that was skipping some variable tuples
- periodogram.cl: Fix typo in docstring
- teosar: Fix dependencies
- teosar: Fix weights convention
- periodogram_cl: Fix codestyle

## Removed
- ferreti_2001: Remove pix that have nan phase
- usage:periodo_ex: Remove code cell


# [0.30.1](https://github.com/Kayrros/eos-sar/compare/0.30.0..0.30.1) -- 2024-09-24

## Changed

- reqs: pin numpy in -dev.txt to help with CI reproducibility
- ci: use PIP_INDEX_URL instead of PIP_EXTRA_INDEX_URL

## Fixed
- grd_cropper: clean-up and increase dem-fetch buffer:
    * This fixes an issue when cropping in high lattitudes
- s1 catalog/cdse: fix EvictionDate filtering:
    * see https://documentation.dataspace.copernicus.eu/APIs/Others/ReleaseNotes.html#datetime-precision-change-for-odata-opensearch-and-stac-apis-2024-09-02
- s1-catalog/cdse: fix EvictionDate expected value:
    * see https://documentation.dataspace.copernicus.eu/APIs/Others/ReleaseNotes.html#odata-catalogue-api-evictiondate-attribute-update-for-null-values-2024-07-31


# [0.30.0](https://github.com/Kayrros/eos-sar/compare/0.29.0..0.30.0) -- 2024-06-26

## Added
- tsx: Add first working tsx usage
- tsx: Add cropper module
- tsx: Add to_dict/from_dict to TSXMetadata

## Changed

- reqs: bump dev dependencies
- RoiProvider: Move RoiProvider from teosar to eos.sar.model
- tsx: Separate metadata and model modules
- tsx: Pass Orbit object to model constructor

## Fixed
- deps: fix compatibility with numpy 2


# [0.29.0](https://github.com/Kayrros/eos-sar/compare/0.28.0..0.29.0) -- 2024-04-23

## Added
- projection_correction: Add bias correction
- tsx: add products.terrasarx subpackage

## Changed

- tsx: read range/azimuth periods directly from xml:
    * and compute range/azimuth frequencies as class properties
- tsx: merge range_frequency and range_sampling_rate


# [0.28.0](https://github.com/Kayrros/eos-sar/compare/0.27.0..0.28.0) -- 2024-04-18

## Added
- io: add read_file_as_str and exists
- s1 assembler: add `get_metadata` for Sentinel1GRDAssembler
- ci: add some pip caching
- dev: add requirements-dev.txt with pinned versions

## Changed

- orbit_catalog: rename num_process_workers to num_parse_workers
- orbit_catalog: replace the process pool by a thread pool:
    * and slight adjustement to speedup orbit file parsing
- s1 metadata: speed-up function isostring_to_timestamp
- grd rtc slicing: initial version
- simulator: free the GIL a bit more
- orbit_catalog: minor comment on Sentinel1OrbitCatalogResult
- regist/apply_affine: expose the 'interpolation' parameter
- simulator: expose parameter extends_roi_n_grid
- readme: update testing and code style instructions
- ci: use python 3.10
- code: ruff formatting

## Fixed
- grd_cropper: fix boundless read and buggy orthorectification
- rtc: fix shadow_value type
- srgr: fix parameter names
- ci: fix install requirements.txt


# [0.27.0](https://github.com/Kayrros/eos-sar/compare/0.26.1..0.27.0) -- 2024-04-03

## Added
- calibration: add as_amplitude parameter to the reader
- calibration: support tiled processing for the reader
- roi: add split_into_tiles
- ferreti: Add parallelism for velo_topo_periodogram without tensorflow

## Changed

- io/read_window: avoid raster copies
- calibration: convert to amplitude unit from cython:
    * This reduces the memory usage significantly sometimes.
- ferreti: Speedup tensorflow compilation time
- ferreti: Optimize tensorflow in batches
- ferreti: Change interface to take years since ref
- sentinel1: don't log warnings if phoenix/bursterio are not installed

## Removed
- ferreti_2001: Remove unecessary comments


# [0.26.1](https://github.com/Kayrros/eos-sar/compare/0.26.0..0.26.1) -- 2024-03-12

## Changed
- 0.26.1:
    * fix py.typed file in sdist
- ci: make weels py 3.9, 3.10, 3.11, 3.12

## Fixed
- ci: fix py.typed
- dist: fix missing py.typed


# [0.26.0](https://github.com/Kayrros/eos-sar/compare/0.25.0..0.26.0) -- 2024-03-12

## Added
- repo: add .pre-commit-config.yaml
- requirements: add requests and typing_extensions

## Changed

- s1 grd_cropper: init
- model: introduce GenericSensorModelHelper (simplify cosmoskymed and snap):
    * sentinel1.proj_model is not modified for now, but could be simplified
    * similarly
- products.snap: initial support
- ci: don't try pip install in deploy:
    * this is already tested in the previous step

## Fixed
- tests/test_geom_phase: fix random test fail
- tests: mark cdse tests as flaky
- tests: don't fail if phoenix is not installed but credentials exist
- tests: fix when phoenix is not available
- teosar: fix polarization conversion

## Removed
- repo: remove setup.cfg:
    * it was used to configure flake8, which we no longer use


# [0.25.0](https://github.com/Kayrros/eos-sar/compare/0.24.0..0.25.0) -- 2024-02-13

## Added
- s1 product: support get_manifest for phoenix product
- usage: add csk.py
- teosar/utils: add Roi param to get_gcps_localization
- cosmo: add initial CosmoSkyMed metadata and model classes

## Changed

- regist: initial implementation of phase_correlation_on_amplitude
- cosmo: implement meta.deramping_phases(roi)


# [0.24.0](https://github.com/Kayrros/eos-sar/compare/0.23.0..0.24.0) -- 2024-01-09

## Added
- ortho: add apply_stack and improve typing
- license: add license agreement text:
    * This license agreement was released by Kayrros legal team on 2023-02-17

## Changed

- s1 cal: densify/extrapolate the range vectors and fix azimuth blocks
- ruff: bump to 0.1.11 and run format
- mypy: bump to 1.8.0 and fix errors:
    * Errors are from a new version of opencv, which now has typing
- tsinsar: use 'spawn' mp context for get_bsids_for_products
- orbit_catalog: use a 'spawn' mp context:
    * On POSIX, the default mp context is 'fork', which is incompatible with
    * threads
- orbit_catalog: reduce num_fetch_workers for phx backend
- s1 assembler: cache calibrators in _get_image_reader
- s1 acquisition: increase tolerance for old products
- teosar: skip secondary products missing some bursts

## Fixed
- s1 catalog: fix CDSE query
- usage: fix expected s3_path
- readme: fix a couple typos


# [0.23.0](https://github.com/Kayrros/eos-sar/compare/0.22.0..0.23.0) -- 2023-11-20

## Added
- s1 product: add extract_ipf and Sentinel1*ProductInfo.ipf
- s1 product: add Sentinel1*ProductInfo.get_manifest()
- s1 product: add CDSEUnzippedSafeSentinel1GRDProductInfo
- s1 catalog: add get_cdse_item for CDSE backends
- s1 catalog: add GRD backends
- packaging: add extra require "kayrros"
- teosar: add types to tsinsar.main
- packaging: add py.typed files:
    * see https://peps.python.org/pep-0561/#packaging-type-information

## Changed

- s1 calibration: use the IPF version instead of the date
- usage/grd: use the CDSE orbit catalog
- s1 orbits: implement CDSE backend
- s1 catalog: replace Sentinel1Catalog by a function
- ci: bump ruff and mypy
- ci: run mypy on teosar/
- mypy: allow variable redefinition with different type:
    * also called "shadowing", it's quite convenient

## Fixed
- io: fix read_xml_file output type
- tests: automatically skip tests if s3/cdse/phx is missing
- teosar/ferreti: fix error introduced in 3950a17
- teosar: fix mypy errors

## Removed
- tsinsar: remove missing products in remove_weird_products


# [0.22.0](https://github.com/Kayrros/eos-sar/compare/0.21.0..0.22.0) -- 2023-11-13

## Added
- teosar/ferreti: add a constant term per pixel
- orbit catalog: add backend LocalFilesSentinel1OrbitCatalogBackend
- tsinsar: add logging and caching
- orbit catalog: add statevectors caching
- cache: add basic eos.cache module

## Changed

- teosar/ferreti: fit a b-spline to estimate the atmospheric phase
- teosar/ferreti: return a Ferreti2001Result object
- teosar/ferreti: lazy import of tensorflow
- style: bump ruff to 0.1.4:
    * also, use == instead of ~= to fix reproducibility issues
- code: format using ruff
- codestyle: use ruff to replace flake8 and formatting
- tutorial: update dem_source and orbit catalog
- usage/grd: update orbit retrieval and fix nodata mask
- s1 catalog: propagate errors for cdse queries
- s1 catalog: cache queries if end_date < now()
- teosar: use the new orbit catalog
- s1 orbits: deprecate the module
- orbit catalog: multithreaded search for phx
- orbit catalog: rewrite access to aux files, simplify assembler
- ci: replace pytest-parallel with pytest-xdist


# [0.21.0](https://github.com/Kayrros/eos-sar/compare/0.20.0..0.21.0) -- 2023-10-30

## Changed

- code: import the teosar package


# [0.20.0](https://github.com/Kayrros/eos-sar/compare/0.19.0..0.20.0) -- 2023-10-23

## Changed

- s1 catalog: implement CDSE backend and add tests
- s1 product: introduce catalog abstraction


# [0.19.0](https://github.com/Kayrros/eos-sar/compare/0.18.0..0.19.0) -- 2023-10-23

## Added
- dem: add tests for OutOfBoundsException
- dem: Add OutOfBoundsException
- io: add ImageReader and ImageOpener protocols:
    * eos/sar/io.py and eos/products/sentinel1/product.py are now strictly
    * typed
- s1 correction: support full_bistatic_reference as dict (deprecated)
- s1 correction: add FullBistaticReference class
- s1 metadata: add temporary __getitem__ for retro compatibility
- usage: add access_to_cdse.py example
- io: add open_image_fsspec
- test: Add scipy unwrapping test
- feat: Add scipy linear program unwrapping
- ci: add mypy (non-strict for now)

## Changed

- dem/dem-stitcher: fill missing tiles with ellipsoid
- demregist: Make default margin=0 for regist dem points
- acquisition: less strict assert about the burst row
- model: replace approx_geom by approx_centroid_lon/lat:
    * this represents less information required to define a model, and is
    * better defined than an "approx geom" (now that we have 3 better-defined get_*_geom
    * in the sensor model)
- proj_model: simplify slc coordinate variables
- model: typing for projection/localization
- model: Coordinate is no longer a mixin but an attribute to models:
    * and add typing to the functions of SLC/GRDCoordinate
- dem: split into DEMSource and DEM classes, support dem_stitcher
- metadata/orbit: make metadata immutable:
    * this breaks the API on the orbit_provider user function, it must now
    * return new metadata (using .with_new_state_vectors)
- assembler: rework orbit_provider parameter
- s1 srgr: introduces Sentinel1GRDSRGRMetadata
- grd metadata: update the code to use Sentinel1GRDMetadata object
- grd: introduce Sentinel1GRDMetadata
- grd: state_vectors as list of object, fix orbits support
- metadata: update the code to use Sentinel1BurstMetadata object
- s1 metadata: introduce the class Sentinel1BurstMetadata
- s1 product: use plain rasterio for CDSE productinfo
- orbit: introduce the StateVector class

## Fixed
- tests: make s1m optional
- grd: fix orbit updates
- tests: test for assemble_multiple_grd_products_into_meta
- clean: Fix typos and add docstring
- typing: fix tests/
- typing: fix usage/zoom.py
- typing: fix eos/sar/projection_correction.py and eos/products/sentinel1/coordinate_correction.py
- typing: fix eos/products/sentinel1/orbits.py
- typing: fix eos/products/sentinel1/doppler_info.py
- typing: fix eos/products/sentinel1/burst_resamp.py
- typing: fix eos/products/sentinel1/metadata.py
- typing: fix eos/products/sentinel1/calibration.py
- typing: fix eos/products/sentinel1/assembler.py
- typing: fix eos/products/sentinel1/acquisition.py
- typing: fix eos/products/sentinel1/proj_model.py
- typing: fix eos/products/sentinel1/product.py
- typing: fix eos/sar/fourier_zoom.py
- typing: fix eos/sar/coherence.py
- typing: fix eos/sar/model.py
- typing: fix eos/sar/rtc.py
- typing: fix eos/sar/roi.py
- typing: fix eos/sar/const.py

## Removed
- proj_model: remove redundant attributes from the Sentinel1SLCBaseModel


# [0.18.0](https://github.com/Kayrros/eos-sar/compare/0.17.0..0.18.0) -- 2023-07-28

## Added
- test: Add mcf unwrapping tests
- feat: Add unwraping with MCF method

## Changed



# [0.17.0](https://github.com/Kayrros/eos-sar/compare/0.16.1..0.17.0) -- 2023-06-29

## Added
- model: Add new geometry approximation function

## Changed
- ci: use larger workers for testing

- io/osio: forward more parameters to *ReaderAt
- s1 product: cleanup and add class S3UnzippedSafeSentinel1SLCProductInfo
- sar model: replace prints by a logger
- dem: Get dem boundary using buffered geom instead of approx geom

## Fixed
- s1 product/cdse: fix endpoint and requester_pays


# [0.16.1](https://github.com/Kayrros/eos-sar/compare/0.16.0..0.16.1) -- 2023-05-17

## Changed

- s1 acquisition: lower assert threshold on row times rounding

## Fixed
- calibration: fix negative noise values:
    * negative noise map means we add energy during the calibration, which
    * introduces artefacts

## Removed
- requirements: remove `datetime`


# [0.16.0](https://github.com/Kayrros/eos-sar/compare/0.15.0..0.16.0) -- 2023-05-02

## Added
- assembler: Add swath model creation

## Changed

- proj_model: Compute all overlaps rois in swath_model

## Fixed
- overlap: Fix overlap warping to work with dict like debursting
- roi: Fix represention and Add equality and custom padding
- utils: Fix writing single array in a parent array


# [0.15.0](https://github.com/Kayrros/eos-sar/compare/0.14.1..0.15.0) -- 2023-03-22

## Added
- goldstein: Add NaN support
- goldstein: Add tests for padding and normalization
- goldstein: Add padding to reduce border effects
- tests: add a test for range_doppler.ascending_node_crossing_time
- range_doppler: add ascending_node_time function

## Changed

- grd merge: lower requirements on equality of azimuth_time_interval:
    * example with these two consecutive products:
    * S1A_IW_GRDH_1SDV_20220512T050706_20220512T050731_043173_0527F3_3126 0.001486835117836338
    * S1A_IW_GRDH_1SDV_20220512T050731_20220512T050756_043173_0527F3_FB65 0.001486827761353731
- goldstein: Replace fft_size with step
- range_doppler: simplify ascending_node_crossing_time
- codestyle: update to flake8 6.0.0
- codestyle: configure flake8 and autopep8 from config files

## Fixed
- goldstein: Fix patch normalization:
    * The triangular filter is normalized so that the shifted triangular
    * weights add up to one in the middle of the image. The filter applied to
    * each patch is also normalized so that its coefficients sum up to 1.

## Removed
- readme: remove manual install of numpy and cython


# [0.14.1](https://github.com/Kayrros/eos-sar/compare/0.14.0..0.14.1) -- 2023-01-30

## Added
- dem: add env var EOS_SAR_MULTIDEM_SOURCE for get_any_source

## Changed



# [0.14.0](https://github.com/Kayrros/eos-sar/compare/0.13.1..0.14.0) -- 2023-01-25

## Added
- goldstein: Add tests
- tutorial: Add goldstein filter for interf display
- goldstein: Add goldstein phase filter
- metadata: add `slice_count`
- demo: Added small aoi

## Changed

- grd assembler: crop the start and end of the datatake:
    * we observed that IW1 of the start and IW3 of the end contain some issues
    * (intensity gradient), which cannot be fixed by border noise removal


# [0.13.1](https://github.com/Kayrros/eos-sar/compare/0.13.0..0.13.1) -- 2023-01-10

## Changed

- ci: install phoenix

## Fixed
- tests: use update_statevectors_using_phoenix

## Removed
- s1 orbits: remove function update_statevectors_using_our_bucket:
    * the bucket will be removed soon


# [0.13.0](https://github.com/Kayrros/eos-sar/compare/0.12.2..0.13.0) -- 2022-12-16

## Added
- s1 orbits: add lru_cache on phoenix queries
- tuto: Add ortho examples
- ortho: Add complex type support
- mosaic_zoom: Add usage
- mosaic_zoom: Add test
- mosaic_zoom: Add MosaicZoomer
- burst_resamp: Add a second resampler for resampled rois
- doppler_info: Add dict representation
- max_finding: Add module as is
- fourier_zoom: Add fourier zoom module as is
- regist: Add zooming utils

## Changed

- s1 orbits: use aws:proxima:kayrros-prod-sentinel-aux as default source
- s1: optimize date string to timestamp conversion
- mosaic_zoom: Ensure zoom factor is int earlier in code
- burst_resamp: Split functions with grid_eval into two functions
- burst_resamp: Clean burst resampler
- product: Optimize pattern searching in manifest links
- product: Make SafeSentinel1ProductInfo work with s3
- max_finding: Improve handling of failures
- max_finding: Harmonize outputs and add asserts
- max_finding: Clean and Add test
- fourier_zoom: Clean and keep necessary code

## Fixed
- burst_resamp: Make small fixes in code and doc
- test_projection: Test projection with Corner Reflectors
- ci: Fix pytest-parallel temporarily with py dependency


# [0.12.2](https://github.com/Kayrros/eos-sar/compare/0.12.1..0.12.2) -- 2022-10-20

## Added
- rtc: add shadow parameters to `normalize`
- docs: add description of the border masking for GRD products

## Changed



# [0.12.1](https://github.com/Kayrros/eos-sar/compare/0.12.0..0.12.1) -- 2022-09-15

## Added
- s1 product: add parameters source and image_opener to from_product_id
- io: add open_image_osio

## Changed

- grd assembler: reimplement using utils.stitch_arrays:
    * This allows to get rid of boundless=True, which has issues with osio.


# [0.12.0](https://github.com/Kayrros/eos-sar/compare/0.11.0..0.12.0) -- 2022-09-14

## Added
- test grd assembler: add localization/projection test
- tests: add a test for grd assembler
- grd: add Sentinel1GRDAssembler to crop consecutive products
- s1 metadata: add slice_number

## Changed

- s1 product: use logging instead of prints

## Fixed
- s1 orbits: fix ASF phoenix source


# [0.11.0](https://github.com/Kayrros/eos-sar/compare/0.10.1..0.11.0) -- 2022-09-01

## Added
- border noise: added border noise masking method and tests
- calibration.py: add noise scaling factor for noise removal
- ortho: add previous_orthorectifier param to from_transform:
    * this allows to reuse previously cropped DEM

## Changed

- calibration: minor cleanup on the scaling factor
- correct date validity IPF version 2.5 (2015-07-02)
- requirements: rasterio>=1.3

## Fixed
- s1 mosaic model: fix issue on multiple from_dict calls


# [0.10.1](https://github.com/Kayrros/eos-sar/compare/0.10.0..0.10.1) -- 2022-08-01

## Added
- s1 srgr: support extrapolation outside azimuth bounds

## Changed

- ci: pin version of flake8:
    * flake8 v5.0 triggers E275 errors that were not triggered before
    * and these errors are not fixed by autopep8 yet
- s1 product: phoenix GRDP ProductInfo does not require burster
- calibration: handle non-regular noise grid:
    * Correction of way the pixels are managed in the noise loading function.
    * Because of issues of pixels list sizes for certain product, we interpolate
    * the noise LUT based on a array of pixels corresponding to the pixels
    * positions of the first row.


# [0.10.0](https://github.com/Kayrros/eos-sar/compare/0.9.0..0.10.0) -- 2022-07-12

## Added
- ortho: add eos.sar.ortho
- rtc: add terrain_flattening_cython.pyx to MANIFEST.in
- Add terrain flatting library
- coordinates: add to_col/row/azt/rng for GRDCoordinateMixin
- coordinates: add to_azt,to_rng,to_col,to_row methods

## Changed

- rtc: cleanup and introduce RadiometricTerrainCorrector
- rtc: rework the definition of the oversampling
- rtc: refactor to speed-up and integrate into eos

## Fixed
- rtc: fix get_outer_roi when dem has nans

## Removed
- rtc: remove test/ folder
- rtc: remove slow path


# [0.9.0](https://github.com/Kayrros/eos-sar/compare/0.8.0..0.9.0) -- 2022-07-04

## Added
- grd: add usage/grd.py
- grd: add sentinel1.srgr and to_azt_rng
- grd: add GRDCoordinateMixin and Sentinel1GRDModel
- regist: add function translation_matrix
- io: support **kwargs to read_window
- s1 orbits: add support of phoenix as orbit provider
- asm: Add corrector
- asm: Add doppler creation
- incidence: Add function to compute incidence

## Changed

- proj_model: better separation between SLC and GRD classes
- s1 grd: parse metadata and phoenix binding
- clean: Do some cleanups
- correction: Cleanup Various things:
    * Renamed ControlPoint and CorrectionControlPoint into Points,
    * ImagePoints, GeoPoints, GeoImagePoints. Updated Corrections accordingly.
    * Removed lon, lat, alt from constructor of GeoPoints. Cleaned shifts and
    * empty.
- tuto: Update for new interface
- debursting: Refactor resampler and debursting
- orbit/asm: Aggregate state_vectors into single orbit:
    * Single orbit instance in assembler
- regist: Update registration estimation
- proj_model: Update with orbit and corrector
- correction: Write corrections as classes
- doppler: Update doppler instanciation

## Fixed
- calibration: fix parsing of GRD noise files
- tests: fix tests

## Removed
- model: Remove apd function


# [0.8.0](https://github.com/Kayrros/eos-sar/compare/0.7.0..0.8.0) -- 2022-05-19

## Added
- assembler: add Sentinel1Assembler constructor
- assembler: Add polarization
- projection: add parameter as_azt_rng
- mosaic: add S1AcquisitionCutter
- roi: add method intersects_roi
- proj_model: add Sentinel1MosaicModel
- feat: Add product info from SAFE dir
- mosaic: add RoiProjModelWrapper.localize_without_alt
- mosaic: add file mosaic.py
- s1 proj model: add adjust_roi_to_swath to get_read_write_rois
- deburst: support multiple readers in warp_rois_read_resample_deburst:
    * this is required for assembling multiple products
- roi: add method `clip`

## Changed

- tutorial: update with the new assembler interface
- acquisition: clean up SecondarySentinel1AcquisitionCutter
- acquisition: split into (Primary|Secondary)Sentinel1AcquisitionCutter
- acquisition: simplifies mask_pts_in_burst (azt/rng as input)
- renamed mosaic.py to assembler.py
- mosaic: create objects from the Assembler, dont use SwathModel
- s1 regist: use azt/rng instead of row/col
- mosaic: update tutorial
- mosaic: don't use orbit_provider when fetching iw2 for bistatic
- calibration: make method mandatory for CalibrationReader
- calibration: use roi as parameter instead of window:
    * it's better to use a Roi instead of an arbitrary tuple
    * there was a bug with h/w handling, fixed by using Roi
- deburst: expose reramp parameter
- sar model: clean up Roi import
- s1: use bsid as indices instead of burst numbers
- deburst: optional parameter out to avoid array creation:
    * also, the generator instead of lists for stitch_array allows more memory
    * efficient creation of mosaic
- ci: extract step `checkstyle` out of `test`

## Fixed
- build: fix a cythonize compilation error happening on macos:
    * fix inspired from https://github.com/Netflix/vmaf/pull/812
- fix: Modify tutorial for current api
- proj_model: fix get_read_write_rois for horizontal intersection

## Removed
- mosaic: remove S1Assembler.get_swath_proj_model
- proj_model: remove slant_range_time from Sentinel1BaseModel


# [0.7.0](https://github.com/Kayrros/eos-sar/compare/0.6.0..0.7.0) -- 2022-04-14

## Added
- readme: add code formatting information
- tutorial: Add exhaustive tutorial
- regist!: Add roi and margin params
- calibration: Add reader class

## Changed

- s3: use bucket kayrros-dev-satellite-test-data
- io: rely on boto3 for credential management
- s1 orbits: use bucket from AWS S3
- tuto: Use dem.get_any_source in tutorial
- doc: Update srtm4 installation
- ci: check pep8 compliance with flake8
- code: apply autopep8 to fix formatting issues:
    * ```
    * autopep8 --in-place -r --ignore E501,E702,E703,E704,E711,E712 .
    * find . -type f | grep py | xargs sed -i 's/[ \t]*$//'
    * ```
- burst_resamp: Separate doppler computation:
    * Doppler computation in Sentinel1BurstResample has been put in two
    * functions outside deramping/reramping:
    * 1) original_doppler: For the regular grid before resampling
    * 2) resampled_doppler: For any rows, cols in the destination grid (after
    * resampling)
- download: Clean download script:
    * Moved download script to tools/ folder. Script accepts arguments URL
    * OUTDIR unzip(optional). Should be called for each url. User will need to
    * pass the urls and outdirs for each file.
- tutorial: display coherence
- roi: roi(0,0,0,0) in make_valid
- tutorial: Corrected product id instanciation
- tutorial: minor formatting cleanup
- roi!: Return (0,0,0,0) instead of assert
- __init__: Import coherence
- orbits: Return None if FileNotFound
- resampling!: Move functions from deburst:
    * burst_resamp: moved functions from deburst. Simplified the interface to deal with 1 burst.
    * overlap: moved functions from deburst and use new simplified funcs from
    * burst_resamp.
    * deburst: less functions and new simplified funcs from burst_resamp. Move
    * some functions to eos.sar.utils
    * utils: Add writing roi to array functions from deburst
    * regist: return dict of Nones when no resampling should occur
    * __init__: import overlap

## Fixed
- code: fix pep8 errors
- tutorial: fix bug in filter burst and poe
- dem_to_radar: fix shapely deprecation
- download: fix unzip bug
- flat_earth: Fix prediction in get_geom_config
- dem_to_radar: Fix docstring

## Removed
- clean: Remove usage, update README:
    * remove old usage files
    * update README and requirements and .gitignore
- io: Remove boto3 necessary import


# [0.6.0](https://github.com/Kayrros/eos-sar/compare/0.5.1..0.6.0) -- 2022-02-01

## Added
- Added srtm4 source
- dem: add MultidemSource and SRTM4Source:
    * SRTM4Source is not yet complete (`crop` is missing)
- test resampling: add a few asserts for burst resampling
- s1 orbits: add function `update_statevectors_using_local_folder`:
    * Combined with the tool `s1-download-orbits`, it removes the strong
    * dependency to the s1-orbits S3 bucket.

## Changed

- ci: install multidem
- s1 orbits: simplify the S3 filelist cache:
    * functools.lru_cache was caching the s3 clients, thus keeping them alive

## Fixed
- deburst: fix get_bursts_intersection return value:
    * since f01ef77961e48f083cf64d231c604bc523714317 this functions returns a
    * single array, but the 'no intersection' case was still returning two
    * values
- test resampling: minor cleanup

## Removed
- Removed multidem from requirements
- Removed TODO
- s1 proj_model: remove lines_per_burst from the swath model
- proj model: remove samples_per_burst from Sentinel1SwathModel


# [0.5.1](https://github.com/Kayrros/eos-sar/compare/0.5.0..0.5.1) -- 2022-01-14

## Added
- ci: add fakedeploy job to validate packaging
- tests: add some bsid hard cases
- tests: add annotation with azimuthAnxTimeanxtime > T_orb

## Changed

- burst ids: wrap the relative_burst_id to N_bursts_per_cycle:
    * This makes sure that at the end of orbit 175 the relative_burst_id is
    * not 375890 but 3, for example.
- burst ids: change the definition of the absolute_burst_id:
    * The new definition is more reliable (assuming the relative_burst_id is
    * reliable).
    * The residual of the rounding is now consistent across the mission
    * (except for some products).
- s1 metadata: replace assert by warning if burst ids are different
- s1 metadata: use _mid_burst_sensing_time_correction
- s1 metadata: use burst sensingTime in burstID formula:
    * and refactor burstID function


# [0.5.0](https://github.com/Kayrros/eos-sar/compare/0.4.0..0.5.0) -- 2022-01-14

## Added
- calibration: add some asserts to make sure the grid is as expected
- tests: add test for eos.sar.coherence
- tests: add test for calibration
- calibration: add parameter dont_clip_noise
- add missing scipy requirements for coherence.py
- calibration: add note about SNAP
- add eos.products.sentinel1.calibration
- add eos.sar.coherence
- Added pytest fixtures
- Added option to save when getting regist dem pts
- Added docstring
- Added nan to non valid parts of debursted array
- Added return value roi and resampler
- Added usage for any swath
- Added resampling based on projection corrections:
    * Registration based on projection corrections. Debursting has been
    * adapted. Overlap funcs also added.

## Changed

- ci: use pytest-parallel
- setup: don't package tests
- ci: install numpy
- ci: install cython
- Fixed with new debursting api
- Updated usage
- write_roi instead of dcols for ovl

## Fixed
- code: minor formating fixes and doc
- coherence: fix set_borders_to_nan when filter_size=1
- fixed geom test

## Removed
- calibration: remove s1c dependency by copypasting its functions


# [0.4.0](https://github.com/Kayrros/eos-sar/compare/0.3.0..0.4.0) -- 2021-12-13

## Added
- s1 metadata: add attribut `approx_altitude`:
    * it corresponds to the heights given in the gcp of the product, for the
    * four corners of the burst
- Added usage for radarcoding with lonlat
- Added x,y interpolation in radarcoding
- Added initialization for projection/localization
- Added tests geom phase
- Added usage geom phase
- Added geometric phase computation
- Added some utils
- Added shapely and scipy as requirements
- Added tests for radarcoding
- Added radarcoding
- Added roi localization
- Added elevation as argument
- Added tests for localize_without_alt
- Added usage for localize without alt
- Added docstring
- Added localize without alt
- Added attributes to the Sensor model
- Added tests
- Added computation of geometric configuration
- Added 2D polynomial fitting
- Added ROI class for reading
- Added ROI class for S1 models
- Added ROI class
- Added type check for reading
- Added support for amplitude debursting
- Added support for amplitude resampling
- Added support for amplitude reading
- proj_model: add method adjust_roi_to_swath:
    * it can be used to make sure to use the same roi as the one used in
    * get_read_write_rois during debursting and subsequent calls to projection
- s1.orbits: add documentation for update_statevectors_using_our_bucket

## Changed

- ci: enable unsafe SSL for gdal:
    * once the SSL issues are fixed on the docker images / python
    * dependancies, we should revert this commit
- Updated tests and usage
- Fixed docstring
- s1 proj: implement full_bistatic_correction and intra_pulse_correction:
    * refactor some functions into `eos/products/sentinel1/doppler_info.py`
- Modified radarcoding test
- Moved raster grid computation to utils
- Updated tests and usage of localize_without_alt
- s1 orbits: expose search_valid_orbit_files and add parameter fullsearch:
    * the additional function is useful for debugging purposes
- Fixed uppercase in var and class names
- s1 orbits: select the latest orbit file available:
    * (the following was observed for restituted orbit files)
    * for a given product, there might be a few compatible orbit files
    * some are generated just 30minutes after the first generation
    * others might appear even 10 days after the date
- Updated usage and tests with ROI class
- Modified usage and tests
- s1 orbits: improve state vectors extraction for products <=2015:
    * in products of 2015, the state vectors are stored every 1sec instead of 10sec
    * but only cover ~30 secs instead of ~3 minutes
    * eg:
    *     S1A_IW_SLC__1SDV_20150509T144857_20150509T144924_005846_007860_CFEE-iw3.xml
    *     S1A_IW_SLC__1SDV_20150813T144901_20150813T144928_007246_009EB8_A00C-iw3.xml
- Update usage/secondary_image_debursting.py
- regist: specify a geometry for dem_points instead of using approx_geom
- burst_resamp: slight optimization of deramp:
    * on a S1 burst, it takes 2.1s instead of 2.6s on my computer
- orbits: move out of metadata and add helper update_statevectors_using_our_bucket

## Fixed
- s1 orbits: fix orbit file selection:
    * issue found on BSID 322379_IW3 (S1B_IW_SLC__1SDV_20210106T005940_20210106T010006_025027_02FA8B_223A)
    * whose timing is at the border of one of the orbit files

## Removed
- Removed Doppler from SwathModel
- Removed corrections from Swath Model
- Removed inplace from parameters when default


# [0.3.0](https://github.com/Kayrros/eos-sar/compare/0.2.0..0.3.0) -- 2021-08-19

## Added
- io: support credentials through env vars
- Added tests for stitching
- Added resampling tests
- Added support for OpenIO reading.:
    * Functions to read tiff images and xml annotations from openio have been
    * migrated from s1m. Deburst module functions where images are read have
    * been updated, and the usage as well. Support for s1-burster needs to be
    * added.
- Added usage for debursting
- Added some usage functionnalities
- Added intersection of bursts.:
    * When doing interferometry, the sentinel1 swath may contain different
    * burst relative ids (spatial location) in the secondary image ( usually
    * it may happen to have a burst translation from a product to another, or
    * just some additional bursts). The common bursts can be found by the
    * added function.
- Added deburst module:
    * The deburst module, along with the Sentine1Swath Model now enable us to
    * deduce the regions that need to be read from each burst to get an aoi.
    * Furthermore, these regions are adapted to the secondary location, are
    * read, resampled and debursted.
- Added some debursting capabilites
- Added possibility of resampling a roi within a burst
- Added roi manipulation functions
- Added support for Swath projection:
    * Sentinel1 projection is now more generic. The only difference between a
    * swath and a burst is in the origin times and positions. The actual
    * physical sensor model algorithm with bistatic and apd correction is the
    * same.

## Changed

- Updated README for usage
- Updated for Env Var OIO reading
- Improved docstring
- Refined deburst functions i/o
- Moved io module to sar package
- modified reading func name
- Improved swathModel instantiation
- Fixed Docstring

## Removed
- Removed Uppercase variables


# [0.2.0](https://github.com/Kayrros/eos-sar/compare/0.1.0..0.2.0) -- 2021-08-17

## Added
- src: add __init__.py files to fix packaging
- S1 metadata: add burst attribute "bsid":
    * it is just the concatenation of the `relative_burst_id` and the `swath`
- S1 metadata: add functions to parse precise orbit's state_vectors

## Changed
- bump to 0.2.0
- S1 metadata: adjust T_pre for IW2 and IW3
- S1 metadata: adjust T_pre (with supporting tests)
- S1 metadata: adjust T_pre for IW2 (with supporting tests)
- S1 metadata: check against all burstId samples from ESA

## Fixed
- tests: fix indentation
- S1 metadata: fix burst id computation when changing orbits


# [0.1.0](https://github.com/Kayrros/eos-sar/compare/..0.1.0) -- 2021-07-08

## Added
- S1 metadata: add function extract_burst_metadata and fix tests
- tests: add tests from the IPF 3.9 draft (burst id)
- S1 metadata: add 'swath' field
- Added metadata reading from xml.:
    * s1m dependency has been removed. Metadata reading is now performed
    * directly from an xml file content as string.
- Added initialization from meta
- Added usage
- Added deramping and reramping:
    * Resampling a complex burst requires deramping, resampling, reramping.
- Added io function
- Added Deramping/Reramping
- Added affine registration estimation:
    * A (sensor) generic affine regist matrix estimation has been added.
    * Points on the dem are projected on both images, and the matrix is then
    * estimated. A resampling function is also provided. Deramping and
    * reramping still need to be implemented.
- Added requirements
- Added some tests
- Added creation from s1m
- Added products
- burst metadata: add attributes azimuth_anx_time and lon_lat_bbox:
    * this requires the future latest s1m version
- Add burst localization function:
    * It is now possible to localize from the burst coordinate system +
    * altitude to lon, lat. APD correction and Bistatic correction is
    * supported. It has been verified that this function is the inverse of
    * projection.
- Added orbit param for projection:
    * In burstprojection, it is now possible to pass the function a fitted
    * orbit, this way we avoid fitting the polynomial coefficients each time
    * the projection function is called.
- Added const module:
    * Constants are now stored in a const module. Each constant name is formed
    * by joining the physical meaning with the unit ( all in uppercase) using
    * underscores.
- Added burst projection:
    * The projection into a sentinel1 burst has been added here. The burst is
    * referenced to its first valid line and column. The metadata is extracted
    * using s1m for a product/subswath and burst index.
- Added backprojection:
    * Low level functions for projection and localisation common to all sar
    * satellites using state-vectors and focused to Zero-Doppler. Two
    * different implementations of projection available for use.
- Added Chebychev orbit interpolation:
    * Ephemerides in the metadata is given as state vectors,a list of samples
    * (time, position, speed). Interpolation can be done using chebychev
    * polynomials

## Changed
- metadata: small cleanup
- Modified metadata reading:
    * It is now possible to read the metadata for multiple bursts at once. The
    * common metadata (relating to the swath and the sensor) are read once.
    * However, they are still stored multiple times ( in each burst
    * dictionnary). This might change later.
- S1 metadata: compute relative and absolute burst ids:
    * the computation was calibrated and validated against the three products provided as samples:
    * S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml
    * S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650.SAFE/annotation/s1a-iw2-slc-vh-20210216t151208-20210216t151233-036617-044d40-002.xml
    * S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650.SAFE/annotation/s1a-iw3-slc-vh-20210216t151206-20210216t151231-036617-044d40-003.xml
    * S1A_IW_SLC__1SDV_20191124T141637_20191124T141709_030054_036E99_E375.SAFE/annotation/s1a-iw1-slc-vh-20191124t141637-20191124t141707-030054-036e99-001.xml
    * S1A_IW_SLC__1SDV_20191124T141637_20191124T141709_030054_036E99_E375.SAFE/annotation/s1a-iw2-slc-vh-20191124t141638-20191124t141708-030054-036e99-002.xml
    * S1A_IW_SLC__1SDV_20191124T141637_20191124T141709_030054_036E99_E375.SAFE/annotation/s1a-iw3-slc-vh-20191124t141639-20191124t141709-030054-036e99-003.xml
    * S1B_IW_SLC__1SDV_20210210T183043_20210210T183110_025548_030B56_37E0.SAFE/annotation/s1b-iw1-slc-vh-20210210t183045-20210210t183110-025548-030b56-001.xml
    * S1B_IW_SLC__1SDV_20210210T183043_20210210T183110_025548_030B56_37E0.SAFE/annotation/s1b-iw2-slc-vv-20210210t183043-20210210t183109-025548-030b56-005.xml
    * S1B_IW_SLC__1SDV_20210210T183043_20210210T183110_025548_030B56_37E0.SAFE/annotation/s1b-iw3-slc-vv-20210210t183044-20210210t183110-025548-030b56-006.xml
    * 
    * further testing will be required once the IPF is used to generate the SLC products
- regist: use cv2.warpAffine to apply the affine transform
- Reformated files for readability
- Adapted meta reading for test
- Sentinel1BurstResample: reduce memory usage:
    * to register a complete burst:
    * before: 5231MB
    * after: 2377MB
- Fixed test for s1m master
- Fixed small typo in readme
- Update README.md
- Modified io
- Improved code readability
- Separated registration steps:
    * It is now possible to read the dem in a step. The we need to project in
    * primary frame ourselves. The we call orbital_regist to project in
    * secondary frame. This separation is for practical computational purpose
    * so we can reuse dem and projected points on all secondary bursts.
- Modified requirements
- Fixed projection test
- Ensured coherence of variables:
    * Most important change: projection and localization now deals with row,
    * col in this order.
    * Image coordinates named row, col. Can be converted to azt, rng.
    * geocentric coordinates gx, gy, gz. 3D coordinates x, y, alt.
    * Other parts of the code have been cleaned, and tests adapted.
- Fixed Docstring
- Fixed init
- Filled S1BurstModel
- init
- Separated generic and S1 algos
- Renamed backproj to range_doppler
- Switched to row col convention
- Fixed spelling and readability
- Performed min, max on list
- Formatted with AutoPEP8
- Vectorized localization function:
    * solve_range_doppler is now vectorized
- Cleaned localization formulas:
    * Formulas used for the localization are now prettier. We also avoid
    * copying the constants into local vars. solve_range_doppler signature is
    * also changed.
- Reformatted Docstring
- Change Orbit evaluate method:
    * A single method is now implemented to evaluate the position, speed,
    * acceleration and higher order derivatives along the orbit
- Replaced epsg by crs:
    * The epsg int identifiers might not always work.
    * Thus, we repalced by crs string in burst_projection.
- Forced user to pass orbit to proj & loc func:
    * User is now forced to pass a fitted Orbit instance to burst_projection
    * and burst_localization instead of having the option to fit the orbit
    * inside the functions from the burst_metadata.
- Fixed docstring and camelCase:
    * Docstring is now formatted uniformly in all modules. camleCase functions
    * renamed.
- Fixed imports
- init
- Initial commit

## Fixed
- ci: fix pytest execution
- S1 metadata: fix issue with burst id when completing an orbit
- tests: fix running tests with pytest

## Removed
- Removed closest_approach function
- Removed relative imports
- Removed code authors



