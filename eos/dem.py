import os
from typing import Iterable

import numpy as np
import rasterio
import rasterio.session
import rasterio.errors
import pyproj

try:
    import multidem
    has_multidem = True
except ImportError:
    has_multidem = False

try:
    import srtm4
    has_srtm4 = True
except ImportError:
    has_srtm4 = False


# copy-pasted from multidem
def write_crop_to_file(array, transform, crs, path):
    """
    Write a georeferenced raster to a GeoTIFF file.

    Args:
        array (np.ndarray): raster array
        transform (affine.Affine): raster transform
        crs (rasterio.crs.CRS): raster CRS
        path (str): path to output file
    """
    height, width = array.shape
    profile = dict(driver="GTiff",
                   count=1,
                   width=width,
                   height=height,
                   dtype=array.dtype,
                   transform=transform,
                   crs=crs,
                   tiled=True,
                   compress="deflate",
                   predictor=2,
                   blockxsize=256,
                   blockysize=256)

    with rasterio.open(path, "w", **profile) as f:
        f.write(array, 1)


class DEMSource:
    """
    Notes:
        The only supported datum is ellipsoidal.
    """

    def elevation(self, lons, lats, interpolation="bilinear"):
        """
        Gives the altitude of a (list of) point(s).

        Args:
            lons, lats: longitude (or list of longitudes) and latitude (or list of latitudes)
            interpolation (str): if 'bilinear' (default) returns the height bilinearily interpolated,
                else if 'nearest' returns the nearest neighbor value
        Returns:
            alts: height (or list/array of heights) in meters above the given datum
        """
        raise NotImplementedError

    def crop(self, bounds):
        """
        Return a crop of the `source` DEM on the given `bounds`.

        Args:
            bounds (4-tuple): tuple of floats lon_min, lat_min, lon_max, lat_max

        Returns:
            np.ndarray: raster DEM crop array
            affine.Affine: transform
            rasterio.crs.CRS: CRS
        """
        raise NotImplementedError


class SRTM4Source(DEMSource):

    def elevation(self, lons, lats, interpolation="bilinear"):
        assert interpolation == "bilinear"
        return srtm4.srtm4(lons, lats)

    def crop(self, bounds):
        return srtm4.crop(bounds, datum="ellipsoidal")


class MultidemSource(DEMSource):

    def __init__(self, demsource="SRTM30"):
        """
        Args:
            demsource (str): DEM source "SRTM30" (default), "TDM90", "SRTM90" or
                "SRTM90-CGIAR-CSI"
        """
        multidem._source_validation(demsource)
        self.demsource = demsource

    def elevation(self, lons, lats, interpolation="bilinear"):
        return multidem.elevation(lons, lats, interpolation=interpolation,
                                  source=self.demsource, datum="ellipsoidal")

    def crop(self, bounds):
        return multidem.crop(bounds, source=self.demsource, datum="ellipsoidal")


def glo30_interpolate_uris(lons: list[float], lats: list[float]) -> list[str]:
    buckets: set[tuple[int, int]] = set()
    def bucket_of(lon, lat) -> tuple[int, int]:
        return (int(lon), int(lat))
    for lon, lat in zip(lons, lats):
        buckets.add(bucket_of(lon, lat))

    uri_per_bucket: dict[tuple[int, int], str] = {}
    for bucket in buckets:
        lon, lat = bucket
        if lon < 0:
            lonstr = f'W{abs(lon):03d}'
        else:
            lonstr = f'E{lon:03d}'
        if lat < 0:
            latstr = f'S{abs(lat):02d}'
        else:
            latstr = f'N{lat:02d}'
        name = f"Copernicus_DSM_COG_10_{latstr}_00_{lonstr}_00_DEM"
        uri_per_bucket[bucket] = f"s3://copernicus-dem-30m/{name}/{name}.tif"

    uris = []
    for lon, lat in zip(lons, lats):
        uris.append(uri_per_bucket[bucket_of(lon, lat)])
    return uris

def get_gdal_options():
    return {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "VSI_CACHE": True,
    }

def shift_height(lons, lats, alts):
    # WGS84 with ellipsoid height as vertical axis
    ellipsoid = pyproj.CRS.from_epsg(4979)
    # WGS84 with Gravity-related height (EGM96)
    geoid = pyproj.CRS("EPSG:4326+5773")

    trf = pyproj.Transformer.from_crs(geoid, ellipsoid)

    alts = trf.transform(lats, lons, alts, errcheck=True)[-1]
    return np.around(alts, 5)


def compute_filenames(lons, lats):
    """Computes the name of the DEM tiles which the points belong to.

    Args:
        lons, lats (numpy.ndarray): 1-D arrays of longitudes and latitudes
        source (str): DEM source

    Returns:
        numpy.ndarray: 1-D array of DEM tile filenames.
    """
    ew = np.full(len(lons), "E", dtype=object)
    ns = np.full(len(lats), "N", dtype=object)

    ns[lats < 0] = "S"
    ns_int = np.abs(np.floor(lats)).astype(int)
    ns_str = [str(x).rjust(2, "0") for x in ns_int.tolist()]

    ew[lons < 0] = "W"
    ew[lons == 180] = "W"
    ew_int = np.floor(lons)

    ew_int = np.abs(ew_int).astype(int)
    ew_str = [str(x).rjust(3, "0") for x in ew_int.tolist()]

    filenames = "s3://copernicus-dem-30m/" + "Copernicus_DSM_COG_10_" + ns + ns_str + "_00_" + ew + ew_str + "_00_DEM/" + "Copernicus_DSM_COG_10_" + ns + ns_str + "_00_" + ew + ew_str + "_00_DEM.tif"
    return filenames

def get_files_name(lons, lats):
    """Get a dictionary sorting the input points per DEM tile filename.

    Args:
        lons, lats: longitude (or list of longitudes) and latitude (or list of latitudes)
        source (str): DEM source

    Returns:
        dict: dictionary of tif names (key) and list of lon/lat indices (value)
    """
    if not isinstance(lats, np.ndarray):
        lats = np.array(lats)
    if not isinstance(lons, np.ndarray):
        lons = np.array(lons)

    from collections import defaultdict
    names = defaultdict(list)
    filenames = compute_filenames(lons, lats)
    for i, name in enumerate(filenames):
        names[name].append(i)

    return names

def bilinear_interp(array, x, y):
    """Returns the value for the fractional row/col using bilinear interpolation
        between the cells.

    Args:
        array (numpy.ndarray): 2-D array of floats.
        x (list): horizontal image coordinates.
        y (list): vertical image coordinates.

    Returns:
        numpy.ndarray: 1-D array of floats.
    """
    i = np.array(np.floor(x), dtype=int)
    j = np.array(np.floor(y), dtype=int)

    dx, dy = x - i, y - j

    ones = np.ones(len(x))
    u, v = ones - dx, ones - dy

    h_interp = (
        u * v * array[:, 0] + dx * v * array[:, 1] + u * dy * array[:, 2] + dx * dy * array[:, 3]
    )

    return np.around(h_interp, 5)

def get_window(x, y):
    x_int, y_int = int(x), int(y)
    if x == x_int and y == y_int:
        return rasterio.windows.Window(x_int, y_int, 1, 1)
    elif x == x_int:
        return rasterio.windows.Window(x_int, y_int, 1, 2)
    elif y == y_int:
        return rasterio.windows.Window(x_int, y_int, 2, 1)
    else:
        return rasterio.windows.Window(x_int, y_int, 2, 2)

def get_heights_from_img(img, lons, lats, interpolation="bilinear"):
    """Get the height of the given points in the given DEM tile.

    Args:
        img (str): path to the DEM image file.
        lons, lats: lists of longitudes and latitudes
        interpolation (str): if 'bilinear' (default) returns the height bilinearily
            interpolated, else if 'nearest' returns the nearest neighbor value

    Returns:
        numpy.ndarray: 1-D array of heights.
    """
    if interpolation not in ["nearest", "bilinear"]:
        raise ValueError("interpolation must be nearest or bilinear")

    geo_coords = np.array([lons, lats])

    # Handling special case when the longitude is equal to 180 to avoid having
    # image coordinates after projection to be out of bounds
    geo_coords[0][geo_coords[0] == 180] = -180

    with rasterio.open(img) as src:
        nodata = src.nodata

        if interpolation == "nearest":
            gen = src.sample(zip(geo_coords[0], geo_coords[1]))
            heights = np.array([np.nan if px[0] == nodata else px[0] for px in gen])

        else:
            # The (0.5, 0.5) pixels offset converts image coordinates using the "PixelIsArea"
            # convention to image coordinates using the "PixelIsPoint" convention.
            # These two conventions are defined in the GeoTIFF specification [1].
            # With the PixelIsArea convention the origin of the coordinate system
            # is located on the top-left corner of the top-left pixel of the image.
            # With the PixelIsPoint convention the origin of the coordinate system
            # is located at the center of the top-left pixel of the image.
            # GDAL and rasterio use the PixelIsArea convention [2] thus the coordinates obtained
            # with ~src.transform are to be interpreted with the PixelIsArea convention.
            # We convert them to the PixelIsPoint convention because it's more practical
            # to read and interpolate the image values.
            #
            # References:
            # [1] http://geotiff.maptools.org/spec/geotiff2.5.html#2.5.2.2
            # [2] https://gdal.org/user/raster_data_model.html#raster-data-model
            img_coords = np.around(~src.transform * geo_coords, 6) - 0.5

            dem_subparts = []
            for x, y in zip(img_coords[0], img_coords[1]):
                win = get_window(x, y)
                dem = src.read(1, window=win, out_shape=(2, 2)).astype(np.float32)
                dem[dem == nodata] = np.nan
                dem_subparts.append(dem.flatten())

    if interpolation != "nearest":
        dem_subparts = np.stack(dem_subparts, axis=0)
        heights = bilinear_interp(dem_subparts, img_coords[0], img_coords[1])

    return heights

def elevation(lons, lats, interpolation="bilinear"):
    """
    Gives the altitude of a (list of) point(s).

    Args:
        lons, lats: longitude (or list of longitudes) and latitude (or list of latitudes)
        source (str): DEM source 'SRTM30' (default), 'TDM90', 'SRTM90' or 'SRTM90-CGIAR-CSI'
        interpolation (str): if 'bilinear' (default) returns the height bilinearily interpolated,
            else if 'nearest' returns the nearest neighbor value
        datum (str): 'ellipsoidal' (height above WGS84 ellipsoid, default),
            'orthometric' (height above EGM96 geoid)
    Returns:
        alts: height (or list/array of heights) in meters above the given datum
    """

    is_input_iterable = isinstance(lons, Iterable)

    if not isinstance(lats, Iterable):
        lats = [lats]
    if not isinstance(lons, Iterable):
        lons = [lons]

    assert len(lons) == len(lats), "arguments must have same length"

    names = get_files_name(lons, lats)

    gdal_options = get_gdal_options()

    alts = np.empty(len(lons), dtype=np.float32)
    with rasterio.Env(aws_unsigned=True, **gdal_options):
        for img_path, indexes in names.items():
            lons_array = [lons[i] for i in indexes]
            lats_array = [lats[i] for i in indexes]

            import time
            print(time.time(), img_path)
            try:
                heights = get_heights_from_img(img_path, lons_array, lats_array, interpolation)
            except rasterio.errors.RasterioIOError as e:
                if str(e) == "The specified key does not exist.":
                    import warnings
                    class OutOfCoverageWarning(UserWarning):
                        pass
                    warnings.warn("Geo-point(s) not in a covered area.", OutOfCoverageWarning)
                    heights = [np.nan for _ in indexes]
                else:
                    raise e

            alts[indexes] = heights

    alts = shift_height(lons, lats, alts)

    if not is_input_iterable:
        return alts[0]
    elif isinstance(lons, np.ndarray):
        return np.asarray(alts)
    else:
        return alts.tolist() if isinstance(alts, np.ndarray) else alts



class GLO30Source(DEMSource):

    def __init__(self):
        pass

    def elevation(self, lons, lats, interpolation="bilinear"):
        if False:
            if not isinstance(lons, Iterable):
                lons = [lons]
                lats = [lats]
            import shapely.geometry
            points = shapely.geometry.MultiPoint([(lon, lat) for lon, lat in zip(lons, lats)])
            import phoenix.catalog
            client = phoenix.catalog.Client()
            collection = client.get_collection("esa-dem-glo-30").at("aws:s3:copernicus-dem-30m")
            items = list(collection.search_items(geometry=points))

            print(points)
            print(items)

        return elevation(lons, lats, interpolation)

    def crop(self, bounds):
        raise NotImplemented


def get_any_source():
    if has_multidem:
        demsource = os.environ.get('EOS_SAR_MULTIDEM_SOURCE', 'SRTM30')
        return MultidemSource(demsource)
    if has_srtm4:
        return SRTM4Source()
    return GLO30Source()
