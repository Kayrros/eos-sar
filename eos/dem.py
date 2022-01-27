import rasterio

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


def get_any_source():
    if has_multidem:
        return MultidemSource()
    if has_srtm4:
        return SRTM4Source()
    raise RuntimeError("couldn't find a DEM source; please install multidem or srtm4")

