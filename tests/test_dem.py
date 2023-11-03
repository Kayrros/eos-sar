import time

import numpy as np
import pytest

import eos.dem


def test_dem_sources_elevation():
    # interp = "bilinear"
    interp = "nearest"

    # bounds = (3, 42, 8, 47)
    bounds = (3, 44, 4, 46)
    lons = np.random.rand(10) * (bounds[2] - bounds[0]) + bounds[0]
    lats = np.random.rand(10) * (bounds[3] - bounds[1]) + bounds[1]

    dem_source: eos.dem.DEMSource

    dem_source = eos.dem.SRTM4Source()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    # np.save("/tmp/t/a", dem.array)
    # eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/a.tif")

    dem_source = eos.dem.MultidemSource()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    # np.save("/tmp/t/b", dem.array)
    # eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/b.tif")

    dem_source = eos.dem.DEMStitcherSource()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    # np.save("/tmp/t/d", dem.array)
    # eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/d.tif")

    # TODO: assert things


def test_dem_crop(tmp_path):
    bounds1 = (3, 44, 4, 46)
    dem = eos.dem.SRTM4Source().fetch_dem(bounds1)

    bounds2 = (3.1, 44.3333, 4, 46)
    a, b, c = dem.crop(bounds2)

    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, tmp_path / "orig.tif")
    eos.dem.write_crop_to_file(a, b, c, tmp_path / "crop.tif")

    # TODO: assert things


def test_dem_crop2(tmp_path):
    bounds1 = (3, 44, 4, 46)
    bounds2 = (3.1, 44.3333, 3.6, 45.9)

    dem = eos.dem.SRTM4Source().fetch_dem(bounds1)
    a, b, c = dem.crop(bounds2)

    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, tmp_path / "orig.tif")
    eos.dem.write_crop_to_file(a, b, c, tmp_path / "crop.tif")


if False:
    # not passing yet, because the different DEMSource tend to more or less respect the bounds
    # and also they all have different transform conventions?..
    def test_dem_subset_same():
        bounds = (3, 44, 4, 46)
        dem = eos.dem.DEMStitcherSource().fetch_dem(bounds)
        dem2 = dem.subset(bounds)
        assert dem == dem2


def test_dem_subset_and_crop():
    bounds1 = (3, 44, 4, 46)
    bounds2 = (3.1, 44.3333, 3.6, 45.9)
    bounds3 = (3.2, 44.5333, 3.6, 45.8)

    dem = eos.dem.SRTM4Source().fetch_dem(bounds1)
    dem2 = dem.subset(bounds2)
    assert dem2.array.shape[0] < dem.array.shape[0]
    assert dem2.array.shape[1] < dem.array.shape[1]

    # cropping the parent or the child should give the same result
    a1, b1, c1 = dem.crop(bounds3)
    a2, b2, c2 = dem2.crop(bounds3)
    assert (a1 == a2).all()
    # due to the float computation, some precision is lost in the transform
    assert np.allclose(b1, b2)
    assert c1 == c2


def test_dem_crop_outofbounds():
    bounds1 = (3, 44, 3.1, 45)
    bounds2 = (2.95, 44, 3.1, 45)

    dem = eos.dem.SRTM4Source().fetch_dem(bounds1)
    with pytest.raises(eos.dem.OutOfBoundsException):
        dem.subset(bounds2)


def test_dem_elevation():
    import srtm4

    lon = 31.1
    lat = 31.5
    alt = srtm4.srtm4(lon, lat)

    dem = eos.dem.SRTM4Source().fetch_dem(
        (lon - 1.3, int(lat - 1), lon + 0.2, lat + 0.2)
    )
    alt2 = dem.elevation(lon, lat, "nearest")

    assert abs(alt - alt2) < 0.1


def test_dem_elevation_outofbounds():
    lon = 31.1
    lat = 31.5

    dem = eos.dem.SRTM4Source().fetch_dem(
        (lon - 0.5, int(lat - 0.5), lon + 0.5, lat + 0.5)
    )
    dem.elevation(lon, lat)

    with pytest.raises(eos.dem.OutOfBoundsException):
        dem.elevation(lon - 0.6, lat)
    with pytest.raises(eos.dem.OutOfBoundsException):
        dem.elevation(lon + 0.6, lat)
    with pytest.raises(eos.dem.OutOfBoundsException):
        dem.elevation(lon, lat - 0.6)
    with pytest.raises(eos.dem.OutOfBoundsException):
        dem.elevation(lon, lat + 0.6)
