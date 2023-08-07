import eos.dem


def test_phx_dem():
    # lons, lats = (3, 43)
    lons, lats = (3.1, 43.9)
    # lons, lats = (3, 43)

    interp = "bilinear"
    # interp = "nearest"

    import numpy as np

    bounds = (3, 42, 8, 47)
    lons = np.random.rand(10) * (bounds[2] - bounds[0]) + bounds[0]
    lats = np.random.rand(10) * (bounds[3] - bounds[1]) + bounds[1]

    import time

    dem_source = eos.dem.SRTM4Source()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    print(dem.elevation(lons, lats))
    print(time.time() - t)
    np.save("/tmp/t/a", dem.array)
    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/a.tif")

    dem_source = eos.dem.MultidemSource()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    np.save("/tmp/t/b", dem.array)
    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/b.tif")

    dem_source = eos.dem.DEMStitcherSource()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    np.save("/tmp/t/c", dem.array)
    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/c.tif")

    assert False
