import eos.dem


def test_phx_dem():
    # lons, lats = (3, 43)
    lons, lats = (3.1, 43.9)
    # lons, lats = (3, 43)

    interp = "bilinear"
    # interp = "nearest"

    import numpy as np
    # lons = np.random.rand(100) * 360 - 180
    # lats = np.random.rand(100) * 180 - 90

    lons = np.random.rand(10) * 5 + 3
    lats = np.random.rand(10) * 5 + 42

    import time

    # dem = eos.dem.SRTM4Source()
    # t = time.time()
    # print(dem.elevation(lons, lats))
    # print(time.time() - t)

    dem = eos.dem.MultidemSource()
    t = time.time()
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)

    dem = eos.dem.GLO30Source()
    t = time.time()
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)

    assert False
