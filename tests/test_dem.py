import time

import numpy as np
import pytest
from shapely import box

import eos.dem


def test_dem_sources_elevation():
    # interp = "bilinear"
    interp = "nearest"

    # bounds = (3, 42, 8, 47)
    bounds = (3, 44, 4, 46)

    dem_source: eos.dem.DEMSource

    dem_source = eos.dem.SRTM4Source()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    actual_bounds = dem.get_extent()
    lons = np.random.rand(10) * (actual_bounds[2] - actual_bounds[0]) + actual_bounds[0]
    lats = np.random.rand(10) * (actual_bounds[3] - actual_bounds[1]) + actual_bounds[1]
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    # np.save("/tmp/t/a", dem.array)
    # eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/a.tif")

    dem_source = eos.dem.DEMStitcherSource()
    t = time.time()
    dem = dem_source.fetch_dem(bounds)
    actual_bounds = dem.get_extent()
    lons = np.random.rand(10) * (actual_bounds[2] - actual_bounds[0]) + actual_bounds[0]
    lats = np.random.rand(10) * (actual_bounds[3] - actual_bounds[1]) + actual_bounds[1]
    print(dem.elevation(lons, lats, interp))
    print(time.time() - t)
    # np.save("/tmp/t/d", dem.array)
    # eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, "/tmp/t/d.tif")

    # TODO: assert things


@pytest.mark.parametrize(
    "dem_name",
    [
        "srtm4",
        "glo_30",
        "glo_90",
    ],
)
def test_dem_crop(tmp_path, dem_name):
    dem_source: eos.dem.DEMSource
    if dem_name == "srtm4":
        dem_source = eos.dem.SRTM4Source()
    else:
        dem_source = eos.dem.DEMStitcherSource(dem_name)

    bounds1 = (3, 44, 4, 46)
    dem = dem_source.fetch_dem(bounds1)
    assert box(*bounds1).within(box(*dem.get_extent()))

    bounds2 = (3.1, 44.3333, 4, 46)
    a, b, c = dem.crop(bounds2)

    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, tmp_path / "orig.tif")
    eos.dem.write_crop_to_file(a, b, c, tmp_path / "crop.tif")


def test_dem_crop2(tmp_path):
    bounds1 = (3, 44, 4, 46)
    bounds2 = (3.1, 44.3333, 3.6, 45.9)

    dem = eos.dem.SRTM4Source().fetch_dem(bounds1)
    a, b, c = dem.crop(bounds2)

    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, tmp_path / "orig.tif")
    eos.dem.write_crop_to_file(a, b, c, tmp_path / "crop.tif")


def test_dem_subset_same():
    bounds = (3, 44, 4, 46)
    dem = eos.dem.DEMStitcherSource().fetch_dem(bounds)
    dem2 = dem.subset(bounds)

    assert np.all(dem.array == dem2.array)
    assert dem.transform == dem2.transform
    assert dem.crs == dem2.crs


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


def test_mydemsource(tmp_path):
    lon = 31.1
    lat = 31.5
    bounds = (lon - 0.5, int(lat - 0.5), lon + 0.5, lat + 0.5)
    dem = eos.dem.SRTM4Source().fetch_dem(bounds)

    path = tmp_path / "dem.tif"
    eos.dem.write_crop_to_file(dem.array, dem.transform, dem.crs, path)

    source = eos.dem.MyDEMSource(path)
    dem2 = source.fetch_dem(bounds)
    assert dem.elevation(lon, lat) == dem2.elevation(lon, lat)
    assert dem.get_extent() == dem2.get_extent()

    # check the `margin` parameter
    source = eos.dem.MyDEMSource(path, margin=0.1)
    dem2 = source.fetch_dem(bounds)
    assert dem.elevation(lon, lat) == dem2.elevation(lon, lat)
    assert dem.get_extent() == dem2.get_extent()
    assert source.get_extent()[0] < dem.get_extent()[0]
    assert source.get_extent()[1] < dem.get_extent()[1]
    assert source.get_extent()[2] > dem.get_extent()[2]
    assert source.get_extent()[3] > dem.get_extent()[3]


@pytest.fixture(scope="module")
def small_dem() -> eos.dem.DEM:
    bounds = (3, 44, 3.1, 44.05)
    dem_source = eos.dem.DEMStitcherSource()
    dem = dem_source.fetch_dem(bounds)
    return dem


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_interp_btw_last_two_cols_rows(small_dem, interpolation: str):
    h, w = small_dem.array.shape
    # should be able to interpolate points that fall between last 2 cols and last 2 rows
    n_test = 10

    # sample some points for interpolation
    # avoid falling exactly on h - 1 and w - 1 for this test as it is a special case

    # first do it between last two columns
    x_coords = (
        np.linspace(0, 0.99, num=n_test) + w - 2
    )  # points in [w-2, w - 1) interval
    y_coords = np.linspace(0, h - 1.02, n_test)  # points between 0 and h - 1.02
    interpolated = small_dem.interpolate_array(
        x_coords, y_coords, interpolation=interpolation, raise_error=True
    )
    assert len(interpolated) == len(x_coords)
    assert interpolated.dtype == small_dem.array.dtype

    # Then do it between last two rows
    y_coords = (
        np.linspace(0, 0.99, num=n_test) + h - 2
    )  # points in [h-2, h - 1) interval
    x_coords = np.linspace(0, w - 1.02, n_test)  # points between 0 and w- 1.02
    interpolated = small_dem.interpolate_array(
        x_coords, y_coords, interpolation=interpolation, raise_error=True
    )

    assert len(interpolated) == len(x_coords)
    assert interpolated.dtype == small_dem.array.dtype


def test_bilinear_vs_avg_last_two_cols_rows(small_dem):
    h, w = small_dem.array.shape
    # First last two cols
    # sample x exactly middle of last two cols
    x_coords = np.full((h - 1,), w - 1.5, dtype=np.float64)
    # integer line coords
    y_coords = np.arange(h - 1, dtype=np.float64)
    interpolated = small_dem.interpolate_array(
        x_coords, y_coords, interpolation="bilinear", raise_error=True
    )
    np.testing.assert_allclose(
        interpolated, np.mean(small_dem.array[: h - 1, w - 2 : w], axis=1)
    )

    # Then last two lines
    # sample x exactly middle of last two cols
    y_coords = np.full((w - 1,), h - 1.5, dtype=np.float64)
    # integer line coords
    x_coords = np.arange(w - 1, dtype=np.float64)
    interpolated = small_dem.interpolate_array(
        x_coords, y_coords, interpolation="bilinear", raise_error=True
    )
    np.testing.assert_allclose(
        interpolated, np.mean(small_dem.array[h - 2 : h, : w - 1], axis=0)
    )


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_interp_pts_border(small_dem, interpolation: str):
    h, w = small_dem.array.shape

    # Points falling exactly on first or last column
    # Exactly in the middle between two lines

    # lines coords in the middle of all lines [0.5, 1.5, 2.5, ...]
    y_coords = np.arange(0.5, h - 1, 1, dtype=np.float64)
    for col, col_str in zip([0, w - 1], ["first col", "last col"]):
        x_coords = np.full((h - 1,), col, dtype=np.float64)
        interpolated = small_dem.interpolate_array(
            x_coords, y_coords, interpolation=interpolation, raise_error=True
        )

        col_dem = small_dem.array[:, col]
        if interpolation == "bilinear":
            gt = (col_dem[:-1] + col_dem[1:]) / 2
        if interpolation == "nearest":
            gt = np.zeros_like(interpolated)
            # np.round for nearest neighbors will always choose the even index for .5 floats, i.e
            # 1.5 rounds to 2 and 2.5 rounds to 2 as well
            gt[::2] = col_dem[:-1:2]
            gt[1::2] = col_dem[2::2]
        np.testing.assert_allclose(
            interpolated, gt, atol=1e-3, rtol=1e-5, err_msg=f"Error for {col_str}"
        )

    # Points falling exactly on first or last line
    # Exactly in the middle between two columns

    interpolation = "nearest"
    # column coords in the middle of all columns [0.5, 1.5, 2.5, ...]
    x_coords = np.arange(0.5, w - 1, 1, dtype=np.float64)
    for row, row_str in zip([0, h - 1], ["first row", "last row"]):
        y_coords = np.full((w - 1,), row, dtype=np.float64)
        interpolated = small_dem.interpolate_array(
            x_coords, y_coords, interpolation=interpolation, raise_error=True
        )

        row_dem = small_dem.array[row]
        if interpolation == "bilinear":
            gt = (row_dem[:-1] + row_dem[1:]) / 2

        if interpolation == "nearest":
            gt = np.zeros_like(interpolated)
            # np.round for nearest neighbors will always choose the even index for .5 floats, i.e
            # 1.5 rounds to 2 and 2.5 rounds to 2 as well
            gt[::2] = row_dem[:-1:2]
            gt[1::2] = row_dem[2::2]
        np.testing.assert_allclose(
            interpolated, gt, atol=1e-3, rtol=1e-5, err_msg=f"Error for {row_str}"
        )


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
@pytest.mark.parametrize("raise_error", [True, False])
def test_interp_with_exceeding_dem_limits(
    small_dem, interpolation: str, raise_error: bool
):
    h, w = small_dem.array.shape

    eps = 0.01
    margin = 5
    # check on the right

    # integer coords + eps, to check that as soon as we exceed w - 1 by eps, we get error or nan
    x_coords = np.arange(w - margin + eps, w + margin + 2 * eps, 1)
    y_coords = np.full_like(x_coords, h // 2)
    if raise_error:
        with pytest.raises(eos.dem.OutOfBoundsException):
            interpolated = small_dem.interpolate_array(
                x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
            )
    else:
        interpolated = small_dem.interpolate_array(
            x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
        )

        # margin - 1 is the cutoff where everything becomes nan (outside of raster)
        assert not np.any(np.isnan(interpolated[: margin - 1]))
        assert np.all(np.isnan(interpolated[margin - 1 :]))

    # check on the left
    # integer coords - eps, to check that as soon as we hit -eps, we get error or nan
    x_coords = np.arange(-margin - eps, margin, 1)
    y_coords = np.full_like(x_coords, h // 2)
    if raise_error:
        with pytest.raises(eos.dem.OutOfBoundsException):
            interpolated = small_dem.interpolate_array(
                x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
            )
    else:
        interpolated = small_dem.interpolate_array(
            x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
        )

        # margin + 1 is the cutoff where everything becomes nan (outside of raster)
        assert np.all(np.isnan(interpolated[: margin + 1]))
        assert not np.any(np.isnan(interpolated[margin + 1 :]))

    # Repeat to check with vertical lines now (top and bottom), by inverting role of x and y

    # Check on the bottom
    # integer coords + eps, to check that as soon as we exceed h - 1 by eps, we get error or nan
    y_coords = np.arange(h - margin + eps, h + margin + 2 * eps, 1)
    x_coords = np.full_like(y_coords, w // 2)

    if raise_error:
        with pytest.raises(eos.dem.OutOfBoundsException):
            interpolated = small_dem.interpolate_array(
                x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
            )
    else:
        interpolated = small_dem.interpolate_array(
            x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
        )

        # margin - 1 is the cutoff where everything becomes nan (outside of raster)
        assert not np.any(np.isnan(interpolated[: margin - 1]))
        assert np.all(np.isnan(interpolated[margin - 1 :]))

    # check on the top
    # integer coords - eps, to check that as soon as we hit -eps, we get error or nan
    y_coords = np.arange(-margin - eps, margin, 1)
    x_coords = np.full_like(y_coords, w // 2)

    if raise_error:
        with pytest.raises(eos.dem.OutOfBoundsException):
            interpolated = small_dem.interpolate_array(
                x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
            )
    else:
        interpolated = small_dem.interpolate_array(
            x_coords, y_coords, interpolation=interpolation, raise_error=raise_error
        )

        # margin + 1 is the cutoff where everything becomes nan (outside of raster)
        assert np.all(np.isnan(interpolated[: margin + 1]))
        assert not np.any(np.isnan(interpolated[margin + 1 :]))
