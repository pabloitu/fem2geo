import numpy as np
import pytest

from fem2geo.utils.projections import (
    unit_factor,
    flip_z,
    reproject_xy,
    rotate_xy,
    to_lonlat,
    bbox_mask,
)


# unit_factor

def test_unit_factor_basic():
    assert unit_factor("m") == 1.0
    assert unit_factor("km") == 1000.0


def test_unit_factor_case_and_whitespace():
    assert unit_factor("M") == 1.0
    assert unit_factor(" Km ") == 1000.0


def test_unit_factor_invalid():
    with pytest.raises(ValueError, match="must be 'm' or 'km'"):
        unit_factor("ft")


# flip_z

def test_flip_z_same_convention():
    z = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(flip_z(z, "down", "down"), z)
    np.testing.assert_array_equal(flip_z(z, "up", "up"), z)


def test_flip_z_negates_when_different():
    np.testing.assert_array_equal(flip_z([1, 2, 3], "down", "up"), [-1, -2, -3])
    np.testing.assert_array_equal(flip_z([1, 2, 3], "up", "down"), [-1, -2, -3])


def test_flip_z_returns_float_array():
    out = flip_z([1, 2, 3], "down", "up")
    assert isinstance(out, np.ndarray)
    assert out.dtype == float


def test_flip_z_invalid_convention():
    with pytest.raises(ValueError, match="src_positive"):
        flip_z([1.0], "sideways", "up")
    with pytest.raises(ValueError, match="dst_positive"):
        flip_z([1.0], "up", "diagonal")


# reproject_xy

def test_reproject_identity():
    X, Y = reproject_xy([1.0, 2.0], [3.0, 4.0], "epsg:32719", "epsg:32719")
    np.testing.assert_allclose(X, [1.0, 2.0])
    np.testing.assert_allclose(Y, [3.0, 4.0])


def test_reproject_lonlat_to_utm19s():
    X, Y = reproject_xy([-71.07], [-20.09], "epsg:4326", "epsg:32719")
    assert X[0] == pytest.approx(283554.0, abs=10.0)
    assert Y[0] == pytest.approx(7777216.0, abs=10.0)


def test_reproject_roundtrip():
    lon, lat = np.array([-72.0, -70.5, -68.3]), np.array([-21.0, -20.0, -19.5])
    X, Y = reproject_xy(lon, lat, "epsg:4326", "epsg:32719")
    lon2, lat2 = reproject_xy(X, Y, "epsg:32719", "epsg:4326")
    np.testing.assert_allclose(lon2, lon, atol=1e-7)
    np.testing.assert_allclose(lat2, lat, atol=1e-7)


def test_reproject_returns_ndarray():
    X, Y = reproject_xy([1.0], [2.0], "epsg:32719", "epsg:32719")
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)


# rotate_xy

def test_rotate_zero_degrees():
    x, y = rotate_xy([1.0, 2.0], [3.0, 4.0], 0.0, 0.0, 0.0)
    np.testing.assert_allclose(x, [1.0, 2.0])
    np.testing.assert_allclose(y, [3.0, 4.0])


def test_rotate_90_ccw_around_origin():
    x, y = rotate_xy([1.0, 0.0], [0.0, 1.0], 0.0, 0.0, 90.0)
    np.testing.assert_allclose(x, [0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(y, [1.0, 0.0], atol=1e-12)


def test_rotate_pivot_is_fixed():
    x, y = rotate_xy([5.0], [7.0], 5.0, 7.0, 137.0)
    np.testing.assert_allclose(x, [5.0])
    np.testing.assert_allclose(y, [7.0])


def test_rotate_around_offset_pivot():
    x, y = rotate_xy([2.0], [1.0], 1.0, 1.0, 90.0)
    np.testing.assert_allclose(x, [1.0], atol=1e-12)
    np.testing.assert_allclose(y, [2.0], atol=1e-12)


def test_rotate_360_is_identity():
    x0, y0 = np.array([3.2, -1.7]), np.array([0.5, 4.4])
    x, y = rotate_xy(x0, y0, 0.0, 0.0, 360.0)
    np.testing.assert_allclose(x, x0, atol=1e-12)
    np.testing.assert_allclose(y, y0, atol=1e-12)


# to_lonlat

def test_to_lonlat_geographic_passthrough():
    lon, lat = to_lonlat([-71.07, -70.0], [-20.09, -19.5], "epsg:4326", "deg")
    np.testing.assert_allclose(lon, [-71.07, -70.0])
    np.testing.assert_allclose(lat, [-20.09, -19.5])


def test_to_lonlat_geographic_returns_ndarray():
    lon, lat = to_lonlat([1.0], [2.0], "epsg:4326", "deg")
    assert isinstance(lon, np.ndarray)
    assert isinstance(lat, np.ndarray)


def test_to_lonlat_from_utm_meters():
    lon, lat = to_lonlat([283554.3], [7777215.8], "epsg:32719", "m")
    assert lon[0] == pytest.approx(-71.07, abs=1e-4)
    assert lat[0] == pytest.approx(-20.09, abs=1e-4)


def test_to_lonlat_from_utm_km():
    lon, lat = to_lonlat([283.5543], [7777.2158], "epsg:32719", "km")
    assert lon[0] == pytest.approx(-71.07, abs=1e-4)
    assert lat[0] == pytest.approx(-20.09, abs=1e-4)


def test_to_lonlat_invalid_units_for_projected():
    with pytest.raises(ValueError, match="must be 'm' or 'km'"):
        to_lonlat([0.0], [0.0], "epsg:32719", "deg")


# bbox_mask

def test_bbox_mask_no_ranges_keeps_all():
    m = bbox_mask([1, 2, 3], [4, 5, 6], [7, 8, 9])
    np.testing.assert_array_equal(m, [True, True, True])


def test_bbox_mask_lon_only():
    lon = np.array([-72, -70, -68, -65])
    m = bbox_mask(lon, [0, 0, 0, 0], [0, 0, 0, 0], lon_range=(-71, -67))
    np.testing.assert_array_equal(m, [False, True, True, False])


def test_bbox_mask_combined_ranges():
    lon = np.array([-72, -70, -68])
    lat = np.array([-21, -20, -19])
    d = np.array([5, 50, 100])
    m = bbox_mask(
        lon, lat, d,
        lon_range=(-71, -67), lat_range=(-21, -19), depth_range_km=(0, 60),
    )
    np.testing.assert_array_equal(m, [False, True, False])


def test_bbox_mask_inclusive_bounds():
    m = bbox_mask([0, 1, 2], [0, 0, 0], [0, 0, 0], lon_range=(0, 2))
    np.testing.assert_array_equal(m, [True, True, True])


def test_bbox_mask_empty_result():
    m = bbox_mask([1, 2, 3], [0, 0, 0], [0, 0, 0], lon_range=(10, 20))
    assert not m.any()