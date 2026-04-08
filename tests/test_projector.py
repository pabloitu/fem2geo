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
import unittest

import numpy as np
import pyvista as pv

from fem2geo.data import CatalogData
from fem2geo.projector import Projector


# common configs

def _geo_to_utm_km_anchored():
    return Projector(
        src_crs="epsg:4326", dst_crs="epsg:32719",
        src_xy_units="deg", dst_xy_units="km",
        src_z_units="km", dst_z_units="km",
        src_z_positive="down", dst_z_positive="up",
        anchor_geo=(-71.07, -20.09, 15.6),
        anchor_fem=(0, 0, -21),
        rotation_deg=-10,
    )


def _utm_m_to_km():
    return Projector(
        src_crs="epsg:32719", dst_crs="epsg:32719",
        src_xy_units="m", dst_xy_units="km",
        src_z_units="m", dst_z_units="km",
        src_z_positive="up", dst_z_positive="up",
    )


class TestConstructorValidation(unittest.TestCase):

    def test_geographic_source_requires_deg(self):
        with self.assertRaisesRegex(ValueError, "must be 'deg'"):
            Projector("epsg:4326", "epsg:32719", src_xy_units="m")

    def test_projected_source_rejects_deg(self):
        with self.assertRaisesRegex(ValueError, "must be 'm' or 'km'"):
            Projector("epsg:32719", "epsg:32719",
                      src_xy_units="deg", dst_xy_units="m")

    def test_dst_xy_units_must_be_m_or_km(self):
        with self.assertRaisesRegex(ValueError, "dst_xy_units"):
            Projector("epsg:4326", "epsg:32719", dst_xy_units="deg")

    def test_invalid_z_unit(self):
        with self.assertRaisesRegex(ValueError, "must be 'm' or 'km'"):
            Projector("epsg:4326", "epsg:32719", src_z_units="ft")

    def test_invalid_z_positive(self):
        with self.assertRaisesRegex(ValueError, "src_z_positive"):
            Projector("epsg:4326", "epsg:32719", src_z_positive="sideways")

    def test_anchor_requires_both(self):
        with self.assertRaisesRegex(ValueError, "set together"):
            Projector("epsg:4326", "epsg:32719",
                      anchor_geo=(-71, -20, 15))

    def test_rotation_without_anchor(self):
        with self.assertRaisesRegex(ValueError, "rotation_deg requires"):
            Projector("epsg:4326", "epsg:32719", rotation_deg=10)

    def test_anchor_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "length 3"):
            Projector("epsg:4326", "epsg:32719",
                      anchor_geo=(-71, -20),
                      anchor_fem=(0, 0, 0))


class TestAnchorMath(unittest.TestCase):

    def test_anchor_lands_at_fem_origin(self):
        p = _geo_to_utm_km_anchored()
        X, Y, Z = p.transform([-71.07], [-20.09], [15.6])
        self.assertAlmostEqual(X[0], 0.0, places=9)
        self.assertAlmostEqual(Y[0], 0.0, places=9)
        self.assertAlmostEqual(Z[0], -21.0, places=9)

    def test_anchor_z_only_lands_correctly(self):
        # depth_km=15.6 down -> in km up that is -15.6, anchor offset = -21 - (-15.6) = -5.4
        p = _geo_to_utm_km_anchored()
        self.assertAlmostEqual(p.dz, -5.4, places=9)

    def test_no_anchor_no_offset(self):
        p = _utm_m_to_km()
        self.assertEqual((p.dx, p.dy, p.dz), (0.0, 0.0, 0.0))


class TestUnitConversion(unittest.TestCase):

    def test_meters_to_km_no_anchor(self):
        p = _utm_m_to_km()
        X, Y, Z = p.transform([283554.3], [7777215.8], [-1500.0])
        self.assertAlmostEqual(X[0], 283.5543, places=4)
        self.assertAlmostEqual(Y[0], 7777.2158, places=4)
        self.assertAlmostEqual(Z[0], -1.5, places=9)

    def test_z_sign_flip(self):
        # depth km down -> elevation km up
        p = Projector(
            src_crs="epsg:4326", dst_crs="epsg:32719",
            src_xy_units="deg", dst_xy_units="km",
            src_z_units="km", dst_z_units="km",
            src_z_positive="down", dst_z_positive="up",
        )
        _, _, Z = p.transform([-71.07], [-20.09], [10.0])
        self.assertAlmostEqual(Z[0], -10.0, places=9)

    def test_z_no_flip_when_same(self):
        p = _utm_m_to_km()
        _, _, Z = p.transform([0], [0], [500.0])
        self.assertAlmostEqual(Z[0], 0.5, places=9)


class TestRoundTrip(unittest.TestCase):

    def test_array_roundtrip_through_inverse(self):
        # Forward then build the inverse projector and check we recover input
        fwd = _geo_to_utm_km_anchored()
        lon = np.array([-71.07, -72.0, -70.5])
        lat = np.array([-20.09, -20.5, -19.8])
        dep = np.array([15.6, 30.0, 5.0])
        X, Y, Z = fwd.transform(lon, lat, dep)

        # Round-trip via inverse pipeline (manual, since Projector is one-way)
        # undo rotation
        from fem2geo.utils.projections import rotate_xy, reproject_xy
        x0, y0, _ = fwd.anchor_fem
        Xu, Yu = rotate_xy(X, Y, x0, y0, -fwd.rotation_deg)
        Xu = Xu - fwd.dx
        Yu = Yu - fwd.dy
        Zu = Z - fwd.dz
        # back to meters then lon/lat
        lon2, lat2 = reproject_xy(Xu * 1000.0, Yu * 1000.0,
                                  "epsg:32719", "epsg:4326")
        # z back to depth km down
        dep2 = -Zu

        np.testing.assert_allclose(lon2, lon, atol=1e-7)
        np.testing.assert_allclose(lat2, lat, atol=1e-7)
        np.testing.assert_allclose(dep2, dep, atol=1e-9)


class TestTransformCatalog(unittest.TestCase):

    def test_returns_new_catalog(self):
        p = _geo_to_utm_km_anchored()
        cat = CatalogData(
            x=[-71.07, -72.0], y=[-20.09, -20.5], z=[15.6, 30.0],
            attrs={"mag": [5.0, 4.5]},
        )
        out = p.transform_catalog(cat)
        self.assertIsInstance(out, CatalogData)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out.x[0], 0.0, places=9)
        self.assertAlmostEqual(out.y[0], 0.0, places=9)
        self.assertAlmostEqual(out.z[0], -21.0, places=9)

    def test_attrs_preserved(self):
        p = _geo_to_utm_km_anchored()
        cat = CatalogData(
            x=[-71.07], y=[-20.09], z=[15.6],
            attrs={"mag": [5.0], "rms": [0.3]},
        )
        out = p.transform_catalog(cat)
        np.testing.assert_array_equal(out.attrs["mag"], [5.0])
        np.testing.assert_array_equal(out.attrs["rms"], [0.3])

    def test_input_catalog_not_mutated(self):
        p = _geo_to_utm_km_anchored()
        cat = CatalogData(x=[-71.07], y=[-20.09], z=[15.6])
        original_x = cat.x.copy()
        p.transform_catalog(cat)
        np.testing.assert_array_equal(cat.x, original_x)

    def test_attrs_independent_after_transform(self):
        p = _geo_to_utm_km_anchored()
        cat = CatalogData(
            x=[-71.07], y=[-20.09], z=[15.6], attrs={"mag": [5.0]},
        )
        out = p.transform_catalog(cat)
        out.attrs["mag"][0] = 99.0
        self.assertEqual(cat.attrs["mag"][0], 5.0)


class TestTransformMesh(unittest.TestCase):

    def _utm_mesh_meters(self):
        # tiny PolyData around the anchor
        pts = np.array([
            [283554.3, 7777215.8, -1500.0],
            [283554.3 + 1000.0, 7777215.8, -1500.0],
            [283554.3, 7777215.8 + 1000.0, -2000.0],
        ])
        mesh = pv.PolyData(pts)
        mesh.point_data["depth_m"] = np.array([1500.0, 1500.0, 2000.0])
        return mesh

    def test_meters_to_km_no_anchor(self):
        p = _utm_m_to_km()
        mesh = self._utm_mesh_meters()
        out = p.transform_mesh(mesh)
        self.assertAlmostEqual(out.points[0, 0], 283.5543, places=4)
        self.assertAlmostEqual(out.points[0, 1], 7777.2158, places=4)
        self.assertAlmostEqual(out.points[0, 2], -1.5, places=9)

    def test_input_mesh_not_mutated(self):
        p = _utm_m_to_km()
        mesh = self._utm_mesh_meters()
        original = mesh.points.copy()
        p.transform_mesh(mesh)
        np.testing.assert_array_equal(mesh.points, original)

    def test_point_data_preserved(self):
        p = _utm_m_to_km()
        mesh = self._utm_mesh_meters()
        out = p.transform_mesh(mesh)
        np.testing.assert_array_equal(out.point_data["depth_m"], [1500, 1500, 2000])

    def test_anchor_lands_at_origin(self):
        p = Projector(
            src_crs="epsg:32719", dst_crs="epsg:32719",
            src_xy_units="m", dst_xy_units="km",
            src_z_units="m", dst_z_units="km",
            src_z_positive="up", dst_z_positive="up",
            anchor_geo=(-71.07, -20.09, 1.5),
            anchor_fem=(0, 0, -8),
        )
        mesh = self._utm_mesh_meters()
        out = p.transform_mesh(mesh)
        # first point IS the anchor (utm meters of (-71.07,-20.09), z=-1500m -> 1.5km depth)
        self.assertAlmostEqual(out.points[0, 0], 0.0, places=4)
        self.assertAlmostEqual(out.points[0, 1], 0.0, places=4)
        self.assertAlmostEqual(out.points[0, 2], -8.0, places=9)

    def test_mesh_units_rescale(self):
        # mesh in km, projector configured for m
        p = Projector(
            src_crs="epsg:32719", dst_crs="epsg:32719",
            src_xy_units="m", dst_xy_units="km",
            src_z_units="m", dst_z_units="km",
            src_z_positive="up", dst_z_positive="up",
        )
        pts_km = np.array([[283.5543, 7777.2158, -1.5]])
        mesh = pv.PolyData(pts_km)
        out = p.transform_mesh(mesh, mesh_units="km")
        self.assertAlmostEqual(out.points[0, 0], 283.5543, places=4)
        self.assertAlmostEqual(out.points[0, 1], 7777.2158, places=4)
        self.assertAlmostEqual(out.points[0, 2], -1.5, places=9)

    def test_invalid_mesh_units(self):
        p = _utm_m_to_km()
        mesh = self._utm_mesh_meters()
        with self.assertRaisesRegex(ValueError, "mesh_units"):
            p.transform_mesh(mesh, mesh_units="ft")




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