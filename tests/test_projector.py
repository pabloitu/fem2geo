import unittest

import numpy as np
import pyvista as pv

from fem2geo.data import CatalogData
from fem2geo.projector import Projector
from fem2geo.utils.projections import (
    unit_factor, flip_z, reproject_xy, rotate_xy, to_lonlat, bbox_mask,
)


# unit_factor

class TestUnitFactor(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(unit_factor("m"), 1.0)
        self.assertEqual(unit_factor("km"), 1000.0)

    def test_case_and_whitespace(self):
        self.assertEqual(unit_factor("M"), 1.0)
        self.assertEqual(unit_factor(" Km "), 1000.0)

    def test_invalid(self):
        with self.assertRaisesRegex(ValueError, "must be 'm' or 'km'"):
            unit_factor("ft")


# flip_z

class TestFlipZ(unittest.TestCase):

    def test_same_convention(self):
        z = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(flip_z(z, "down", "down"), z)
        np.testing.assert_array_equal(flip_z(z, "up", "up"), z)

    def test_negates_when_different(self):
        np.testing.assert_array_equal(flip_z([1, 2, 3], "down", "up"), [-1, -2, -3])
        np.testing.assert_array_equal(flip_z([1, 2, 3], "up", "down"), [-1, -2, -3])

    def test_returns_float_array(self):
        out = flip_z([1, 2, 3], "down", "up")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, float)

    def test_invalid_src(self):
        with self.assertRaisesRegex(ValueError, "src_positive"):
            flip_z([1.0], "sideways", "up")

    def test_invalid_dst(self):
        with self.assertRaisesRegex(ValueError, "dst_positive"):
            flip_z([1.0], "up", "diagonal")


# reproject_xy

class TestReprojectXY(unittest.TestCase):

    def test_identity(self):
        X, Y = reproject_xy([1.0, 2.0], [3.0, 4.0], "epsg:32719", "epsg:32719")
        np.testing.assert_allclose(X, [1.0, 2.0])
        np.testing.assert_allclose(Y, [3.0, 4.0])

    def test_lonlat_to_utm19s(self):
        X, Y = reproject_xy([-71.07], [-20.09], "epsg:4326", "epsg:32719")
        self.assertAlmostEqual(X[0], 283554.0, delta=10.0)
        self.assertAlmostEqual(Y[0], 7777216.0, delta=10.0)

    def test_roundtrip(self):
        lon = np.array([-72.0, -70.5, -68.3])
        lat = np.array([-21.0, -20.0, -19.5])
        X, Y = reproject_xy(lon, lat, "epsg:4326", "epsg:32719")
        lon2, lat2 = reproject_xy(X, Y, "epsg:32719", "epsg:4326")
        np.testing.assert_allclose(lon2, lon, atol=1e-7)
        np.testing.assert_allclose(lat2, lat, atol=1e-7)

    def test_returns_ndarray(self):
        X, Y = reproject_xy([1.0], [2.0], "epsg:32719", "epsg:32719")
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(Y, np.ndarray)


# rotate_xy

class TestRotateXY(unittest.TestCase):

    def test_zero_degrees(self):
        x, y = rotate_xy([1.0, 2.0], [3.0, 4.0], 0.0, 0.0, 0.0)
        np.testing.assert_allclose(x, [1.0, 2.0])
        np.testing.assert_allclose(y, [3.0, 4.0])

    def test_90_ccw_around_origin(self):
        x, y = rotate_xy([1.0, 0.0], [0.0, 1.0], 0.0, 0.0, 90.0)
        np.testing.assert_allclose(x, [0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(y, [1.0, 0.0], atol=1e-12)

    def test_pivot_is_fixed(self):
        x, y = rotate_xy([5.0], [7.0], 5.0, 7.0, 137.0)
        np.testing.assert_allclose(x, [5.0])
        np.testing.assert_allclose(y, [7.0])

    def test_around_offset_pivot(self):
        x, y = rotate_xy([2.0], [1.0], 1.0, 1.0, 90.0)
        np.testing.assert_allclose(x, [1.0], atol=1e-12)
        np.testing.assert_allclose(y, [2.0], atol=1e-12)

    def test_360_is_identity(self):
        x0 = np.array([3.2, -1.7])
        y0 = np.array([0.5, 4.4])
        x, y = rotate_xy(x0, y0, 0.0, 0.0, 360.0)
        np.testing.assert_allclose(x, x0, atol=1e-12)
        np.testing.assert_allclose(y, y0, atol=1e-12)


# to_lonlat

class TestToLonLat(unittest.TestCase):

    def test_geographic_passthrough(self):
        lon, lat = to_lonlat([-71.07, -70.0], [-20.09, -19.5], "epsg:4326", "deg")
        np.testing.assert_allclose(lon, [-71.07, -70.0])
        np.testing.assert_allclose(lat, [-20.09, -19.5])

    def test_geographic_returns_ndarray(self):
        lon, lat = to_lonlat([1.0], [2.0], "epsg:4326", "deg")
        self.assertIsInstance(lon, np.ndarray)
        self.assertIsInstance(lat, np.ndarray)

    def test_from_utm_meters(self):
        lon, lat = to_lonlat([283554.3], [7777215.8], "epsg:32719", "m")
        self.assertAlmostEqual(lon[0], -71.07, places=4)
        self.assertAlmostEqual(lat[0], -20.09, places=4)

    def test_from_utm_km(self):
        lon, lat = to_lonlat([283.5543], [7777.2158], "epsg:32719", "km")
        self.assertAlmostEqual(lon[0], -71.07, places=4)
        self.assertAlmostEqual(lat[0], -20.09, places=4)

    def test_invalid_units_for_projected(self):
        with self.assertRaisesRegex(ValueError, "must be 'm' or 'km'"):
            to_lonlat([0.0], [0.0], "epsg:32719", "deg")


# bbox_mask

class TestBBoxMask(unittest.TestCase):

    def test_no_ranges_keeps_all(self):
        m = bbox_mask([1, 2, 3], [4, 5, 6], [7, 8, 9])
        np.testing.assert_array_equal(m, [True, True, True])

    def test_lon_only(self):
        lon = np.array([-72, -70, -68, -65])
        m = bbox_mask(lon, [0, 0, 0, 0], [0, 0, 0, 0], lon_range=(-71, -67))
        np.testing.assert_array_equal(m, [False, True, True, False])

    def test_combined_ranges(self):
        lon = np.array([-72, -70, -68])
        lat = np.array([-21, -20, -19])
        d = np.array([5, 50, 100])
        m = bbox_mask(
            lon, lat, d,
            lon_range=(-71, -67), lat_range=(-21, -19), depth_range_km=(0, 60),
        )
        np.testing.assert_array_equal(m, [False, True, False])

    def test_inclusive_bounds(self):
        m = bbox_mask([0, 1, 2], [0, 0, 0], [0, 0, 0], lon_range=(0, 2))
        np.testing.assert_array_equal(m, [True, True, True])

    def test_empty_result(self):
        m = bbox_mask([1, 2, 3], [0, 0, 0], [0, 0, 0], lon_range=(10, 20))
        self.assertFalse(m.any())


# Projector helpers

def _geo_to_utm_km_anchored():
    return Projector(
        src_crs="epsg:4326", dst_crs="epsg:32719",
        src_xy_units="deg", dst_xy_units="km",
        src_z_units="km", dst_z_units="km",
        src_z_positive="down", dst_z_positive="up",
        anchor_geo=(-71.07, -20.09, 15.6),
        anchor_local=(0, 0, -21),
        rotation_deg=-10,
    )


def _utm_m_to_km():
    return Projector(
        src_crs="epsg:32719", dst_crs="epsg:32719",
        src_xy_units="m", dst_xy_units="km",
        src_z_units="m", dst_z_units="km",
        src_z_positive="up", dst_z_positive="up",
    )


# Projector constructor validation

class TestProjectorValidation(unittest.TestCase):

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
                      anchor_local=(0, 0, 0))


# Anchor math

class TestAnchorMath(unittest.TestCase):

    def test_anchor_lands_at_fem_origin(self):
        p = _geo_to_utm_km_anchored()
        X, Y, Z = p.transform([-71.07], [-20.09], [15.6])
        self.assertAlmostEqual(X[0], 0.0, places=9)
        self.assertAlmostEqual(Y[0], 0.0, places=9)
        self.assertAlmostEqual(Z[0], -21.0, places=9)

    def test_anchor_z_offset(self):
        p = _geo_to_utm_km_anchored()
        self.assertAlmostEqual(p.dz, -5.4, places=9)

    def test_no_anchor_no_offset(self):
        p = _utm_m_to_km()
        self.assertEqual((p.dx, p.dy, p.dz), (0.0, 0.0, 0.0))


# Unit conversion pipeline

class TestUnitConversion(unittest.TestCase):

    def test_meters_to_km_no_anchor(self):
        p = _utm_m_to_km()
        X, Y, Z = p.transform([283554.3], [7777215.8], [-1500.0])
        self.assertAlmostEqual(X[0], 283.5543, places=4)
        self.assertAlmostEqual(Y[0], 7777.2158, places=4)
        self.assertAlmostEqual(Z[0], -1.5, places=9)

    def test_z_sign_flip(self):
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


# Round-trip via manual inverse

class TestRoundTrip(unittest.TestCase):

    def test_array_roundtrip(self):
        fwd = _geo_to_utm_km_anchored()
        lon = np.array([-71.07, -72.0, -70.5])
        lat = np.array([-20.09, -20.5, -19.8])
        dep = np.array([15.6, 30.0, 5.0])
        X, Y, Z = fwd.transform(lon, lat, dep)

        # manual inverse: undo rotation, undo offset, back-project
        x0, y0, _ = fwd.anchor_local
        Xu, Yu = rotate_xy(X, Y, x0, y0, -fwd.rotation_deg)
        Xu = Xu - fwd.dx
        Yu = Yu - fwd.dy
        Zu = Z - fwd.dz
        lon2, lat2 = reproject_xy(Xu * 1000.0, Yu * 1000.0,
                                  "epsg:32719", "epsg:4326")
        dep2 = -Zu

        np.testing.assert_allclose(lon2, lon, atol=1e-7)
        np.testing.assert_allclose(lat2, lat, atol=1e-7)
        np.testing.assert_allclose(dep2, dep, atol=1e-9)


# transform_catalog

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


# transform_mesh

class TestTransformMesh(unittest.TestCase):

    def _utm_mesh_meters(self):
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
        out = p.transform_mesh(self._utm_mesh_meters())
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
        out = p.transform_mesh(self._utm_mesh_meters())
        np.testing.assert_array_equal(
            out.point_data["depth_m"], [1500, 1500, 2000]
        )

    def test_anchor_lands_at_origin(self):
        p = Projector(
            src_crs="epsg:32719", dst_crs="epsg:32719",
            src_xy_units="m", dst_xy_units="km",
            src_z_units="m", dst_z_units="km",
            src_z_positive="up", dst_z_positive="up",
            anchor_geo=(-71.07, -20.09, 1.5),
            anchor_local=(0, 0, -8),
        )
        out = p.transform_mesh(self._utm_mesh_meters())
        self.assertAlmostEqual(out.points[0, 0], 0.0, places=4)
        self.assertAlmostEqual(out.points[0, 1], 0.0, places=4)
        self.assertAlmostEqual(out.points[0, 2], -8.0, places=9)

    def test_requires_matching_xy_and_z_units(self):
        p = Projector(
            src_crs="epsg:32719", dst_crs="epsg:32719",
            src_xy_units="m", dst_xy_units="km",
            src_z_units="km", dst_z_units="km",
            src_z_positive="up", dst_z_positive="up",
        )
        with self.assertRaisesRegex(ValueError, "xy_units == src_z_units"):
            p.transform_mesh(self._utm_mesh_meters())


if __name__ == "__main__":
    unittest.main()