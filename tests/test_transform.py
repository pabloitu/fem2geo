import unittest
import numpy as np

from fem2geo.utils.transform import (
    enu_to_ned,
    ned_to_enu,
    line_sphe2enu,
    line_enu2sphe,
    line_sphe2ned,
    line_rake2sphe,
    lineplane2rake,
    plane_sphe2enu,
    plane_sphe2ned,
    plane_pole2sphe,
    line_ned2sphe,
)


class TestTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tol = 1e-10
        rng = np.random.default_rng(42)
        rand = rng.normal(size=(80, 3))
        cls.v = np.array([x for x in rand if np.linalg.norm(x) > 1e-8])

    def vclose(self, a, b, tol=None):
        tol = self.tol if tol is None else tol
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        self.assertTrue(np.allclose(a, b, atol=tol, rtol=0.0), msg=f"{a} != {b}")

    def aclose360(self, got, exp, places=10):
        g = float(got) % 360.0
        e = float(exp) % 360.0
        d = (g - e + 180.0) % 360.0 - 180.0
        self.assertEqual(0.0, round(abs(d), places), msg=f"{g} != {e} (mod 360)")

    def colinear(self, a, b, tol=None):
        tol = self.tol if tol is None else tol
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        self.assertTrue(
            (np.linalg.norm(a - b) <= tol) or (np.linalg.norm(a + b) <= tol),
            msg=f"not colinear (axis): a={a}, b={b}",
        )

    def test_basis(self):
        self.vclose(enu_to_ned([1, 0, 0]), [0, 1, 0])
        self.vclose(enu_to_ned([0, 1, 0]), [1, 0, 0])
        self.vclose(enu_to_ned([0, 0, 1]), [0, 0, -1])

        self.vclose(ned_to_enu([1, 0, 0]), [0, 1, 0])
        self.vclose(ned_to_enu([0, 1, 0]), [1, 0, 0])
        self.vclose(ned_to_enu([0, 0, 1]), [0, 0, -1])

        for x in self.v[:20]:
            self.vclose(ned_to_enu(enu_to_ned(x)), x)

    def test_lines(self):
        self.vclose(line_sphe2enu([0.0, 0.0]), [0.0, 1.0, 0.0])
        self.vclose(line_sphe2enu([0.0, 90.0]), [1.0, 0.0, 0.0])
        self.vclose(line_sphe2enu([0.0, 180.0]), [0.0, -1.0, 0.0])
        self.vclose(line_sphe2enu([0.0, 270.0]), [-1.0, 0.0, 0.0])

        s = line_ned2sphe([1.0, 0.0, 0.0])
        self.assertAlmostEqual(s[0], 0.0, places=10)
        self.aclose360(s[1], 0.0)

        s = line_ned2sphe([0.0, 1.0, 0.0])
        self.assertAlmostEqual(s[0], 0.0, places=10)
        self.aclose360(s[1], 90.0)

        s = line_ned2sphe([0.0, 0.0, 1.0])
        self.assertAlmostEqual(s[0], 90.0, places=10)
        self.aclose360(s[1], 0.0)

        s2 = line_ned2sphe([0.0, 0.0, -1.0])
        self.assertAlmostEqual(s2[0], 90.0, places=10)
        self.aclose360(s2[1], 0.0)

        s1 = line_ned2sphe([0.2, 0.1, 0.3])
        s2 = line_ned2sphe([-0.2, -0.1, -0.3])
        self.assertAlmostEqual(s1[0], s2[0], places=10)
        self.aclose360(s1[1], s2[1], places=10)

        for plunge, azm in [(0, 0), (0, 90), (30, 45), (60, 135), (10, 359)]:
            sphe = np.array([plunge, azm], dtype=float)
            sphe2 = line_ned2sphe(line_sphe2ned(sphe))
            self.assertAlmostEqual(sphe2[0], plunge, places=9)
            self.aclose360(sphe2[1], azm, places=9)

        for x in self.v[:20]:
            s1 = line_enu2sphe(x)
            s2 = line_enu2sphe(-x)
            self.assertAlmostEqual(s1[0], s2[0], places=9)
            self.aclose360(s1[1], s2[1], places=9)

            xr = line_sphe2enu(s1)
            self.colinear(x, xr, tol=1e-9)


class TestPlanes(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tol = 1e-10

    def vclose(self, a, b, tol=None):
        tol = self.tol if tol is None else tol
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        self.assertTrue(np.allclose(a, b, atol=tol, rtol=0.0), msg=f"{a} != {b}")

    def colinear(self, a, b, tol=None):
        tol = self.tol if tol is None else tol
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        self.assertTrue(
            (np.linalg.norm(a - b) <= tol) or (np.linalg.norm(a + b) <= tol),
            msg=f"not colinear: a={a}, b={b}",
        )

    def test_horizontal(self):
        # Any strike, dip=0 should give vertical normals:
        # ENU: Up; NED: Down
        for strike in [0.0, 37.0, 120.0, 359.0]:
            n_enu = plane_sphe2enu([strike, 0.0])
            n_ned = plane_sphe2ned([strike, 0.0])
            self.colinear(n_enu, [0.0, 0.0, 1.0], tol=1e-9)
            self.colinear(n_ned, [0.0, 0.0, 1.0], tol=1e-9)

    def test_vertical_cardinals(self):
        # Vertical planes: dip=90. Normal should be horizontal and perpendicular to strike.
        # strike=0 (N-S plane) -> normal ~ +/-East (ENU)
        n = plane_sphe2enu([0.0, 90.0])
        self.colinear(n, [1.0, 0.0, 0.0], tol=1e-9)

        # strike=90 (E-W plane) -> normal ~ +/-North (ENU)
        n = plane_sphe2enu([90.0, 90.0])
        self.colinear(n, [0.0, 1.0, 0.0], tol=1e-9)

    def test_pole_to_plane(self):
        # Consistency: plane -> pole(line sphe) -> plane
        #
        # We compute pole in NED, convert to line spherical, then back to plane.
        # For planes, strike is defined modulo 180 (because strike+180 is same plane).
        cases = [
            (0.0, 10.0),
            (30.0, 45.0),
            (120.0, 60.0),
            (270.0, 80.0),
            (10.0, 90.0),  # vertical
        ]

        for strike, dip in cases:
            pole_ned = plane_sphe2ned([strike, dip])
            pole_sphe = line_ned2sphe(pole_ned)
            plane2 = plane_pole2sphe(pole_sphe)

            print("\ncase:", (strike, dip))
            print("  pole_ned :", pole_ned)
            print("  pole_sphe:", pole_sphe)
            print("  plane2   :", plane2)

            # dip should be consistent (within numerical tolerance)
            self.assertAlmostEqual(float(plane2[1]), float(dip), places=8)

            # strike modulo 180 (plane ambiguity)
            s0 = float(strike) % 180.0
            s1 = float(plane2[0]) % 180.0
            d = (s1 - s0 + 90.0) % 180.0 - 90.0
            self.assertAlmostEqual(d, 0.0, places=8)


class TestRake(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tol = 1e-8

    def aclose(self, a, b, places=8):
        self.assertAlmostEqual(float(a), float(b), places=places)

    def test_roundtrip(self):
        strike, dip = 30.0, 60.0
        for rake in [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]:
            sphe = line_rake2sphe([strike, dip, rake])
            enu = line_sphe2enu(sphe)

            out = lineplane2rake(enu, [strike, dip])
            self.aclose(out[0], strike, places=10)
            self.aclose(out[1], dip, places=10)

            r = float(rake)
            r_out = float(out[2])
            self.assertTrue(
                abs(r_out - r) < 1e-7 or abs(r_out - (180.0 - r)) < 1e-7,
                msg=f"rake ambiguity: expected {r} or {180.0-r}, got {r_out}",
            )

            out2 = lineplane2rake(-enu, [strike, dip])
            r_out2 = float(out2[2])
            self.assertTrue(
                abs(r_out2 - r) < 1e-7 or abs(r_out2 - (180.0 - r)) < 1e-7,
                msg=f"rake ambiguity (-enu): expected {r} or {180.0-r}, got {r_out2}",
            )

    def test_containment_reject(self):
        strike, dip = 10.0, 45.0
        enu = np.array([0.0, 0.0, 1.0])
        with self.assertRaises(Exception):
            lineplane2rake(enu, [strike, dip], tol=1e-6)

    def test_endmembers(self):
        strike, dip = 123.0, 37.0
        for rake in [0.0, 90.0, 180.0]:
            sphe = line_rake2sphe([strike, dip, rake])
            enu = line_sphe2enu(sphe)
            out = lineplane2rake(enu, [strike, dip])

            r = float(rake)
            r_out = float(out[2])
            self.assertTrue(
                abs(r_out - r) < 1e-7 or abs(r_out - (180.0 - r)) < 1e-7,
                msg=f"rake ambiguity: expected {r} or {180.0-r}, got {r_out}",
            )



class TestStereo(unittest.TestCase):

    def a360(self, a, b, places=7):
        a = float(a) % 360.0
        b = float(b) % 360.0
        d = (a - b + 180.0) % 360.0 - 180.0
        self.assertEqual(0.0, round(abs(d), places), msg=f"{a} != {b} (mod 360)")

    def test_mpl(self):
        try:
            import mplstereonet.stereonet_math as sm
        except Exception:
            self.skipTest("mplstereonet not available")

        cases = [
            (30.0, 60.0, 0.0),
            (30.0, 60.0, 15.0),
            (30.0, 60.0, 30.0),
            (30.0, 60.0, 60.0),
            (30.0, 60.0, 90.0),
            (123.0, 37.0, 150.0),
            (123.0, 37.0, 180.0),
        ]

        for s, d, r in cases:
            p1, a1 = line_rake2sphe([s, d, r])

            lon, lat = sm.rake(s, d, r)
            p2, a2 = sm.geographic2plunge_bearing(lon, lat)
            p2 = np.asarray(p2).item()
            a2 = np.asarray(a2).item()

            self.assertAlmostEqual(float(p1), float(p2), places=7)
            self.a360(a1, a2, places=7)

    def test_inv(self):
        try:
            import mplstereonet.stereonet_math as sm
        except Exception:
            self.skipTest("mplstereonet not available")

        cases = [
            (30.0, 60.0, 0.0),
            (30.0, 60.0, 15.0),
            (30.0, 60.0, 30.0),
            (30.0, 60.0, 60.0),
            (30.0, 60.0, 90.0),
            (123.0, 37.0, 150.0),
            (123.0, 37.0, 180.0),
        ]

        for s, d, r in cases:
            lon, lat = sm.rake(s, d, r)
            pr, ar = sm.geographic2plunge_bearing(lon, lat)
            pr = np.asarray(pr).item()
            ar = np.asarray(ar).item()

            sp = line_rake2sphe([s, d, r])
            enu = line_sphe2enu(sp)
            out = lineplane2rake(enu, [s, d])

            rr = float(out[2])
            ok = (abs(rr - r) < 1e-7) or (abs(rr - (180.0 - r)) < 1e-7)
            self.assertTrue(ok, msg=f"expected {r} or {180.0-r}, got {rr}")

            self.assertAlmostEqual(float(sp[0]), float(pr), places=7)
            self.a360(sp[1], ar, places=7)


if __name__ == "__main__":
    unittest.main()
