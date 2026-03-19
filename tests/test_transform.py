import unittest
import numpy as np

from fem2geo.utils.transform import (
    unit, enu2ned, ned2enu,
    line_sphe2ned, line_ned2sphe, line_sphe2enu, line_enu2sphe,
    line_rake2sphe, line_enu2rake,
    plane_basis_enu,
    slip_rake2enu, slip_enu2rake,
    plane_sphe2enu, plane_sphe2ned, plane_pole2sphe,
)


class TestBasicFunctions(unittest.TestCase):
    """unit, enu2ned, ned2enu."""

    # unit

    def test_unit_already_normalized(self):
        np.testing.assert_allclose(unit([1, 0, 0]), [1, 0, 0])

    def test_unit_scales(self):
        v = unit([3, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=14)

    def test_unit_arbitrary(self):
        v = unit([1, 2, 3])
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=14)

    def test_unit_zero_raises(self):
        with self.assertRaises(ValueError):
            unit([0, 0, 0])

    # enu2ned

    def test_enu2ned_basis(self):
        np.testing.assert_allclose(enu2ned([1, 0, 0]), [0, 1, 0])   # E -> NED_E
        np.testing.assert_allclose(enu2ned([0, 1, 0]), [1, 0, 0])   # N -> NED_N
        np.testing.assert_allclose(enu2ned([0, 0, 1]), [0, 0, -1])  # U -> -D

    def test_enu2ned_arbitrary(self):
        np.testing.assert_allclose(enu2ned([3, 5, 7]), [5, 3, -7])

    def test_enu2ned_vectorized(self):
        vv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        np.testing.assert_allclose(enu2ned(vv), expected)

    # ned2enu

    def test_ned2enu_basis(self):
        np.testing.assert_allclose(ned2enu([1, 0, 0]), [0, 1, 0])   # N -> ENU_N
        np.testing.assert_allclose(ned2enu([0, 1, 0]), [1, 0, 0])   # E -> ENU_E
        np.testing.assert_allclose(ned2enu([0, 0, 1]), [0, 0, -1])  # D -> -U

    def test_ned2enu_vectorized(self):
        vv = np.array([[5, 3, -7], [1, 0, 0]], dtype=float)
        expected = np.array([[3, 5, 7], [0, 1, 0]], dtype=float)
        np.testing.assert_allclose(ned2enu(vv), expected)

    # roundtrips

    def test_enu_ned_roundtrip(self):
        v = np.array([1.5, -2.3, 0.7])
        np.testing.assert_allclose(ned2enu(enu2ned(v)), v)
        np.testing.assert_allclose(enu2ned(ned2enu(v)), v)

    def test_roundtrip_batch(self):
        rng = np.random.default_rng(42)
        vv = rng.normal(size=(20, 3))
        np.testing.assert_allclose(ned2enu(enu2ned(vv)), vv, atol=1e-14)
        np.testing.assert_allclose(enu2ned(ned2enu(vv)), vv, atol=1e-14)


class TestLines(unittest.TestCase):
    """line_sphe2ned and upcoming line functions."""

    def test_sphe2ned_cardinals(self):
        # plunge=0 azm=0: horizontal North -> [1,0,0]
        # plunge=0 azm=90: horizontal East -> [0,1,0]
        # plunge=90 azm=0: vertical down -> [0,0,1]
        np.testing.assert_allclose(line_sphe2ned(0, 0), [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(line_sphe2ned(0, 90), [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(line_sphe2ned(90, 0), [0, 0, 1], atol=1e-14)

    def test_sphe2ned_d_non_negative(self):
        # D component should always be >= 0 (axis convention)
        for pl, az in [(0, 0), (30, 45), (60, 135), (90, 0), (0, 270)]:
            ned = line_sphe2ned(pl, az)
            self.assertGreaterEqual(ned[2], -1e-14)
            self.assertAlmostEqual(np.linalg.norm(ned), 1.0, places=14)

    def test_sphe2ned_vectorized(self):
        plunges = np.array([0, 30, 60, 90], dtype=float)
        azimuths = np.array([0, 45, 135, 0], dtype=float)
        ned = line_sphe2ned(plunges, azimuths)
        self.assertEqual(ned.shape, (4, 3))
        self.assertTrue(np.all(ned[:, 2] >= -1e-14))
        norms = np.linalg.norm(ned, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    # line_ned2sphe

    def test_ned2sphe_cardinals(self):
        p, a = line_ned2sphe([1, 0, 0])  # North
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(a, 0.0)

        p, a = line_ned2sphe([0, 1, 0])  # East
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(a, 90.0)

        p, a = line_ned2sphe([0, 0, 1])  # Down
        self.assertAlmostEqual(p, 90.0)
        self.assertAlmostEqual(a, 0.0)  # vertical: azimuth = 0 by convention

    def test_ned2sphe_axis_symmetry(self):
        # v and -v should give the same plunge/azimuth
        p1, a1 = line_ned2sphe([0.2, 0.1, 0.3])
        p2, a2 = line_ned2sphe([-0.2, -0.1, -0.3])
        self.assertAlmostEqual(p1, p2, places=10)
        self.assertAlmostEqual((a1 - a2 + 180) % 360 - 180, 0.0, places=10)

    def test_ned2sphe_vectorized(self):
        ned = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        plunges, azimuths = line_ned2sphe(ned)
        np.testing.assert_allclose(plunges, [0, 0, 90], atol=1e-10)
        np.testing.assert_allclose(azimuths, [0, 90, 0], atol=1e-10)

    # sphe2ned <-> ned2sphe roundtrip

    def test_sphe_ned_roundtrip(self):
        cases = [(0, 0), (0, 90), (30, 45), (60, 135), (10, 359), (90, 0)]
        for pl, az in cases:
            ned = line_sphe2ned(pl, az)
            p2, a2 = line_ned2sphe(ned)
            self.assertAlmostEqual(p2, pl, places=9)
            self.assertAlmostEqual((a2 - az + 180) % 360 - 180, 0.0, places=9)

    # line_sphe2enu

    def test_sphe2enu_cardinals(self):
        # plunge=0 azm=0: North -> ENU [0,1,0]
        # plunge=0 azm=90: East -> ENU [1,0,0]
        # plunge=0 azm=180: South -> ENU [0,-1,0]
        # plunge=0 azm=270: West -> ENU [-1,0,0]
        np.testing.assert_allclose(line_sphe2enu(0, 0), [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(line_sphe2enu(0, 90), [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(line_sphe2enu(0, 180), [0, -1, 0], atol=1e-14)
        np.testing.assert_allclose(line_sphe2enu(0, 270), [-1, 0, 0], atol=1e-14)

    # line_enu2sphe

    def test_enu2sphe_axis_symmetry(self):
        # v and -v give the same result (axis convention)
        rng = np.random.default_rng(42)
        for v in rng.normal(size=(10, 3)):
            if np.linalg.norm(v) < 1e-8:
                continue
            p1, a1 = line_enu2sphe(v)
            p2, a2 = line_enu2sphe(-v)
            self.assertAlmostEqual(p1, p2, places=9)
            self.assertAlmostEqual((a1 - a2 + 180) % 360 - 180, 0.0, places=9)

    def test_enu_sphe_roundtrip(self):
        # enu -> sphe -> enu should give colinear vector
        rng = np.random.default_rng(42)
        for v in rng.normal(size=(10, 3)):
            if np.linalg.norm(v) < 1e-8:
                continue
            p, a = line_enu2sphe(v)
            v2 = line_sphe2enu(p, a)
            # colinear check: v and v2 are parallel or antiparallel
            v_n = v / np.linalg.norm(v)
            dot = abs(np.dot(v_n, v2))
            self.assertAlmostEqual(dot, 1.0, places=9)

    def test_enu2sphe_vectorized(self):
        rng = np.random.default_rng(99)
        vv = rng.normal(size=(10, 3))
        vv = vv[np.linalg.norm(vv, axis=1) > 1e-8]
        plunges, azimuths = line_enu2sphe(vv)
        self.assertEqual(plunges.shape[0], vv.shape[0])
        enu2 = line_sphe2enu(plunges, azimuths)
        for i in range(len(vv)):
            v_n = vv[i] / np.linalg.norm(vv[i])
            dot = abs(np.dot(v_n, enu2[i]))
            self.assertAlmostEqual(dot, 1.0, places=9)

    # line_rake2sphe

    def test_rake2sphe_endmembers(self):
        # rake=0: horizontal along strike
        p, a = line_rake2sphe(30, 60, 0)
        self.assertAlmostEqual(p, 0.0, places=9)
        self.assertAlmostEqual((a - 30 + 180) % 360 - 180, 0.0, places=9)
        # rake=90: pure dip-slip, plunge = dip
        p, a = line_rake2sphe(30, 60, 90)
        self.assertAlmostEqual(p, 60.0, places=9)

    def test_rake2sphe_vectorized(self):
        strikes = np.array([30, 30, 30], dtype=float)
        dips = np.array([60, 60, 60], dtype=float)
        rakes = np.array([0, 45, 90], dtype=float)
        plunges, azimuths = line_rake2sphe(strikes, dips, rakes)
        self.assertEqual(plunges.shape, (3,))

    # line_enu2rake

    def test_enu2rake_roundtrip(self):
        # use slip_rake2enu (directed) to get a vector, test both v and -v
        for strike, dip in [(30, 60), (230, 55), (0, 90)]:
            for rake in [0, 15, 45, 90, 120, 150, 180]:
                enu = slip_rake2enu(strike, dip, rake)
                r1 = line_enu2rake(enu, strike, dip)
                r2 = line_enu2rake(-enu, strike, dip)
                ok1 = abs(r1 - rake) < 1e-6 or abs(r1 - (180 - rake)) < 1e-6
                ok2 = abs(r2 - rake) < 1e-6 or abs(r2 - (180 - rake)) < 1e-6
                self.assertTrue(ok1, msg=f"s={strike} d={dip} r={rake}: got {r1}")
                self.assertTrue(ok2, msg=f"s={strike} d={dip} r={rake}: -enu got {r2}")

    def test_enu2rake_containment_reject(self):
        with self.assertRaises(ValueError):
            line_enu2rake([0, 0, 1], 10, 45, check=True, tol=1e-6)


class TestSlips(unittest.TestCase):
    """slip_rake2enu, slip_enu2rake."""

    def test_rake90_is_updip(self):
        for strike in [0, 45, 230]:
            _, _, updip = plane_basis_enu(strike, 60)
            slip = slip_rake2enu(strike, 60, 90)
            np.testing.assert_allclose(slip, updip, atol=1e-12)

    def test_rake_neg90_is_downdip(self):
        _, downdip, _ = plane_basis_enu(230, 60)
        slip = slip_rake2enu(230, 60, -90)
        np.testing.assert_allclose(slip, downdip, atol=1e-12)

    def test_rake0_is_strike(self):
        s_dir, _, _ = plane_basis_enu(45, 60)
        slip = slip_rake2enu(45, 60, 0)
        np.testing.assert_allclose(slip, s_dir, atol=1e-12)

    def test_z_sign(self):
        # positive rake -> upward Z (reverse)
        self.assertGreater(slip_rake2enu(230, 60, 90)[2], 0)
        # negative rake -> downward Z (normal)
        self.assertLess(slip_rake2enu(230, 60, -90)[2], 0)

    def test_enu2rake_roundtrip(self):
        for strike in [0, 45, 120, 230, 315]:
            for rake in [-170, -90, -45, 0, 45, 90, 135, 180]:
                enu = slip_rake2enu(strike, 60, rake)
                r_out = slip_enu2rake(enu, strike, 60)
                d = (r_out - rake + 180) % 360 - 180
                self.assertAlmostEqual(d, 0.0, places=7,
                                       msg=f"s={strike} r={rake}: got {r_out}")

    def test_enu2rake_sign_preserved(self):
        self.assertGreater(slip_enu2rake(slip_rake2enu(230, 60, 90), 230, 60), 0)
        self.assertLess(slip_enu2rake(slip_rake2enu(230, 60, -90), 230, 60), 0)

    def test_vectorized(self):
        rakes = np.array([90, -45, 0, 135, -90], dtype=float)
        enus = slip_rake2enu(230, 60, rakes)
        self.assertEqual(enus.shape, (5, 3))
        rakes_out = slip_enu2rake(enus, 230, 60)
        for i in range(len(rakes)):
            d = (rakes_out[i] - rakes[i] + 180) % 360 - 180
            self.assertAlmostEqual(d, 0.0, places=7)


class TestPlanes(unittest.TestCase):
    """plane_basis_enu, plane_sphe2enu, plane_sphe2ned, plane_pole2sphe."""

    # plane_basis_enu

    def test_basis_orthogonality(self):
        for strike in [0, 45, 90, 180, 230, 315]:
            s_dir, dd_dir, ud_dir = plane_basis_enu(strike, 60)
            self.assertAlmostEqual(np.dot(s_dir, dd_dir), 0.0, places=12)
            self.assertAlmostEqual(np.linalg.norm(s_dir), 1.0, places=14)
            self.assertAlmostEqual(np.linalg.norm(dd_dir), 1.0, places=14)
            np.testing.assert_allclose(ud_dir, -dd_dir, atol=1e-14)

    def test_basis_downdip_z_negative(self):
        for strike in [0, 45, 120, 230, 315]:
            _, dd, ud = plane_basis_enu(strike, 60)
            self.assertLess(dd[2], 0)
            self.assertGreater(ud[2], 0)

    def test_basis_vectorized(self):
        strikes = np.array([0, 45, 230], dtype=float)
        dips = np.array([60, 60, 60], dtype=float)
        s_dir, dd_dir, ud_dir = plane_basis_enu(strikes, dips)
        self.assertEqual(s_dir.shape, (3, 3))
        for i in range(3):
            self.assertAlmostEqual(np.dot(s_dir[i], dd_dir[i]), 0.0, places=12)

    # plane_sphe2enu

    def test_sphe2enu_horizontal(self):
        # dip=0: horizontal plane, pole is vertical [0,0,1] in ENU
        for strike in [0, 37, 120, 359]:
            n = plane_sphe2enu(strike, 0)
            dot = abs(np.dot(n, [0, 0, 1]))
            self.assertAlmostEqual(dot, 1.0, places=9)

    def test_sphe2enu_vertical_cardinals(self):
        # strike=0, dip=90: vertical N-S plane, normal ~ East
        n = plane_sphe2enu(0, 90)
        dot = abs(np.dot(n / np.linalg.norm(n), [1, 0, 0]))
        self.assertAlmostEqual(dot, 1.0, places=9)

    def test_sphe2enu_u_non_negative(self):
        for strike in [0, 45, 120, 230, 315]:
            for dip in [10, 30, 60, 80]:
                n = plane_sphe2enu(strike, dip)
                self.assertGreater(n[2], -1e-12)

    def test_sphe2enu_perpendicular_to_strike(self):
        for strike in [0, 45, 120, 230]:
            s_dir, _, _ = plane_basis_enu(strike, 60)
            n = plane_sphe2enu(strike, 60)
            self.assertAlmostEqual(np.dot(n, s_dir), 0.0, places=10)

    def test_sphe2enu_vectorized(self):
        strikes = np.array([0, 45, 120, 230], dtype=float)
        dips = np.array([30, 60, 45, 80], dtype=float)
        normals = plane_sphe2enu(strikes, dips)
        self.assertEqual(normals.shape, (4, 3))
        self.assertTrue(np.all(normals[:, 2] > -1e-12))

    # plane_sphe2ned

    def test_sphe2ned_d_non_negative(self):
        for strike in [0, 45, 120, 230, 315]:
            for dip in [10, 30, 60, 80]:
                n = plane_sphe2ned(strike, dip)
                self.assertGreater(n[2], -1e-12)

    # plane_pole2sphe

    def test_pole2sphe_basic(self):
        s, d = plane_pole2sphe(60, 180)
        self.assertAlmostEqual(s, 270.0)
        self.assertAlmostEqual(d, 30.0)

    def test_pole2sphe_vectorized(self):
        plunges = np.array([60, 30, 0], dtype=float)
        azimuths = np.array([180, 90, 0], dtype=float)
        strikes, dips = plane_pole2sphe(plunges, azimuths)
        np.testing.assert_allclose(dips, [30, 60, 90])

    # pole roundtrip: plane -> NED pole -> line_ned2sphe -> plane_pole2sphe

    def test_pole_roundtrip(self):
        for strike, dip in [(0, 10), (30, 45), (120, 60), (270, 80), (10, 90)]:
            n_ned = plane_sphe2ned(strike, dip)
            pl, az = line_ned2sphe(n_ned)
            s2, d2 = plane_pole2sphe(pl, az)
            self.assertAlmostEqual(d2, dip, places=8)
            ds = ((s2 - strike + 90) % 180) - 90  # strike mod 180
            self.assertAlmostEqual(ds, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()