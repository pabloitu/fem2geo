import unittest
import numpy as np

from fem2geo.transform import (
    enu_to_ned,
    ned_to_enu,
    line_sphe2enu,
    line_enu2sphe,
    line_sphe2ned,
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


if __name__ == "__main__":
    unittest.main()