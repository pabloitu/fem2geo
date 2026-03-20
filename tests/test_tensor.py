import unittest
import numpy as np

from fem2geo.utils import transform as tr
import fem2geo.utils.tensor as tm


class TestRotation(unittest.TestCase):

    def test_rotation_matrices_are_orthogonal(self):
        for ax in (1, 2, 3):
            for ang in (0, 17, 90, -45):
                R = tm.rot_matrix(ang, ax)
                np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_bad_axis(self):
        with self.assertRaises(ValueError):
            tm.rot_matrix(10, 4)

    def test_tensor_rotation_preserves_invariants(self):
        T = np.array([[3, .2, -.1], [.2, 2, .3], [-.1, .3, 1.0]])
        ev0 = np.sort(np.linalg.eigvalsh(T))
        for ax in (1, 2, 3):
            Tr = tm.rot_tensor(T, 67, ax)
            np.testing.assert_allclose(
                np.sort(np.linalg.eigvalsh(Tr)), ev0, atol=1e-9)
            self.assertAlmostEqual(np.trace(Tr), np.trace(T), places=9)


class TestResolvedShear(unittest.TestCase):

    def test_isotropic_gives_zero(self):
        S = 5.0 * np.eye(3)
        for p in ([0, 0], [30, 10], [120, 60], [270, 85]):
            tau, tau_hat = tm.resolved_shear_enu(S, plane=p)
            self.assertAlmostEqual(tau, 0.0, places=12)
            np.testing.assert_allclose(tau_hat, 0, atol=1e-12)


class TestTendencies(unittest.TestCase):

    def test_planes_vs_normals_agree(self):
        S = np.diag([1.0, 2.0, 3.0])
        P = np.array([[0, 10], [30, 60], [120, 45]], dtype=float)
        N = tr.plane_sphe2enu(P[:, 0], P[:, 1])

        np.testing.assert_allclose(
            tm.slip_tendency(S, planes=P),
            tm.slip_tendency(S, normals=N), atol=1e-9)
        np.testing.assert_allclose(
            tm.dilation_tendency(S, planes=P),
            tm.dilation_tendency(S, normals=N), atol=1e-9)

    def test_scalar_vs_batch(self):
        S = np.array([[4, .1, .2], [.1, 2, .3], [.2, .3, 1.0]])
        rng = np.random.default_rng(123)
        N = rng.normal(size=(25, 3))
        N /= np.linalg.norm(N, axis=1, keepdims=True)

        batch = tm.slip_tendency(S, normals=N)
        scalar = [tm.slip_tendency(S, normals=N[i]) for i in range(25)]
        np.testing.assert_allclose(batch, scalar, atol=1e-9)

    def test_slip_bounded_01(self):
        S = np.diag([-3.0, -1.0, -0.5])
        rng = np.random.default_rng(42)
        N = rng.normal(size=(100, 3))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        ts = tm.slip_tendency(S, normals=N)
        self.assertTrue(np.all(ts >= 0))
        self.assertTrue(np.all(ts <= 1.0 + 1e-12))
        # optimal plane should be near 1
        self.assertGreater(np.max(ts), 0.9)

    def test_slip_isotropic_zero(self):
        S = 5.0 * np.eye(3)
        ts = tm.slip_tendency(S, planes=[30, 45])
        self.assertAlmostEqual(ts, 0.0, places=10)

    def test_dilation_bounded_01(self):
        S = np.diag([-3.0, -1.0, -0.5])
        td = tm.dilation_tendency(S, planes=[30, 45])
        self.assertGreaterEqual(td, 0.0)
        self.assertLessEqual(td, 1.0 + 1e-12)

    def test_combined_is_sum(self):
        S = np.diag([-3.0, -1.0, -0.5])
        P = np.array([[0, 30], [90, 60]], dtype=float)
        ts = tm.slip_tendency(S, planes=P)
        td = tm.dilation_tendency(S, planes=P)
        tc = tm.combined_tendency(S, planes=P)
        np.testing.assert_allclose(tc, ts + td, atol=1e-12)

    def test_combined_bounded_02(self):
        S = np.diag([-3.0, -1.0, -0.5])
        rng = np.random.default_rng(7)
        N = rng.normal(size=(100, 3))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        tc = tm.combined_tendency(S, normals=N)
        self.assertTrue(np.all(tc >= 0))
        self.assertTrue(np.all(tc <= 2.0 + 1e-12))


class TestUnpackVoigt(unittest.TestCase):

    def test_diagonal_and_symmetry(self):
        packed = np.array([[1, 2, 3, .4, .5, .6]])
        t = tm.unpack_voigt6(packed)
        np.testing.assert_allclose(t[0], t[0].T)
        np.testing.assert_allclose(np.diag(t[0]), [1, 2, 3])
        self.assertAlmostEqual(t[0, 0, 1], 0.4)
        self.assertAlmostEqual(t[0, 1, 2], 0.5)
        self.assertAlmostEqual(t[0, 0, 2], 0.6)

    def test_eigenvalue_roundtrip(self):
        T = np.array([[3, .2, -.1], [.2, 2, .3], [-.1, .3, 1.0]])
        packed = np.array([[T[0,0], T[1,1], T[2,2], T[0,1], T[1,2], T[0,2]]])
        np.testing.assert_allclose(tm.unpack_voigt6(packed)[0], T, atol=1e-14)


class TestUnpackComponents(unittest.TestCase):

    def test_matches_voigt(self):
        packed = np.array([[1, 2, 3, .4, .5, .6]])
        arrays = dict(xx=[1], yy=[2], zz=[3], xy=[.4], yz=[.5], zx=[.6])
        np.testing.assert_allclose(
            tm.unpack_components(arrays), tm.unpack_voigt6(packed), atol=1e-14)


class TestKostrov(unittest.TestCase):

    def test_sinistral_vertical(self):
        K = tm.kostrov_tensor([0], [90], [0])
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        vals = np.linalg.eigvalsh(K)
        self.assertAlmostEqual(vals[0], -0.5, places=10)
        self.assertAlmostEqual(vals[2],  0.5, places=10)

    def test_always_symmetric(self):
        K = tm.kostrov_tensor([30, 120, 0], [60, 45, 90], [90, -45, 0])
        np.testing.assert_allclose(K, K.T, atol=1e-14)


if __name__ == "__main__":
    unittest.main()