import unittest
import numpy as np

from fem2geo import transform as tr
import fem2geo.tensor as tm


class TestTensor(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.atol = 1e-10
        cls.rng = np.random.default_rng(123)

    def mclose(self, A, B, atol=None):
        atol = self.atol if atol is None else atol
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        self.assertTrue(np.allclose(A, B, atol=atol, rtol=0.0), msg=f"\nA=\n{A}\nB=\n{B}\n")

    def aclose(self, a, b, places=10):
        self.assertAlmostEqual(float(a), float(b), places=places)

    def test_rot_matrix(self):
        I = np.eye(3)

        for ax in (1, 2, 3):
            for ang in (0.0, 17.0, 90.0, 123.0, -45.0):
                R = tm.rot_matrix(ang, ax)
                self.mclose(R.T @ R, I, atol=1e-12)
                self.aclose(np.linalg.det(R), 1.0, places=10)

        with self.assertRaises(ValueError):
            tm.rot_matrix(10.0, 4)

    def test_rot_tensor_invariants(self):
        # symmetric tensor with distinct eigenvalues
        T = np.array([[3.0, 0.2, -0.1],
                      [0.2, 2.0,  0.3],
                      [-0.1, 0.3, 1.0]], dtype=float)

        ev0 = np.sort(np.linalg.eigvalsh(T))
        tr0 = float(np.trace(T))
        fn0 = float(np.linalg.norm(T))

        for ax in (1, 2, 3):
            for ang in (13.0, 67.0, 123.0):
                Tr = tm.rot_tensor(T, ang, ax)
                ev1 = np.sort(np.linalg.eigvalsh(Tr))
                self.mclose(ev1, ev0, atol=1e-9)
                self.aclose(np.trace(Tr), tr0, places=9)
                self.aclose(np.linalg.norm(Tr), fn0, places=9)

    def test_resolved_shear_isotropic(self):
        # isotropic stress => traction parallel to normal => shear = 0
        S = 5.0 * np.eye(3)

        planes = [
            np.array([0.0, 0.0]),
            np.array([30.0, 10.0]),
            np.array([120.0, 60.0]),
            np.array([270.0, 85.0]),
        ]
        for p in planes:
            tau, tau_hat = tm.resolved_shear_enu(S, plane=p)
            self.aclose(tau, 0.0, places=12)
            self.mclose(tau_hat, [0.0, 0.0, 0.0], atol=1e-12)

        # also via normals
        n = np.array([1.0, 0.0, 0.0])
        tau, tau_hat = tm.resolved_shear_enu(S, normal=n)
        self.aclose(tau, 0.0, places=12)
        self.mclose(tau_hat, [0.0, 0.0, 0.0], atol=1e-12)

    def test_slip_shapes_and_planes_vs_normals(self):
        S = np.diag([1.0, 2.0, 3.0])

        # scalar plane
        p = np.array([30.0, 60.0])
        n = tr.plane_sphe2enu(p)
        a = tm.slip_tendency(S, planes=p)
        b = tm.slip_tendency(S, normals=n)
        self.assertTrue(np.isscalar(a))
        self.assertTrue(np.isscalar(b))
        self.aclose(a, b, places=9)

        # array planes
        P = np.array([[0.0, 10.0],
                      [30.0, 60.0],
                      [120.0, 45.0]], dtype=float)
        N = np.array([tr.plane_sphe2enu(x) for x in P], dtype=float)
        A = tm.slip_tendency(S, planes=P)
        B = tm.slip_tendency(S, normals=N)
        self.assertEqual(A.shape, (3,))
        self.assertEqual(B.shape, (3,))
        self.mclose(A, B, atol=1e-9)

    def test_dilation_shapes_and_planes_vs_normals(self):
        S = np.diag([1.0, 2.0, 3.0])

        p = np.array([30.0, 60.0])
        n = tr.plane_sphe2enu(p)
        a = tm.dilation_tendency(S, planes=p)
        b = tm.dilation_tendency(S, normals=n)
        self.assertTrue(np.isscalar(a))
        self.assertTrue(np.isscalar(b))
        self.aclose(a, b, places=9)

        P = np.array([[0.0, 10.0],
                      [30.0, 60.0],
                      [120.0, 45.0]], dtype=float)
        N = np.array([tr.plane_sphe2enu(x) for x in P], dtype=float)
        A = tm.dilation_tendency(S, planes=P)
        B = tm.dilation_tendency(S, normals=N)
        self.assertEqual(A.shape, (3,))
        self.assertEqual(B.shape, (3,))
        self.mclose(A, B, atol=1e-9)

    def test_slip_sigma_n_zero(self):
        # Construct stress so that sigma_n = n·S·n = 0 for n = [1,0,0]
        S = np.diag([0.0, 2.0, 3.0])
        n = np.array([1.0, 0.0, 0.0])

        ts = tm.slip_tendency(S, normals=n, eps=1e-14)
        self.assertTrue(np.isinf(ts))

        # for isotropic with sigma_n != 0 => slip is 0 (shear 0)
        S2 = 5.0 * np.eye(3)
        ts2 = tm.slip_tendency(S2, normals=n)
        self.aclose(ts2, 0.0, places=12)

    def test_vectorized_normals(self):
        S = np.array([[4.0, 0.1, 0.2],
                      [0.1, 2.0, 0.3],
                      [0.2, 0.3, 1.0]], dtype=float)

        N = self.rng.normal(size=(25, 3))
        N = N / np.linalg.norm(N, axis=1)[:, None]

        a = tm.slip_tendency(S, normals=N)
        b = np.array([tm.slip_tendency(S, normals=N[i]) for i in range(N.shape[0])])
        self.mclose(a, b, atol=1e-9)

        a = tm.dilation_tendency(S, normals=N)
        b = np.array([tm.dilation_tendency(S, normals=N[i]) for i in range(N.shape[0])])
        self.mclose(a, b, atol=1e-9)


if __name__ == "__main__":
    unittest.main()