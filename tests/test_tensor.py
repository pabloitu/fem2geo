import unittest
import numpy as np

from fem2geo.utils import transform as tr
import fem2geo.utils.tensor as tm


class TestRotation(unittest.TestCase):

    def test_tensor_rotation_preserves_invariants(self):
        T = np.array([[3, .2, -.1], [.2, 2, .3], [-.1, .3, 1.0]])
        ev0 = np.sort(np.linalg.eigvalsh(T))
        for ax in (0, 1, 2):
            Tr = tm.rot_tensor(T, 67, ax)
            np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(Tr)), ev0, atol=1e-9)
            self.assertAlmostEqual(np.trace(Tr), np.trace(T), places=9)

    def test_invalid_axis_raises(self):
        T = np.eye(3)
        with self.assertRaises(ValueError):
            tm.rot_tensor(T, 30, axis=3)

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            tm.rot_tensor(np.eye(2), 30, axis=0)


class TestValidateNormals(unittest.TestCase):

    def test_single_vector_returned_2d(self):
        n = tm.validate_normals([3.0, 0.0, 0.0])
        self.assertEqual(n.shape, (1, 3))
        np.testing.assert_allclose(n[0], [1, 0, 0])

    def test_array_normalized_per_row(self):
        n = tm.validate_normals([[3, 0, 0], [0, 4, 0]])
        self.assertEqual(n.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), [1, 1])

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            tm.validate_normals([0.0, 0.0, 0.0])

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            tm.validate_normals([1.0, 0.0])

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            tm.validate_normals(np.zeros((2, 2, 3)))


class TestReconstructFromPrincipals(unittest.TestCase):

    def test_diagonal_round_trip(self):
        vals = np.array([[-3.0, -1.0, 2.0]])
        dirs = np.eye(3)[None, :, :]
        T = tm.reconstruct_from_principals(vals, dirs)
        self.assertEqual(T.shape, (1, 3, 3))
        np.testing.assert_allclose(np.diag(T[0]), [-3, -1, 2])
        np.testing.assert_allclose(T[0], T[0].T, atol=1e-12)

    def test_eigendecomposition_round_trip(self):
        rng = np.random.default_rng(0)
        T0 = rng.normal(size=(3, 3))
        T0 = 0.5 * (T0 + T0.T)
        vals, vecs = np.linalg.eigh(T0)
        T = tm.reconstruct_from_principals(vals[None, :], vecs[None, :, :])
        np.testing.assert_allclose(T[0], T0, atol=1e-12)

    def test_batch_shape(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(size=(5, 3))
        dirs = np.tile(np.eye(3), (5, 1, 1))
        T = tm.reconstruct_from_principals(vals, dirs)
        self.assertEqual(T.shape, (5, 3, 3))


class TestResolvedShear(unittest.TestCase):

    def test_isotropic_gives_zero(self):
        S = 5.0 * np.eye(3)
        for p in ([0, 0], [30, 10], [120, 60], [270, 85]):
            tau, tau_hat = tm.resolved_shear_enu(S, plane=p)
            self.assertAlmostEqual(tau, 0.0, places=12)
            np.testing.assert_allclose(tau_hat, 0, atol=1e-12)

    def test_normal_argument_matches_plane(self):
        S = np.array([[4, .1, .2], [.1, 2, .3], [.2, .3, 1.0]])
        n = tr.plane_sphe2enu(45.0, 60.0)
        tau_p, hat_p = tm.resolved_shear_enu(S, plane=[45.0, 60.0])
        tau_n, hat_n = tm.resolved_shear_enu(S, normal=n)
        self.assertAlmostEqual(tau_p, tau_n, places=12)
        np.testing.assert_allclose(hat_p, hat_n, atol=1e-12)

    def test_both_plane_and_normal_raises(self):
        S = np.eye(3)
        with self.assertRaises(ValueError):
            tm.resolved_shear_enu(S, plane=[0, 0], normal=[1, 0, 0])

    def test_neither_plane_nor_normal_raises(self):
        S = np.eye(3)
        with self.assertRaises(ValueError):
            tm.resolved_shear_enu(S)

    def test_wrong_sigma_shape_raises(self):
        with self.assertRaises(ValueError):
            tm.resolved_shear_enu(np.eye(2), plane=[0, 0])


class TestResolvedRakes(unittest.TestCase):

    def test_isotropic_gives_nan(self):
        S = 5.0 * np.eye(3)
        rakes = tm.resolved_rake(S, [0, 30, 120], [10, 60, 85])
        self.assertTrue(np.all(np.isnan(rakes)))

    def test_matches_scalar_resolved_shear(self):
        S = np.array([[4, .1, .2], [.1, 2, .3], [.2, .3, 1.0]])
        strikes = np.array([0, 30, 120, 270], dtype=float)
        dips = np.array([10, 60, 45, 85], dtype=float)
        rakes = tm.resolved_rake(S, strikes, dips)
        for i in range(len(strikes)):
            tau, tau_hat = tm.resolved_shear_enu(S, plane=[strikes[i], dips[i]])
            if tau < 1e-12:
                self.assertTrue(np.isnan(rakes[i]))
            else:
                expected = tr.slip_enu2rake(tau_hat, strikes[i], dips[i])
                d = (rakes[i] - expected + 180) % 360 - 180
                self.assertAlmostEqual(d, 0.0, places=7)

    def test_wrong_sigma_shape_raises(self):
        with self.assertRaises(ValueError):
            tm.resolved_rake(np.eye(2), [0], [30])


class TestMohrTsMax(unittest.TestCase):

    def test_diagonal_stress(self):
        S = np.diag([-3.0, -1.0, -0.5])
        ts = tm.mohr_ts_max(S)
        self.assertGreater(ts, 0)

    def test_isotropic_returns_zero(self):
        ts = tm.mohr_ts_max(5.0 * np.eye(3))
        self.assertAlmostEqual(ts, 0.0, places=12)

    def test_zero_denominator_handled(self):
        S = np.diag([1.0, -1.0, 0.0])
        ts = tm.mohr_ts_max(S)
        self.assertGreaterEqual(ts, 0.0)


class TestTendencies(unittest.TestCase):

    def test_scalar_vs_batch(self):
        S = np.array([[4, .1, .2], [.1, 2, .3], [.2, .3, 1.0]])
        rng = np.random.default_rng(123)
        N = rng.normal(size=(25, 3))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        strikes, dips = [], []
        for i in range(25):
            p, a = tr.line_enu2sphe(N[i])
            s, d = tr.plane_pole2sphe(p, a)
            strikes.append(s)
            dips.append(d)
        strikes = np.array(strikes)
        dips = np.array(dips)

        batch = tm.slip_tendency(S, strikes, dips)
        scalar = [float(tm.slip_tendency(S, strikes[i], dips[i])) for i in range(25)]
        np.testing.assert_allclose(batch, scalar, atol=1e-9)

    def test_slip_bounded_01(self):
        S = np.diag([-3.0, -1.0, -0.5])
        rng = np.random.default_rng(42)
        strikes = rng.uniform(0, 360, 100)
        dips = rng.uniform(1, 89, 100)
        ts = tm.slip_tendency(S, strikes, dips)
        self.assertTrue(np.all(ts >= 0))
        self.assertTrue(np.all(ts <= 1.0 + 1e-12))
        self.assertGreater(np.max(ts), 0.9)

    def test_slip_isotropic_zero(self):
        S = 5.0 * np.eye(3)
        ts = tm.slip_tendency(S, 30, 45)
        self.assertAlmostEqual(ts, 0.0, places=10)

    def test_slip_wrong_sigma_shape_raises(self):
        with self.assertRaises(ValueError):
            tm.slip_tendency(np.eye(2), 30, 45)

    def test_dilation_bounded_01(self):
        S = np.diag([-3.0, -1.0, -0.5])
        td = tm.dilation_tendency(S, 30, 45)
        self.assertGreaterEqual(td, 0.0)
        self.assertLessEqual(td, 1.0 + 1e-12)

    def test_dilation_isotropic_returns_nan(self):
        S = 5.0 * np.eye(3)
        td = tm.dilation_tendency(S, 30, 45)
        self.assertTrue(np.isnan(td))

    def test_dilation_wrong_sigma_shape_raises(self):
        with self.assertRaises(ValueError):
            tm.dilation_tendency(np.eye(2), 30, 45)

    def test_summarized_is_sum(self):
        S = np.diag([-3.0, -1.0, -0.5])
        strikes = np.array([0, 90], dtype=float)
        dips = np.array([30, 60], dtype=float)
        ts = tm.slip_tendency(S, strikes, dips)
        td = tm.dilation_tendency(S, strikes, dips)
        tc = tm.summarized_tendency(S, strikes, dips)
        np.testing.assert_allclose(tc, ts + td, atol=1e-12)

    def test_summarized_bounded_02(self):
        S = np.diag([-3.0, -1.0, -0.5])
        rng = np.random.default_rng(7)
        strikes = rng.uniform(0, 360, 100)
        dips = rng.uniform(1, 89, 100)
        tc = tm.summarized_tendency(S, strikes, dips)
        self.assertTrue(np.all(tc >= 0))
        self.assertTrue(np.all(tc <= 2.0 + 1e-12))


class TestAxesMisfit(unittest.TestCase):

    def test_identical_gives_zero(self):
        vecs = np.eye(3)
        angles, pairs = tm.axes_misfit(vecs, vecs)
        np.testing.assert_allclose(angles, 0.0, atol=1e-12)
        self.assertEqual(len(pairs), 3)

    def test_permuted_axes(self):
        a = np.eye(3)
        b = np.eye(3)[:, [2, 0, 1]]
        angles, pairs = tm.axes_misfit(a, b)
        np.testing.assert_allclose(angles, 0.0, atol=1e-12)

    def test_tilted_axes(self):
        a = np.eye(3)
        c = np.cos(np.deg2rad(45))
        s = np.sin(np.deg2rad(45))
        b = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float).T
        angles, _ = tm.axes_misfit(a, b)
        self.assertAlmostEqual(min(angles), 0.0, places=9)
        self.assertAlmostEqual(max(angles), 45.0, places=9)


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
        packed = np.array([[T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[1, 2], T[0, 2]]])
        np.testing.assert_allclose(tm.unpack_voigt6(packed)[0], T, atol=1e-14)


class TestUnpackComponents(unittest.TestCase):

    def test_matches_voigt(self):
        packed = np.array([[1, 2, 3, .4, .5, .6]])
        arrays = dict(xx=[1], yy=[2], zz=[3], xy=[.4], yz=[.5], zx=[.6])
        np.testing.assert_allclose(tm.unpack_components(arrays), tm.unpack_voigt6(packed), atol=1e-14)


class TestEigenFunctions(unittest.TestCase):

    def setUp(self):
        T = np.array([[3, .2, -.1], [.2, 2, .3], [-.1, .3, 1.0]])
        self.tensors = np.stack([T, 2 * T, -T])

    def test_eigenvalues_shape_and_sorted(self):
        vals = tm.eigenvalues(self.tensors)
        self.assertEqual(vals.shape, (3, 3))
        self.assertTrue(np.all(vals[:, 0] <= vals[:, 1]))
        self.assertTrue(np.all(vals[:, 1] <= vals[:, 2]))

    def test_eigenvectors_shape_and_orthonormal(self):
        vecs = tm.eigenvectors(self.tensors)
        self.assertEqual(vecs.shape, (3, 3, 3))
        for i in range(3):
            np.testing.assert_allclose(vecs[i].T @ vecs[i], np.eye(3), atol=1e-12)

    def test_eigenvalues_consistent_with_numpy(self):
        vals = tm.eigenvalues(self.tensors)
        for i in range(3):
            expected = np.sort(np.linalg.eigvalsh(self.tensors[i]))
            np.testing.assert_allclose(vals[i], expected, atol=1e-12)

    def test_eigenvectors_reconstruct_tensor(self):
        vecs = tm.eigenvectors(self.tensors)
        vals = tm.eigenvalues(self.tensors)
        for i in range(3):
            T_reconstructed = (vecs[i] * vals[i]) @ vecs[i].T
            np.testing.assert_allclose(T_reconstructed, self.tensors[i], atol=1e-12)


class TestKostrov(unittest.TestCase):

    def test_sinistral_vertical(self):
        K = tm.kostrov_tensor([0], [90], [0])
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        vals = np.linalg.eigvalsh(K)
        self.assertAlmostEqual(vals[0], -0.5, places=10)
        self.assertAlmostEqual(vals[2], 0.5, places=10)

    def test_always_symmetric(self):
        K = tm.kostrov_tensor([30, 120, 0], [60, 45, 90], [90, -45, 0])
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_reverse_population_axes(self):
        rng = np.random.default_rng(1)
        n = 30
        strikes = rng.uniform(0, 360, n)
        dips = np.full(n, 45.0)
        rakes = np.full(n, 90.0)
        K = tm.kostrov_tensor(strikes, dips, rakes)
        _, vecs = np.linalg.eigh(K)
        p_axis = vecs[:, 0]
        t_axis = vecs[:, 2]
        self.assertLess(abs(p_axis[2]), 0.6)
        self.assertGreater(abs(t_axis[2]), 0.4)


class TestRakeRoundTrip(unittest.TestCase):

    def test_slip_rake2enu_roundtrip(self):
        rng = np.random.default_rng(0)
        strikes = rng.uniform(0, 360, 200)
        dips = rng.uniform(1, 89, 200)
        rakes = rng.uniform(-179.9, 180.0, 200)
        enu = tr.slip_rake2enu(strikes, dips, rakes)
        back = tr.slip_enu2rake(enu, strikes, dips)
        d = (back - rakes + 180) % 360 - 180
        np.testing.assert_allclose(d, 0, atol=1e-7)

    def test_rake_180_and_minus_180_equivalent(self):
        v_pos = tr.slip_rake2enu(45.0, 60.0, 180.0)
        v_neg = tr.slip_rake2enu(45.0, 60.0, -180.0)
        dot = abs(np.dot(v_pos, v_neg))
        self.assertAlmostEqual(dot, 1.0, places=12)

    def test_pure_reverse(self):
        enu = tr.slip_rake2enu(0.0, 45.0, 90.0)
        self.assertAlmostEqual(enu[2], np.sin(np.deg2rad(45.0)), places=12)

    def test_pure_normal(self):
        enu = tr.slip_rake2enu(0.0, 45.0, -90.0)
        self.assertAlmostEqual(enu[2], -np.sin(np.deg2rad(45.0)), places=12)


class TestLineRakeSphe(unittest.TestCase):

    def test_rake_above_90(self):
        p, a = tr.line_rake2sphe(0.0, 60.0, 120.0)
        self.assertGreaterEqual(p, 0)
        self.assertLess(p, 90)
        self.assertTrue(0 <= a < 360)

    def test_rake_at_90(self):
        p, a = tr.line_rake2sphe(0.0, 60.0, 90.0)
        self.assertAlmostEqual(p, 60.0, places=9)

    def test_rake_zero_horizontal(self):
        p, a = tr.line_rake2sphe(0.0, 60.0, 0.0)
        self.assertAlmostEqual(p, 0.0, places=9)

    def test_rake_180_horizontal(self):
        p, a = tr.line_rake2sphe(0.0, 60.0, 180.0)
        self.assertAlmostEqual(p, 0.0, places=9)


class TestPlaneSphe2ENUVertical(unittest.TestCase):

    def test_vertical_plane_normal_is_horizontal(self):
        n = tr.plane_sphe2enu(0.0, 90.0)
        self.assertAlmostEqual(n[2], 0.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=12)

    def test_vertical_plane_canonicalization(self):
        n1 = tr.plane_sphe2enu(0.0, 90.0)
        n2 = tr.plane_sphe2enu(180.0, 90.0)
        dot = abs(np.dot(n1, n2))
        self.assertAlmostEqual(dot, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()