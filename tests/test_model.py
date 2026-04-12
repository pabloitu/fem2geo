import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from fem2geo.model import Model
from fem2geo.internal.schema import ModelSchema


_N = 4


class _DictLike(dict):
    def get(self, key, default=None):
        return self[key] if key in self else default


def _grid(cell_data, points=None):
    if points is None:
        points = np.array([[0,0,-1],[1,0,-1],[1,1,-2],[0,1,-2]], dtype=float)
    g = MagicMock()
    g.points = points
    g.number_of_cells = _N
    g.n_cells = _N
    g.array_names = list(cell_data.keys())
    g.cell_data = _DictLike(cell_data)
    g.point_data = _DictLike({})
    g.cell_connectivity = np.tile(np.arange(len(points)), _N)
    g.offset = np.arange(_N + 1) * len(points)
    sizes = MagicMock()
    sizes.cell_data = _DictLike({**cell_data, "Volume": np.ones(_N)})
    g.compute_cell_sizes.return_value = sizes
    centers = MagicMock()
    centers.points = points
    g.cell_centers.return_value = centers
    return g


# fixtures

_DIRS = {
    "dir_s1": np.array([[1, 0, 0]] * _N, dtype=float),
    "dir_s3": np.array([[0, 0, 1]] * _N, dtype=float),
}

_COMP_DATA = {
    "_tensor_stress_xx": np.array([-1, -2, -1.5, -1.0]),
    **{f"_tensor_stress_{c}": np.zeros(_N) for c in ("yy","zz","xy","yz","zx")},
    **_DIRS,
}

_SCHEMA_COMP = ModelSchema.from_dict({
    "solver": "t",
    "tensors": {"stress": {"components": {
        c: c for c in ("xx","yy","zz","xy","yz","zx")}}},
    "fields": {"dir_s1": {"field": "dir_s1"}, "dir_s3": {"field": "dir_s3"}},
})

_SCHEMA_RECON = ModelSchema.from_dict({
    "solver": "t",
    "fields": {
        "val_s1": {"field": "val_s1"}, "val_s3": {"field": "val_s3"},
        "dir_s1": {"field": "dir_s1"}, "dir_s3": {"field": "dir_s3"},
    },
})

_SCHEMA_VOIGT = ModelSchema.from_dict({
    "solver": "t",
    "tensors": {"stress": {"voigt6": "stress"}},
    "fields": {"dir_s1": {"field": "dir_s1"}, "dir_s3": {"field": "dir_s3"}},
})

_SCHEMA_PLASTIC = ModelSchema.from_dict({
    "solver": "t",
    "tensors": {"stress": {"components": {
        c: c for c in ("xx","yy","zz","xy","yz","zx")}}},
    "fields": {
        "dir_s1": {"field": "dir_s1"},
        "dir_s3": {"field": "dir_s3"},
        "val_s1": {"field": "val_s1"},
        "val_s3": {"field": "val_s3"},
        "plastic_eff": {"field": "plastic_eff"},
        "plastic_vol": {"field": "plastic_vol"},
    },
})

_SCHEMA_EMPTY = ModelSchema.from_dict({"solver": "t", "fields": {}})


def _model_comp():
    return Model(_grid(_COMP_DATA), _SCHEMA_COMP)


def _model_recon():
    data = {"val_s1": np.full(_N, -1.0), "val_s3": np.zeros(_N), **_DIRS}
    return Model(_grid(data), _SCHEMA_RECON)


class TestStressAssembly(unittest.TestCase):

    def test_from_components(self):
        m = _model_comp()
        s = m.stress
        self.assertEqual(s.shape, (_N, 3, 3))
        np.testing.assert_allclose(s[:, 0, 0], [-1, -2, -1.5, -1])
        np.testing.assert_allclose(s, s.transpose(0, 2, 1), atol=1e-12)

    def test_from_voigt(self):
        packed = np.column_stack([
            [-1, -2, -1.5, -1.0], np.zeros(_N), np.zeros(_N),
            np.zeros(_N), np.zeros(_N), np.zeros(_N)])
        m = Model(_grid({"stress": packed, **_DIRS}), _SCHEMA_VOIGT)
        s = m.stress
        self.assertEqual(s.shape, (_N, 3, 3))
        np.testing.assert_allclose(s[:, 0, 0], [-1, -2, -1.5, -1])

    def test_from_reconstruction(self):
        m = _model_recon()
        with self.assertLogs("fem2geoLogger", level="WARNING") as cm:
            s = m.stress
        self.assertEqual(s.shape, (_N, 3, 3))
        np.testing.assert_allclose(s, s.transpose(0, 2, 1), atol=1e-12)
        self.assertTrue(any("reconstruct" in msg.lower() for msg in cm.output))

    def test_reconstruction_fails_without_principals(self):
        m = Model(_grid({"dir_s1": _DIRS["dir_s1"]}), _SCHEMA_RECON)
        with self.assertRaises(KeyError):
            _ = m.stress


class TestPrincipalDirections(unittest.TestCase):

    def setUp(self):
        self.m = _model_comp()

    def test_s2_orthogonal_and_unit(self):
        d2 = self.m.dir_s2
        np.testing.assert_allclose(
            np.einsum("ij,ij->i", d2, self.m.dir_s1), 0, atol=1e-12)
        np.testing.assert_allclose(
            np.einsum("ij,ij->i", d2, self.m.dir_s3), 0, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(d2, axis=1), 1, atol=1e-12)

    def test_s2_from_file_preferred(self):
        s2 = np.array([[0, 1, 0]] * _N, dtype=float)
        m = Model(_grid({**_COMP_DATA, "dir_s2": s2}), _SCHEMA_COMP)
        np.testing.assert_allclose(m.dir_s2, s2)

    def test_deviatoric_trace_zero(self):
        m = self.m
        np.testing.assert_allclose(
            m.val_s1 + m.val_s2 + m.val_s3, 0, atol=1e-10)


class TestEigen(unittest.TestCase):

    def setUp(self):
        self.m = _model_comp()

    def test_shapes_and_ordering(self):
        vals = self.m.eigenvalues("stress")
        self.assertEqual(vals.shape, (_N, 3))
        self.assertTrue(np.all(vals[:, 0] <= vals[:, 1]))
        self.assertTrue(np.all(vals[:, 1] <= vals[:, 2]))

    def test_orthonormal_eigenvectors(self):
        vecs = self.m.eigenvectors("stress")
        for i in range(_N):
            np.testing.assert_allclose(
                vecs[i].T @ vecs[i], np.eye(3), atol=1e-12)

    def test_eigenvalues_match_val_properties(self):
        vals = self.m.eigenvalues("stress")
        np.testing.assert_allclose(self.m.val_s1, vals[:, 0], atol=1e-12)
        np.testing.assert_allclose(self.m.val_s3, vals[:, 2], atol=1e-12)

    def test_eigenvalues_strain(self):
        vals = self.m.eigenvalues("stress_dev")
        self.assertEqual(vals.shape, (_N, 3))


class TestAverages(unittest.TestCase):

    def test_avg_principal_sorted_and_orthonormal(self):
        m = _model_comp()
        val, vec = m.avg_principals()
        self.assertEqual(val.shape, (3,))
        self.assertTrue(np.all(val[:-1] <= val[1:]))
        np.testing.assert_allclose(vec.T @ vec, np.eye(3), atol=1e-10)

    def test_avg_principal_accepts_name(self):
        m = _model_comp()
        val, vec = m.avg_principals("stress")
        self.assertEqual(val.shape, (3,))
        val_d, vec_d = m.avg_principals("stress_dev")
        self.assertEqual(val_d.shape, (3,))

    def test_avg_tensor_stress(self):
        m = _model_comp()
        avg = m.avg_tensor("stress")
        self.assertEqual(avg.shape, (3, 3))
        np.testing.assert_allclose(avg, avg.T, atol=1e-12)

    def test_avg_tensor_symmetrized(self):
        m = _model_comp()
        avg = m.avg_tensor("stress")
        np.testing.assert_allclose(avg, avg.T, atol=1e-12)

    def test_stress_dev_removes_isotropic(self):
        data = {
            "_tensor_stress_xx": np.array([-3, -3, -3, -3.0]),
            "_tensor_stress_yy": np.array([-2, -2, -2, -2.0]),
            "_tensor_stress_zz": np.array([-1, -1, -1, -1.0]),
            **{f"_tensor_stress_{c}": np.zeros(_N) for c in ("xy", "yz", "zx")},
            **_DIRS,
        }
        m = Model(_grid(data), _SCHEMA_COMP)
        s = m.stress
        sd = m.stress_dev
        traces = np.trace(sd, axis1=1, axis2=2)
        np.testing.assert_allclose(traces, 0, atol=1e-12)
        _, v_full = np.linalg.eigh(s[0])
        _, v_dev = np.linalg.eigh(sd[0])
        for i in range(3):
            dot = abs(np.dot(v_full[:, i], v_dev[:, i]))
            self.assertAlmostEqual(dot, 1.0, places=10)


class TestStrainReconstruction(unittest.TestCase):

    def _plastic_grid(self, eff, vol):
        data = {
            **_COMP_DATA,
            "val_s1": np.full(_N, -2.0),
            "val_s3": np.full(_N, 0.0),
            "plastic_eff": np.full(_N, eff),
            "plastic_vol": np.full(_N, vol),
        }
        return Model(_grid(data), _SCHEMA_PLASTIC)

    def test_zero_plastic_returns_zero_tensor(self):
        m = self._plastic_grid(0.0, 0.0)
        ep = m.strain_plastic
        self.assertEqual(ep.shape, (_N, 3, 3))
        np.testing.assert_allclose(ep, 0, atol=1e-12)

    def test_nonzero_plastic_reconstruction(self):
        m = self._plastic_grid(0.01, 0.005)
        ep = m.strain_plastic
        self.assertEqual(ep.shape, (_N, 3, 3))
        np.testing.assert_allclose(ep, ep.transpose(0, 2, 1), atol=1e-12)

    def test_missing_plastic_fields_assumes_elastic(self):
        m = _model_comp()
        with self.assertLogs("fem2geoLogger", level="INFO"):
            ep = m.strain_plastic
        np.testing.assert_allclose(ep, 0, atol=1e-12)


class TestExtraction(unittest.TestCase):

    def setUp(self):
        self.m = _model_comp()

    def test_extract_returns_model(self):
        self.assertIsInstance(
            self.m.extract([.5, .5, -1.5], 10), Model)

    def test_preserves_schema(self):
        sub = self.m.extract([.5, .5, -1.5], 10)
        self.assertIs(sub.schema, self.m.schema)

    def test_large_radius_selects_all(self):
        self.m.extract([.5, .5, -1.5], 100)
        ids = self.m.grid.extract_cells.call_args[0][0]
        np.testing.assert_array_equal(np.sort(ids), np.arange(_N))

    def test_extract_empty_raises(self):
        empty = MagicMock()
        empty.n_cells = 0
        empty.number_of_cells = 0
        self.m.grid.extract_cells.return_value = empty
        with self.assertRaises(ValueError):
            self.m.extract([1000, 1000, 1000], 0.001)

    def test_extract_cells_returns_unstructured_on_empty(self):
        result = self.m._extract_cells(np.zeros(4, dtype=bool))
        self.assertIsNotNone(result)

    def test_missing_field_raises(self):
        with self.assertRaises(KeyError):
            self.m.u


class TestFromFile(unittest.TestCase):

    def test_string_schema_resolved(self):
        with patch("fem2geo.model.load_solver_output", return_value=_grid({})), \
             patch("fem2geo.model.ModelSchema.builtin",
                   return_value=_SCHEMA_EMPTY) as mock:
            m = Model.from_file("x.vtk", schema="adeli")
        mock.assert_called_once_with("adeli")
        self.assertIsInstance(m, Model)

    def test_schema_retained(self):
        with patch("fem2geo.model.load_solver_output", return_value=_grid({})):
            m = Model.from_file("x.vtk", schema=_SCHEMA_EMPTY)
        self.assertIs(m.schema, _SCHEMA_EMPTY)

class TestStrainVariants(unittest.TestCase):
    """Cover strain, strain_rate and strain_elastic properties."""

    def _make(self, tensors_in_schema, cell_data):
        schema = ModelSchema.from_dict({
            "solver": "t",
            "tensors": {
                name: {"voigt6": name} for name in tensors_in_schema
            },
            "fields": {
                "dir_s1": {"field": "dir_s1"},
                "dir_s3": {"field": "dir_s3"},
            },
        })
        full = {**cell_data, **_DIRS}
        return Model(_grid(full), schema)

    def test_strain_assembled(self):
        packed = np.array([[1e-3, -5e-4, -5e-4, 0, 0, 0]] * _N)
        m = self._make(["strain"], {"strain": packed})
        s = m.strain
        self.assertEqual(s.shape, (_N, 3, 3))
        np.testing.assert_allclose(s, s.transpose(0, 2, 1), atol=1e-12)

    def test_strain_rate_assembled(self):
        packed = np.array([[1e-12, -5e-13, -5e-13, 0, 0, 0]] * _N)
        m = self._make(["strain_rate"], {"strain_rate": packed})
        sr = m.strain_rate
        self.assertEqual(sr.shape, (_N, 3, 3))

    def test_strain_elastic_from_schema(self):
        packed = np.array([[1e-4, 0, 0, 0, 0, 0]] * _N)
        m = self._make(["strain_elastic"], {"strain_elastic": packed})
        se = m.strain_elastic
        self.assertEqual(se.shape, (_N, 3, 3))
        np.testing.assert_allclose(se[:, 0, 0], 1e-4)

    def test_strain_elastic_computed_as_difference(self):
        strain_packed = np.array([[2e-3, 0, 0, 0, 0, 0]] * _N)
        plastic_packed = np.array([[1e-3, 0, 0, 0, 0, 0]] * _N)
        m = self._make(
            ["strain", "strain_plastic"],
            {"strain": strain_packed, "strain_plastic": plastic_packed},
        )
        se = m.strain_elastic
        np.testing.assert_allclose(se[:, 0, 0], 1e-3, atol=1e-12)


class TestAvgTensorVariants(unittest.TestCase):
    """avg_tensor / avg_principals across all tensor kinds."""

    def test_strain_avg_tensor(self):
        packed = np.array([[1e-3, 0, 0, 0, 0, 0]] * _N)
        schema = ModelSchema.from_dict({
            "solver": "t",
            "tensors": {"strain": {"voigt6": "strain"}},
            "fields": {},
        })
        m = Model(_grid({"strain": packed}), schema)
        avg = m.avg_tensor("strain")
        self.assertEqual(avg.shape, (3, 3))
        np.testing.assert_allclose(avg[0, 0], 1e-3, atol=1e-15)

    def test_avg_principals_strain(self):
        packed = np.array([[1e-3, -5e-4, -5e-4, 0, 0, 0]] * _N)
        schema = ModelSchema.from_dict({
            "solver": "t",
            "tensors": {"strain": {"voigt6": "strain"}},
            "fields": {},
        })
        m = Model(_grid({"strain": packed}), schema)
        val, vec = m.avg_principals("strain")
        self.assertEqual(val.shape, (3,))
        self.assertTrue(val[0] <= val[1] <= val[2])
        np.testing.assert_allclose(vec.T @ vec, np.eye(3), atol=1e-10)


class TestScalarFieldProperties(unittest.TestCase):
    """Happy-path and failure-path for scalar field lookups."""

    def _with_fields(self, data):
        schema = ModelSchema.from_dict({
            "solver": "t",
            "fields": {k: {"field": k} for k in data},
        })
        return Model(_grid(data), schema)

    def test_mean_stress(self):
        m = self._with_fields({"mean_stress": np.full(_N, -2.0)})
        np.testing.assert_allclose(m.mean_stress, -2.0)

    def test_j2_stress(self):
        m = self._with_fields({"j2_stress": np.full(_N, 1.5)})
        np.testing.assert_allclose(m.j2_stress, 1.5)

    def test_temperature(self):
        m = self._with_fields({"temperature": np.full(_N, 300.0)})
        np.testing.assert_allclose(m.temperature, 300.0)

    def test_viscosity(self):
        m = self._with_fields({"viscosity": np.full(_N, 1e21)})
        np.testing.assert_allclose(m.viscosity, 1e21)

    def test_vector_field_u(self):
        u = np.tile([1.0, 0.0, -0.5], (_N, 1))
        m = self._with_fields({"u": u})
        np.testing.assert_allclose(m.u, u)


class TestFromFilePathDispatch(unittest.TestCase):
    """Model.from_file should route strings through ModelSchema.load."""

    def test_string_dispatches_via_load(self):
        with patch("fem2geo.model.load_solver_output", return_value=_grid({})), \
             patch("fem2geo.model.ModelSchema.load",
                   return_value=_SCHEMA_EMPTY) as mock:
            m = Model.from_file("x.vtk", schema="adeli")
        mock.assert_called_once_with("adeli")
        self.assertIsInstance(m, Model)

    def test_instance_bypasses_load(self):
        with patch("fem2geo.model.load_solver_output", return_value=_grid({})), \
             patch("fem2geo.model.ModelSchema.load") as mock:
            Model.from_file("x.vtk", schema=_SCHEMA_EMPTY)
        mock.assert_not_called()
if __name__ == "__main__":
    unittest.main()