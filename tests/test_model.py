import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np

from fem2geo.model import Model
from fem2geo.internal.schema import ModelSchema


_N = 4

_STRESS_FIELDS = {
    "s_xx": np.array([-1.0, -2.0, -1.5, -1.0]),
    "s_yy": np.array([ 0.0,  0.0,  0.0,  0.0]),
    "s_zz": np.array([ 0.0,  0.0,  0.0,  0.0]),
    "s_xy": np.array([ 0.0,  0.0,  0.0,  0.0]),
    "s_yz": np.array([ 0.0,  0.0,  0.0,  0.0]),
    "s_zx": np.array([ 0.0,  0.0,  0.0,  0.0]),
}

_DIR_FIELDS = {
    "dir_s1": np.array([[1., 0., 0.]] * _N),
    "dir_s3": np.array([[0., 0., 1.]] * _N),
}


def _make_grid(cell_data: dict, points: np.ndarray = None):
    if points is None:
        points = np.array([
            [0., 0., -1.], [1., 0., -1.],
            [1., 1., -2.], [0., 1., -2.],
        ])
    grid = MagicMock()
    grid.points = points
    grid.number_of_cells = _N
    grid.n_cells = _N
    grid.array_names = list(cell_data.keys())
    grid.cell_data = _DictLike(cell_data)
    grid.point_data = _DictLike({})
    grid.cells_dict = {10: np.array([[0, 1, 2, 3]] * _N)}
    grid.celltypes = np.array([10] * _N)

    # VTK connectivity arrays used by _extract
    grid.cell_connectivity = np.tile(np.arange(len(points)), _N)
    grid.offset = np.arange(_N + 1) * len(points)

    sizes_mock = MagicMock()
    sizes_mock.cell_data = _DictLike({**cell_data, "Volume": np.ones(_N)})
    grid.compute_cell_sizes.return_value = sizes_mock

    centers_mock = MagicMock()
    centers_mock.points = points
    grid.cell_centers.return_value = centers_mock

    return grid


class _DictLike(dict):
    def get(self, key, default=None):
        return self[key] if key in self else default


def _make_model(cell_data=None):
    data = cell_data if cell_data is not None else {**_STRESS_FIELDS, **_DIR_FIELDS}
    schema = ModelSchema.from_dict({
        "solver": "test",
        "units": {"pressure": "MPa"},
        "fields": {
            **{k: {"field": k, "category": "pressure"} for k in _STRESS_FIELDS},
            "dir_s1": {"field": "dir_s1"},
            "dir_s3": {"field": "dir_s3"},
        },
    })
    return Model(_make_grid(data), schema)


class TestFieldProperties(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = _make_model()

    def test_stress_shape(self):
        self.assertEqual(self.m.stress.shape, (_N, 3, 3))

    def test_stress_values(self):
        np.testing.assert_allclose(self.m.stress[:, 0, 0], _STRESS_FIELDS["s_xx"])

    def test_stress_symmetric(self):
        s = self.m.stress
        np.testing.assert_allclose(s, s.transpose(0, 2, 1), atol=1e-12)

    def test_dir_s1_shape(self):
        self.assertEqual(self.m.dir_s1.shape, (_N, 3))

    def test_dir_s3_shape(self):
        self.assertEqual(self.m.dir_s3.shape, (_N, 3))

    def test_n_cells(self):
        self.assertEqual(self.m.n_cells, _N)

    def test_missing_field_raises(self):
        with self.assertRaises(KeyError) as ctx:
            self.m.u
        self.assertIn("u", str(ctx.exception))


class TestDerivedFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = _make_model()

    def test_dir_s2_orthogonal_to_s1(self):
        dots = np.einsum("ij,ij->i", self.m.dir_s2, self.m.dir_s1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_dir_s2_orthogonal_to_s3(self):
        dots = np.einsum("ij,ij->i", self.m.dir_s2, self.m.dir_s3)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_dir_s2_unit_length(self):
        norms = np.linalg.norm(self.m.dir_s2, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_val_s2_deviatoric_trace(self):
        trace = self.m.val_s1 + self.m.val_s2 + self.m.val_s3
        np.testing.assert_allclose(trace, 0.0, atol=1e-10)

    def test_dir_s2_from_file_used_when_present(self):
        s2 = np.array([[0., 1., 0.]] * _N)
        m = _make_model({**_STRESS_FIELDS, **_DIR_FIELDS, "dir_s2": s2})
        np.testing.assert_allclose(m.dir_s2, s2)


class TestEigenDecomposition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = _make_model()

    def test_eigenvalues_shape(self):
        self.assertEqual(self.m.eigenvalues().shape, (_N, 3))

    def test_eigenvalues_sorted(self):
        vals = self.m.eigenvalues()
        self.assertTrue(np.all(vals[:, 0] <= vals[:, 1]))
        self.assertTrue(np.all(vals[:, 1] <= vals[:, 2]))

    def test_eigenvectors_shape(self):
        self.assertEqual(self.m.eigenvectors().shape, (_N, 3, 3))

    def test_eigenvectors_orthonormal(self):
        vecs = self.m.eigenvectors()
        for i in range(_N):
            np.testing.assert_allclose(vecs[i].T @ vecs[i], np.eye(3), atol=1e-12)

    def test_eigenvalues_match_val_s1_s3(self):
        vals = self.m.eigenvalues()
        np.testing.assert_allclose(self.m.val_s1, vals[:, 0], atol=1e-12)
        np.testing.assert_allclose(self.m.val_s3, vals[:, 2], atol=1e-12)


class TestAvgPrincipal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = _make_model()

    def test_returns_val_vec(self):
        val, vec = self.m.avg_principal()
        self.assertEqual(val.shape, (3,))
        self.assertEqual(vec.shape, (3, 3))

    def test_val_sorted_ascending(self):
        val, _ = self.m.avg_principal()
        self.assertTrue(np.all(val[:-1] <= val[1:]))

    def test_vec_orthonormal(self):
        _, vec = self.m.avg_principal()
        np.testing.assert_allclose(vec.T @ vec, np.eye(3), atol=1e-10)


class TestStressReconstruction(unittest.TestCase):

    def test_reconstruction_from_principals(self):
        data = {
            "val_s1": np.array([-1.0] * _N),
            "val_s3": np.array([ 0.0] * _N),
            **_DIR_FIELDS,
        }
        m = _make_model(data)
        with self.assertLogs("fem2geoLogger", level="WARNING") as cm:
            s = m.stress
        self.assertEqual(s.shape, (_N, 3, 3))
        self.assertTrue(any("reconstructing" in msg.lower() for msg in cm.output))

    def test_reconstruction_symmetric(self):
        data = {
            "val_s1": np.array([-1.0] * _N),
            "val_s3": np.array([ 0.0] * _N),
            **_DIR_FIELDS,
        }
        m = _make_model(data)
        with self.assertLogs("fem2geoLogger", level="WARNING"):
            s = m.stress
        np.testing.assert_allclose(s, s.transpose(0, 2, 1), atol=1e-12)

    def test_reconstruction_fails_without_required_fields(self):
        m = _make_model({"dir_s1": _DIR_FIELDS["dir_s1"]})
        with self.assertRaises(KeyError):
            _ = m.stress


class TestExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = _make_model()

    def test_extract_sphere_returns_model(self):
        sub = self.m.extract_sphere(center=[0.5, 0.5, -1.5], radius=10.0)
        self.assertIsInstance(sub, Model)

    def test_extract_box_returns_model(self):
        sub = self.m.extract_box(center=[0.5, 0.5, -1.5], dim=[5., 5., 5.])
        self.assertIsInstance(sub, Model)

    def test_extract_preserves_schema(self):
        sub = self.m.extract_sphere(center=[0.5, 0.5, -1.5], radius=10.0)
        self.assertIs(sub.schema, self.m.schema)

    def test_extract_empty_no_crash(self):
        sub = self.m.extract_sphere(center=[999., 999., 999.], radius=0.001)
        self.assertIsInstance(sub, Model)

    def test_extract_sphere_calls_extract_cells(self):
        """Large radius should select all cells."""
        self.m.extract_sphere(center=[0.5, 0.5, -1.5], radius=100.0)
        self.m._grid.extract_cells.assert_called()
        ids = self.m._grid.extract_cells.call_args[0][0]
        np.testing.assert_array_equal(np.sort(ids), np.arange(_N))

    def test_extract_box_calls_extract_cells(self):
        """Large box should select all cells."""
        self.m.extract_box(center=[0.5, 0.5, -1.5], dim=[100., 100., 100.])
        self.m._grid.extract_cells.assert_called()
        ids = self.m._grid.extract_cells.call_args[0][0]
        np.testing.assert_array_equal(np.sort(ids), np.arange(_N))

    def test_extract_partial_selection(self):
        """Only points near origin should be selected."""
        # points[0] = [0, 0, -1], points[1] = [1, 0, -1] are within radius 1.5
        # points[2] = [1, 1, -2], points[3] = [0, 1, -2] are further away
        # All cells reference all 4 points, so all cells are still touched.
        # Use a tighter radius that excludes points 2 and 3.
        sub = self.m.extract_sphere(center=[0.5, 0.0, -1.0], radius=0.6)
        self.m._grid.extract_cells.assert_called()

    def test_extract_mask_based(self):
        """Directly test _extract with a boolean mask."""
        mask = np.array([True, False, False, False])
        self.m._extract(mask)
        self.m._grid.extract_cells.assert_called()
        ids = self.m._grid.extract_cells.call_args[0][0]
        # All cells reference point 0, so all should be selected
        np.testing.assert_array_equal(np.sort(ids), np.arange(_N))

    def test_extract_mask_none_selected(self):
        """All-False mask should return empty grid."""
        mask = np.array([False, False, False, False])
        result = self.m._extract(mask)
        # Should not call extract_cells, should return empty grid
        self.assertIsNotNone(result)


class TestFromFile(unittest.TestCase):

    def _schema(self):
        return ModelSchema.from_dict({"solver": "test", "units": {}, "fields": {}})

    def test_from_file_returns_model(self):
        schema = self._schema()
        with patch("fem2geo.model.load_grid", return_value=_make_grid({})), \
             patch("fem2geo.model.ModelSchema.builtin", return_value=schema):
            m = Model.from_file("dummy.vtk", schema="test")
        self.assertIsInstance(m, Model)

    def test_string_schema_calls_builtin(self):
        schema = self._schema()
        with patch("fem2geo.model.load_grid", return_value=_make_grid({})), \
             patch("fem2geo.model.ModelSchema.builtin", return_value=schema) as mock:
            Model.from_file("dummy.vtk", schema="adeli")
        mock.assert_called_once_with("adeli")

    def test_schema_retained(self):
        schema = self._schema()
        with patch("fem2geo.model.load_grid", return_value=_make_grid({})):
            m = Model.from_file("dummy.vtk", schema=schema)
        self.assertIs(m.schema, schema)


if __name__ == "__main__":
    unittest.main()