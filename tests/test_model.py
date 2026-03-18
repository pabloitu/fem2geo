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

_ALL_FIELDS = {**_STRESS_FIELDS, **_DIR_FIELDS}


def _make_grid(cell_data: dict, points: np.ndarray = None):
    if points is None:
        points = np.array([
            [0., 0., -1.], [1., 0., -1.],
            [1., 1., -2.], [0., 1., -2.],
        ])
    grid = MagicMock()
    grid.points = points
    grid.number_of_cells = _N
    grid.array_names = list(cell_data.keys())
    grid.cell_data = _DictLike(cell_data)
    grid.point_data = _DictLike({})
    grid.cells_dict = {10: np.array([[0, 1, 2, 3]] * _N)}
    grid.celltypes = np.array([10] * _N)

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
        "fields": {k: {"field": k, "category": "pressure"}
                   for k in _STRESS_FIELDS} | {
            "dir_s1": {"field": "dir_s1"},
            "dir_s3": {"field": "dir_s3"},
        }
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

    def test_points_shape(self):
        self.assertEqual(self.m.points.shape, (_N, 3))

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

    def test_val_s2_between_s1_and_s3(self):
        self.assertTrue(np.all(self.m.val_s1 <= self.m.val_s2))
        self.assertTrue(np.all(self.m.val_s2 <= self.m.val_s3))

    def test_dir_s2_from_file_used_when_present(self):
        s2 = np.array([[0., 1., 0.]] * _N)
        m = _make_model({**_STRESS_FIELDS, **_DIR_FIELDS, "dir_s2": s2})
        np.testing.assert_allclose(m.dir_s2, s2)


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


class TestFromFile(unittest.TestCase):

    def test_from_file_returns_model(self):
        schema = ModelSchema.from_dict({
            "solver": "test", "units": {}, "fields": {}
        })
        grid = _make_grid({})
        with patch("fem2geo.model.load_grid", return_value=grid), \
             patch("fem2geo.model.ModelSchema.builtin", return_value=schema):
            m = Model.from_file("dummy.vtk", schema="test")
        self.assertIsInstance(m, Model)

    def test_from_file_string_schema_calls_builtin(self):
        schema = ModelSchema.from_dict({
            "solver": "test", "units": {}, "fields": {}
        })
        grid = _make_grid({})
        with patch("fem2geo.model.load_grid", return_value=grid), \
             patch("fem2geo.model.ModelSchema.builtin", return_value=schema) as mock:
            Model.from_file("dummy.vtk", schema="adeli")
        mock.assert_called_once_with("adeli")

    def test_schema_retained(self):
        schema = ModelSchema.from_dict({
            "solver": "test", "units": {}, "fields": {}
        })
        grid = _make_grid({})
        with patch("fem2geo.model.load_grid", return_value=grid):
            m = Model.from_file("dummy.vtk", schema=schema)
        self.assertIs(m.schema, schema)


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_dir = Path(__file__).parent / "data"
        vtk_files = list(data_dir.glob("*.vt*")) if data_dir.exists() else []
        cls.path = vtk_files[0] if vtk_files else None

    def setUp(self):
        if self.path is None:
            self.skipTest("No fixture file found in tests/data/")

    def test_loads_without_error(self):
        m = Model.from_file(self.path)
        self.assertIsInstance(m, Model)

    def test_stress_shape(self):
        m = Model.from_file(self.path)
        self.assertEqual(m.stress.ndim, 3)
        self.assertEqual(m.stress.shape[1:], (3, 3))


if __name__ == "__main__":
    unittest.main()