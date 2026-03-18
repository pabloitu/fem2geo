import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from fem2geo.internal.io import load_grid, _unit_vectors
from fem2geo.internal.schema import ModelSchema


_SCHEMA_DICT = {
    "solver": "test",
    "units": {"pressure": "MPa"},
    "fields": {
        "s_xx":  {"field": "Stress_xx", "category": "pressure"},
        "dir_s1": {"field": "Dir1"},
        "t":     {"field": "Time"},
    },
}


def _make_grid(cell_data: dict):
    grid = MagicMock()
    grid.array_names = list(cell_data.keys())
    grid.cell_data = _DictLike(cell_data)
    grid.point_data = _DictLike({})
    return grid


class _DictLike(dict):
    def get(self, key, default=None):
        return self[key] if key in self else default


class TestUnitVectors(unittest.TestCase):

    def test_unit_length(self):
        arr = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
        out = _unit_vectors(arr)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_direction_preserved(self):
        arr = np.array([[2.0, 0.0, 0.0]])
        out = _unit_vectors(arr)
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_zero_vector_no_crash(self):
        arr = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = _unit_vectors(arr)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(out[1]), 1.0, atol=1e-12)


class TestLoadModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schema = ModelSchema.from_dict(_SCHEMA_DICT)

    def _run(self, cell_data):
        grid = _make_grid(cell_data)
        with patch("fem2geo.internal.io.pv.read", return_value=grid):
            return load_grid("dummy.vtk", schema=self.schema)

    def test_canonical_names_present(self):
        grid = self._run({
            "Stress_xx": np.array([1.0, 2.0]),
            "Dir1":      np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "Time":      np.array([0.0, 1.0]),
        })
        self.assertIn("s_xx",  grid.cell_data)
        self.assertIn("dir_s1", grid.cell_data)
        self.assertIn("t",     grid.cell_data)

    def test_solver_names_removed(self):
        grid = self._run({
            "Stress_xx": np.array([1.0]),
            "Dir1":      np.array([[1.0, 0.0, 0.0]]),
            "Time":      np.array([0.0]),
        })
        self.assertNotIn("Stress_xx", grid.cell_data)
        self.assertNotIn("Dir1",      grid.cell_data)
        self.assertNotIn("Time",      grid.cell_data)

    def test_direction_normalized(self):
        grid = self._run({
            "Stress_xx": np.array([1.0]),
            "Dir1":      np.array([[3.0, 0.0, 0.0]]),
            "Time":      np.array([0.0]),
        })
        np.testing.assert_allclose(
            np.linalg.norm(grid.cell_data["dir_s1"], axis=1), 1.0, atol=1e-12
        )

    def test_missing_field_skipped(self):
        grid = self._run({"Stress_xx": np.array([1.0])})
        self.assertIn("s_xx", grid.cell_data)
        self.assertNotIn("dir_s1", grid.cell_data)
        self.assertNotIn("t",      grid.cell_data)

    def test_missing_field_warning(self):
        with self.assertLogs("fem2geoLogger", level="WARNING") as cm:
            self._run({"Stress_xx": np.array([1.0])})
        warnings = [m for m in cm.output if "WARNING" in m]
        self.assertTrue(any("Dir1" in m or "Time" in m for m in warnings))

    def test_string_schema_name_resolved(self):
        grid = _make_grid({"Stress_xx": np.array([1.0])})
        with patch("fem2geo.internal.io.pv.read", return_value=grid), \
             patch("fem2geo.internal.io.ModelSchema.builtin", return_value=self.schema) as mock:
            load_grid("dummy.vtk", schema="adeli")
            mock.assert_called_once_with("adeli")

    def test_empty_file_no_crash(self):
        grid = self._run({})
        self.assertEqual(len(grid.cell_data), 0)


if __name__ == "__main__":
    unittest.main()