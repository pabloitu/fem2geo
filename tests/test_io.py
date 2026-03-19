import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from fem2geo.internal.io import load_grid, _normalize_dir
from fem2geo.internal.schema import ModelSchema


class _DictLike(dict):
    def get(self, key, default=None):
        return self[key] if key in self else default


def _grid(cell_data):
    g = MagicMock()
    g.array_names = list(cell_data.keys())
    g.cell_data = _DictLike(cell_data)
    g.point_data = _DictLike({})
    return g


def _load(cell_data, schema_dict):
    schema = ModelSchema.from_dict(schema_dict)
    g = _grid(cell_data)
    with patch("fem2geo.internal.io.pv.read", return_value=g):
        return load_grid("dummy.vtk", schema=schema)


class TestNormalize(unittest.TestCase):

    def test_scales_to_unit_and_preserves_direction(self):
        g = _grid({"d": np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])})
        _normalize_dir(g, "d")
        np.testing.assert_allclose(
            np.linalg.norm(g.cell_data["d"], axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(g.cell_data["d"][0], [1, 0, 0], atol=1e-12)

    def test_zero_vector_no_crash(self):
        g = _grid({"d": np.array([[0.0, 0.0, 0.0]])})
        _normalize_dir(g, "d")
        self.assertEqual(g.cell_data["d"].shape, (1, 3))


class TestScalarRenaming(unittest.TestCase):

    _SCHEMA = {
        "solver": "t", "units": {},
        "fields": {
            "dir_s1": {"field": "Dir1"},
            "t":      {"field": "Time"},
        },
    }

    def test_renames_and_removes_originals(self):
        g = _load(
            {"Dir1": np.array([[1, 0, 0]]), "Time": np.array([0.0])},
            self._SCHEMA)
        self.assertIn("dir_s1", g.cell_data)
        self.assertIn("t", g.cell_data)
        self.assertNotIn("Dir1", g.cell_data)
        self.assertNotIn("Time", g.cell_data)

    def test_dir_fields_normalized(self):
        g = _load({"Dir1": np.array([[3, 0, 0]])}, self._SCHEMA)
        np.testing.assert_allclose(
            np.linalg.norm(g.cell_data["dir_s1"], axis=1), 1.0, atol=1e-12)

    def test_missing_field_skipped_with_warning(self):
        with self.assertLogs("fem2geoLogger", level="WARNING"):
            g = _load({"Dir1": np.array([[1, 0, 0]])}, self._SCHEMA)
        self.assertNotIn("t", g.cell_data)

    def test_empty_file(self):
        g = _load({}, self._SCHEMA)
        self.assertEqual(len([k for k in g.cell_data if not k.startswith("_")]), 0)


class TestComponentTensorRenaming(unittest.TestCase):

    _SCHEMA = {
        "solver": "t",
        "tensors": {
            "stress": {
                "components": {
                    "xx": "Sxx", "yy": "Syy", "zz": "Szz",
                    "xy": "Sxy", "yz": "Syz", "zx": "Szx",
                },
            },
        },
        "fields": {},
    }

    def _data(self):
        return {k: np.array([float(i)]) for i, k in
                enumerate(["Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"])}

    def test_renames_all_components(self):
        g = _load(self._data(), self._SCHEMA)
        for c in ("xx", "yy", "zz", "xy", "yz", "zx"):
            self.assertIn(f"_tensor_stress_{c}", g.cell_data)
            self.assertNotIn({"xx": "Sxx", "yy": "Syy", "zz": "Szz",
                              "xy": "Sxy", "yz": "Syz", "zx": "Szx"}[c],
                             g.cell_data)

    def test_missing_component_skipped(self):
        with self.assertLogs("fem2geoLogger", level="WARNING"):
            g = _load({"Sxx": np.array([1.0])}, self._SCHEMA)
        self.assertIn("_tensor_stress_xx", g.cell_data)
        self.assertNotIn("_tensor_stress_yy", g.cell_data)


class TestVoigtTensorRenaming(unittest.TestCase):

    _SCHEMA = {
        "solver": "t",
        "tensors": {"stress": {"voigt6": "packed_s"}},
        "fields": {},
    }

    def test_renames_packed_array(self):
        packed = np.array([[1, 2, 3, .1, .2, .3]])
        g = _load({"packed_s": packed}, self._SCHEMA)
        self.assertIn("stress", g.cell_data)
        self.assertNotIn("packed_s", g.cell_data)
        np.testing.assert_array_equal(g.cell_data["stress"], packed)

    def test_missing_voigt_skipped(self):
        with self.assertLogs("fem2geoLogger", level="WARNING"):
            g = _load({}, self._SCHEMA)
        self.assertNotIn("stress", g.cell_data)


if __name__ == "__main__":
    unittest.main()