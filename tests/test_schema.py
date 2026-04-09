import unittest
import tempfile
import os
import yaml

from fem2geo.internal.schema import ModelSchema


_ADELI = {
    "solver": "adeli",
    "tensors": {
        "stress": {
            "components": {
                "xx": "Sxx", "yy": "Syy", "zz": "Szz",
                "xy": "Sxy", "yz": "Syz", "zx": "Szx",
            },
        },
    },
    "fields": {
        "dir_s1": {"field": "dir_DevStress_1"},
        "u":      {"field": "Displacement"},
        "v":      {"field": "Velocity"},
    },
}

_PACKED = {
    "solver": "adeli2",
    "tensors": {
        "stress": {"voigt6": "stresses_(Pa)"},
        "strain": {"voigt6": "strains_(-)"},
    },
    "fields": {"val_s1": {"field": "sig1"}},
}


class TestScalarFields(unittest.TestCase):

    def setUp(self):
        self.s = ModelSchema.from_dict(_ADELI)

    def test_lookup(self):
        self.assertIn("u", self.s.fields)
        self.assertEqual(self.s.fields["u"].solver_key, "Displacement")

    def test_canonical_set(self):
        e = self.s.fields["dir_s1"]
        self.assertEqual(e.canonical, "dir_s1")
        self.assertEqual(e.solver_key, "dir_DevStress_1")

    def test_missing(self):
        self.assertNotIn("nope", self.s.fields)


class TestTensors(unittest.TestCase):

    def test_components(self):
        s = ModelSchema.from_dict(_ADELI)
        t = s.tensors["stress"]
        self.assertFalse(t.is_packed)
        self.assertEqual(t.components["xx"], "Sxx")

    def test_voigt(self):
        s = ModelSchema.from_dict(_PACKED)
        t = s.tensors["stress"]
        self.assertTrue(t.is_packed)
        self.assertEqual(t.voigt6, "stresses_(Pa)")

    def test_multiple(self):
        s = ModelSchema.from_dict(_PACKED)
        self.assertIn("stress", s.tensors)
        self.assertIn("strain", s.tensors)
        self.assertNotIn("bogus", s.tensors)


class TestYamlRoundtrip(unittest.TestCase):

    def test_from_yaml_matches_dict(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
        yaml.dump(_ADELI, tmp); tmp.close()
        try:
            y = ModelSchema.from_yaml(tmp.name)
        finally:
            os.unlink(tmp.name)
        d = ModelSchema.from_dict(_ADELI)
        self.assertEqual(set(d.fields), set(y.fields))
        self.assertEqual(set(d.tensors), set(y.tensors))


class TestEdgeCases(unittest.TestCase):

    def test_empty(self):
        s = ModelSchema.from_dict({"fields": {}})
        self.assertEqual(s.name, "custom")
        self.assertEqual(len(s.fields), 0)
        self.assertEqual(len(s.tensors), 0)

    def test_builtin_bad_name(self):
        with self.assertRaises(ValueError) as ctx:
            ModelSchema.builtin("nope")
        self.assertIn("Available", str(ctx.exception))

    def test_missing_yaml(self):
        with self.assertRaises(FileNotFoundError):
            ModelSchema.from_yaml("/tmp/__does_not_exist__.yaml")


class TestAdeliBuiltin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.s = ModelSchema.builtin("adeli")
        except ValueError:
            cls.s = None

    def setUp(self):
        if self.s is None:
            self.skipTest("adeli.yaml not found")

    def test_loads_with_tensor_and_fields(self):
        self.assertEqual(self.s.name, "adeli")
        self.assertIn("stress", self.s.tensors)
        for f in ("dir_s1", "dir_s3", "val_s1", "val_s3", "u"):
            self.assertIn(f, self.s.fields, msg=f)


if __name__ == "__main__":
    unittest.main()