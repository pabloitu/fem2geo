import unittest
import tempfile
import os
import yaml

from fem2geo.internal.schema import ModelSchema, ScalarEntry, TensorEntry, SI_FACTORS


_ADELI = {
    "solver": "adeli",
    "units": {"pressure": "MPa", "distance": "m", "velocity": "cm/a"},
    "tensors": {
        "stress": {
            "components": {
                "xx": "Sxx", "yy": "Syy", "zz": "Szz",
                "xy": "Sxy", "yz": "Syz", "zx": "Szx",
            },
            "category": "pressure",
        },
    },
    "fields": {
        "dir_s1": {"field": "dir_DevStress_1"},
        "u":      {"field": "Displacement", "category": "distance"},
        "v":      {"field": "Velocity",     "category": "velocity"},
    },
}

_PACKED = {
    "solver": "adeli2",
    "units": {"pressure": "Pa"},
    "tensors": {
        "stress": {"voigt6": "stresses_(Pa)", "category": "pressure"},
        "strain": {"voigt6": "strains_(-)"},
    },
    "fields": {"val_s1": {"field": "sig1", "category": "pressure"}},
}


class TestSIFactors(unittest.TestCase):

    def test_ordering(self):
        self.assertLess(SI_FACTORS["pa"], SI_FACTORS["mpa"])
        self.assertLess(SI_FACTORS["mm"], SI_FACTORS["m"])
        self.assertLess(SI_FACTORS["m"], SI_FACTORS["km"])

    def test_all_lowercase(self):
        for k in SI_FACTORS:
            self.assertEqual(k, k.lower())


class TestScalarFields(unittest.TestCase):

    def setUp(self):
        self.s = ModelSchema.from_dict(_ADELI)

    def test_lookup_and_units(self):
        self.assertIn("u", self.s.fields)
        self.assertEqual(self.s.fields["u"].unit, "m")
        self.assertAlmostEqual(self.s.fields["u"].si_factor, 1.0)

    def test_no_category_gives_none(self):
        e = self.s.fields["dir_s1"]
        self.assertIsNone(e.category)
        self.assertIsNone(e.unit)

    def test_has_missing(self):
        self.assertNotIn("nope", self.s.fields)


class TestTensors(unittest.TestCase):

    def test_components(self):
        s = ModelSchema.from_dict(_ADELI)
        t = s.tensors["stress"]
        self.assertFalse(t.is_packed)
        self.assertEqual(t.components["xx"], "Sxx")
        self.assertEqual(t.unit, "MPa")

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

    def test_no_unit_gives_none(self):
        s = ModelSchema.from_dict(_PACKED)
        self.assertIsNone(s.tensors["strain"].unit)


class TestOverrides(unittest.TestCase):

    def test_unit_override_applies(self):
        s = ModelSchema.from_dict(_ADELI, units={"pressure": "Pa"})
        self.assertEqual(s.tensors["stress"].unit, "Pa")
        self.assertAlmostEqual(s.tensors["stress"].si_factor, 1.0)

    def test_override_leaves_others(self):
        s = ModelSchema.from_dict(_ADELI, units={"pressure": "Pa"})
        self.assertEqual(s.fields["u"].unit, "m")


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

    def test_unknown_unit(self):
        s = ModelSchema.from_dict({
            "units": {"pressure": "furlong"},
            "fields": {"f": {"field": "F", "category": "pressure"}},
        })
        self.assertEqual(s.fields["f"].unit, "furlong")
        self.assertIsNone(s.fields["f"].si_factor)

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