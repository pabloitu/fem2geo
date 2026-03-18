import unittest
import tempfile
import os
import yaml

from fem2geo.internal.schema import ModelSchema, FieldEntry, SI_FACTORS


_ADELI = {
    "solver": "adeli",
    "units": {
        "pressure": "MPa",
        "distance": "m",
        "velocity": "cm/a",
        "time":     "Ma",
    },
    "fields": {
        "s_xx":         {"field": "Stress_xx",       "category": "pressure"},
        "dir_s1":       {"field": "dir_DevStress_1"},
        "displacement": {"field": "Displacement",    "category": "distance"},
        "velocity":     {"field": "Velocity",        "category": "velocity"},
        "time":         {"field": "TIME",            "category": "time"},
    },
}


def _yaml_file(d):
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    yaml.dump(d, tmp)
    tmp.close()
    return tmp.name


class TestSIFactors(unittest.TestCase):

    def test_pressure_ordering(self):
        self.assertLess(SI_FACTORS["pa"], SI_FACTORS["kpa"])
        self.assertLess(SI_FACTORS["kpa"], SI_FACTORS["mpa"])
        self.assertLess(SI_FACTORS["mpa"], SI_FACTORS["gpa"])

    def test_length_ordering(self):
        self.assertLess(SI_FACTORS["mm"], SI_FACTORS["cm"])
        self.assertLess(SI_FACTORS["cm"], SI_FACTORS["m"])
        self.assertLess(SI_FACTORS["m"],  SI_FACTORS["km"])

    def test_geological_rates(self):
        self.assertGreater(SI_FACTORS["cm/a"], 0.0)
        self.assertLess(SI_FACTORS["cm/a"], 1e-7)
        self.assertAlmostEqual(SI_FACTORS["ma"] / 3.156e13, 1.0, places=2)

    def test_keys_are_lowercase(self):
        for k in SI_FACTORS:
            self.assertEqual(k, k.lower())


class TestModelSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schema = ModelSchema.from_dict(_ADELI)

    def test_name(self):
        self.assertEqual(self.schema.name, "adeli")

    def test_solver_key(self):
        self.assertEqual(self.schema.solver_key("s_xx"), "Stress_xx")
        self.assertEqual(self.schema.solver_key("dir_s1"), "dir_DevStress_1")

    def test_category_resolved(self):
        self.assertEqual(self.schema.fields["s_xx"].category, "pressure")
        self.assertIsNone(self.schema.fields["dir_s1"].category)

    def test_unit_resolved_from_category(self):
        self.assertEqual(self.schema.fields["s_xx"].unit, "MPa")
        self.assertIsNone(self.schema.fields["dir_s1"].unit)

    def test_si_factor(self):
        self.assertAlmostEqual(self.schema.si_factor("s_xx"), 1e6, places=0)
        self.assertAlmostEqual(self.schema.si_factor("displacement"), 1.0, places=10)

    def test_si_factor_no_category_returns_one(self):
        self.assertAlmostEqual(self.schema.si_factor("dir_s1"), 1.0, places=10)

    def test_si_factor_unknown_canonical_returns_one(self):
        self.assertAlmostEqual(self.schema.si_factor("not_a_field"), 1.0, places=10)

    def test_has(self):
        self.assertTrue(self.schema.has("s_xx"))
        self.assertFalse(self.schema.has("not_a_field"))

    def test_solver_key_missing_raises(self):
        with self.assertRaises(KeyError):
            self.schema.solver_key("not_a_field")

    def test_unit_override(self):
        schema = ModelSchema.from_dict(_ADELI, units={"pressure": "Pa"})
        self.assertEqual(schema.fields["s_xx"].unit, "Pa")
        self.assertAlmostEqual(schema.si_factor("s_xx"), 1.0, places=10)
        self.assertAlmostEqual(schema.si_factor("displacement"), 1.0, places=10)

    def test_partial_override_leaves_others_unchanged(self):
        schema = ModelSchema.from_dict(_ADELI, units={"pressure": "Pa"})
        self.assertAlmostEqual(schema.si_factor("displacement"), 1.0, places=10)
        f = schema.si_factor("velocity")
        self.assertGreater(f, 0.0)
        self.assertLess(f, 1e-7)

    def test_from_yaml_matches_from_dict(self):
        path = _yaml_file(_ADELI)
        try:
            from_yaml = ModelSchema.from_yaml(path)
        finally:
            os.unlink(path)
        for canonical in _ADELI["fields"]:
            self.assertEqual(self.schema.solver_key(canonical),
                             from_yaml.solver_key(canonical))
            self.assertAlmostEqual(self.schema.si_factor(canonical),
                                   from_yaml.si_factor(canonical), places=10)


class TestEdgeCases(unittest.TestCase):

    def test_empty_schema(self):
        s = ModelSchema.from_dict({"solver": "empty", "fields": {}})
        self.assertEqual(len(s.fields), 0)

    def test_missing_solver_name_defaults(self):
        s = ModelSchema.from_dict({"fields": {"s_xx": {"field": "SigXX",
                                                        "category": "pressure"}}})
        self.assertEqual(s.name, "custom")

    def test_no_units_block_gives_none(self):
        s = ModelSchema.from_dict({"solver": "x",
                                   "fields": {"s_xx": {"field": "F",
                                                        "category": "pressure"}}})
        self.assertIsNone(s.fields["s_xx"].unit)
        self.assertAlmostEqual(s.si_factor("s_xx"), 1.0, places=10)

    def test_unknown_unit_si_factor_is_one(self):
        s = ModelSchema.from_dict({"solver": "x",
                                   "units": {"pressure": "furlong"},
                                   "fields": {"s_xx": {"field": "F",
                                                        "category": "pressure"}}})
        self.assertEqual(s.fields["s_xx"].unit, "furlong")
        self.assertIsNone(s.fields["s_xx"].si_factor)
        self.assertAlmostEqual(s.si_factor("s_xx"), 1.0, places=10)

    def test_missing_yaml_raises(self):
        with self.assertRaises(FileNotFoundError):
            ModelSchema.from_yaml("/tmp/does_not_exist_fem2geo.yaml")

    def test_builtin_bad_name_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            ModelSchema.builtin("not_a_solver")
        self.assertIn("Available", str(ctx.exception))


class TestAdeliBuiltin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.schema = ModelSchema.builtin("adeli")
        except ValueError:
            cls.schema = None

    def setUp(self):
        if self.schema is None:
            self.skipTest("adeli.yaml not found")

    def test_loads(self):
        self.assertEqual(self.schema.name, "adeli")

    def test_expected_fields_present(self):
        for f in ("s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_zx",
                  "dir_s1", "dir_s2", "dir_s3", "val_s1", "val_s3",
                  "u", "v", "t"):
            self.assertTrue(self.schema.has(f), msg=f"missing field: {f}")

    def test_pressure_unit(self):
        self.assertEqual(self.schema.fields["s_xx"].unit, "MPa")
        self.assertAlmostEqual(self.schema.si_factor("s_xx"), 1e6, places=0)

    def test_unit_override(self):
        s = ModelSchema.builtin("adeli", units={"pressure": "Pa"})
        self.assertAlmostEqual(s.si_factor("s_xx"), 1.0, places=10)
        self.assertAlmostEqual(s.si_factor("u"), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()