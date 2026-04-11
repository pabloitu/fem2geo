"""
Tests for fem2geo.plots helpers that don't require actual rendering.

Focuses on get_style() merging logic and _unpack_args() input handling —
the pure-data helpers. Rendering functions are tested indirectly via
test_tutorials.py.
"""

import unittest

import numpy as np

from fem2geo.plots import get_style, _unpack_args, MODEL_COLORS


class TestGetStyle(unittest.TestCase):

    def test_default_only(self):
        out = get_style({"color": "red", "markersize": 8})
        self.assertEqual(out, {"color": "red", "markersize": 8})

    def test_single_override(self):
        out = get_style({"color": "red", "markersize": 8},
                        {"color": "blue"})
        self.assertEqual(out["color"], "blue")
        self.assertEqual(out["markersize"], 8)

    def test_kwargs_override_everything(self):
        out = get_style({"color": "red"}, {"color": "blue"}, color="green")
        self.assertEqual(out["color"], "green")

    def test_drops_show_and_style_by_default(self):
        out = get_style({"color": "red"}, {"show": True, "style": "scatter"})
        self.assertNotIn("show", out)
        self.assertNotIn("style", out)
        self.assertEqual(out["color"], "red")

    def test_custom_drop_list(self):
        out = get_style({"color": "red", "foo": 1}, {"bar": 2},
                        drop=("foo", "bar"))
        self.assertNotIn("foo", out)
        self.assertNotIn("bar", out)

    def test_does_not_mutate_default(self):
        default = {"color": "red", "markersize": 8}
        get_style(default, {"color": "blue"})
        self.assertEqual(default, {"color": "red", "markersize": 8})

    def test_multiple_overrides_merged_in_order(self):
        out = get_style(
            {"color": "red"},
            {"color": "blue", "marker": "o"},
            {"marker": "s"},
        )
        self.assertEqual(out["color"], "blue")
        self.assertEqual(out["marker"], "s")

    def test_empty_default(self):
        out = get_style({}, {"color": "green"})
        self.assertEqual(out, {"color": "green"})


class TestUnpackArgs(unittest.TestCase):

    def test_two_separate_arrays(self):
        a, b = _unpack_args([1, 2, 3], [4, 5, 6], ("plunge", "azimuth"))
        np.testing.assert_array_equal(a, [1, 2, 3])
        np.testing.assert_array_equal(b, [4, 5, 6])

    def test_packed_2d_array(self):
        a, b = _unpack_args([[10, 30], [20, 60]], None,
                            ("strike", "dip"))
        np.testing.assert_array_equal(a, [10, 20])
        np.testing.assert_array_equal(b, [30, 60])

    def test_packed_single_pair(self):
        a, b = _unpack_args([[10, 30]], None, ("strike", "dip"))
        np.testing.assert_array_equal(a, [10])
        np.testing.assert_array_equal(b, [30])

    def test_scalar_first_with_no_second_raises(self):
        with self.assertRaises(ValueError):
            _unpack_args(10, None, ("strike", "dip"))

    def test_scalar_pair(self):
        a, b = _unpack_args(10, 30, ("strike", "dip"))
        np.testing.assert_array_equal(a, [10])
        np.testing.assert_array_equal(b, [30])


class TestModelColors(unittest.TestCase):

    def test_has_enough_colors(self):
        self.assertGreaterEqual(len(MODEL_COLORS), 4)

    def test_all_are_hex_strings(self):
        for c in MODEL_COLORS:
            self.assertTrue(c.startswith("#"))
            self.assertEqual(len(c), 7)


if __name__ == "__main__":
    unittest.main()