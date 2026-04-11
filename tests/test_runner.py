"""
Tests for fem2geo.runner

Covers load_config, resolve_output, run() dispatching, and error paths.
Does not test main() CLI — that would require subprocess invocation.
"""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

from fem2geo.runner import load_config, resolve_output, run


class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, data):
        p = self.tmp / name
        p.write_text(yaml.safe_dump(data))
        return p

    def test_basic_load(self):
        p = self._write("c.yaml", {"job": "principal_directions", "schema": "adeli"})
        cfg = load_config(p)
        self.assertEqual(cfg["job"], "principal_directions")
        self.assertEqual(cfg["schema"], "adeli")

    def test_missing_job_key_raises(self):
        p = self._write("c.yaml", {"schema": "adeli"})
        with self.assertRaisesRegex(ValueError, "missing required key 'job'"):
            load_config(p)

    def test_preserves_nested_structure(self):
        p = self._write("c.yaml", {
            "job": "principal_directions",
            "site": {"center": [1, 2, 3], "radius": 10},
        })
        cfg = load_config(p)
        self.assertEqual(cfg["site"]["center"], [1, 2, 3])
        self.assertEqual(cfg["site"]["radius"], 10)


class TestResolveOutput(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_dir_is_job_dir(self):
        out = resolve_output({}, self.tmp)
        self.assertEqual(out["dir"], self.tmp.resolve())

    def test_explicit_dir_resolved(self):
        cfg = {"output": {"dir": str(self.tmp / "results")}}
        out = resolve_output(cfg, self.tmp)
        self.assertEqual(out["dir"], (self.tmp / "results").resolve())
        self.assertTrue(out["dir"].is_dir())

    def test_creates_missing_directory(self):
        target = self.tmp / "new_dir"
        self.assertFalse(target.exists())
        resolve_output({"output": {"dir": str(target)}}, self.tmp)
        self.assertTrue(target.is_dir())

    def test_returns_dict_with_other_keys(self):
        cfg = {"output": {"dir": str(self.tmp), "figure": "plot.png"}}
        out = resolve_output(cfg, self.tmp)
        self.assertEqual(out["figure"], "plot.png")

    def test_does_not_mutate_input(self):
        cfg = {"output": {"dir": str(self.tmp)}}
        original = cfg["output"].copy()
        resolve_output(cfg, self.tmp)
        self.assertEqual(cfg["output"], original)


class TestRun(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _cfg(self, data):
        p = self.tmp / "config.yaml"
        p.write_text(yaml.safe_dump(data))
        return p

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            run(self.tmp / "does_not_exist.yaml")

    def test_unknown_job_raises(self):
        p = self._cfg({"job": "bogus_job"})
        with self.assertRaisesRegex(ValueError, "Unknown job type"):
            run(p)

    def test_dispatches_to_module(self):
        p = self._cfg({"job": "principal_directions"})
        fake = MagicMock()
        with patch("importlib.import_module", return_value=fake) as im:
            run(p)
        im.assert_called_once_with("fem2geo.jobs.principal_directions")
        fake.run.assert_called_once()
        passed_cfg, passed_dir = fake.run.call_args[0]
        self.assertEqual(passed_cfg["job"], "principal_directions")
        self.assertEqual(passed_dir, self.tmp.resolve())

    def test_output_dir_override(self):
        p = self._cfg({"job": "principal_directions"})
        override = self.tmp / "override"
        fake = MagicMock()
        with patch("importlib.import_module", return_value=fake):
            run(p, output_dir=override)
        passed_cfg, _ = fake.run.call_args[0]
        self.assertEqual(passed_cfg["output"]["dir"], str(override))

    def test_sites_dispatch(self):
        p = self._cfg({"job": "sites.principal_directions"})
        fake = MagicMock()
        with patch("importlib.import_module", return_value=fake) as im:
            run(p)
        im.assert_called_once_with("fem2geo.jobs.sites")
        fake.run.assert_called_once()

    def test_all_registered_jobs_resolvable(self):
        """Every entry in _JOBS should dispatch without an unknown-job error."""
        from fem2geo.runner import _JOBS
        for job_type in _JOBS:
            with self.subTest(job=job_type):
                p = self._cfg({"job": job_type})
                fake = MagicMock()
                with patch("importlib.import_module", return_value=fake):
                    run(p)


if __name__ == "__main__":
    unittest.main()