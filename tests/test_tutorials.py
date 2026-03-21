"""
Integration test: run every config.yaml found in tutorials/{i}_{name}/.

Discovers tutorial directories by matching the pattern ``{int}_{name}``,
collects all ``*.yaml`` files within each, and runs them through
:func:`fem2geo.runner.run`. Skips gracefully if the tutorials directory
is missing or a tutorial has no YAML files.

Usage
-----
pytest tests/test_tutorials.py -v
pytest tests/test_tutorials.py -v -k "1_probing"   # run one tutorial only
"""

import re
import pytest
from pathlib import Path

from fem2geo.runner import run

_TUTORIALS_DIR = Path(__file__).parent.parent / "tutorials"

_DIR_PATTERN = re.compile(r"^\d+_.+$")


def _discover_configs():
    """Yield (tutorial_name, config_path) for every YAML in every tutorial."""
    if not _TUTORIALS_DIR.is_dir():
        return

    for d in sorted(_TUTORIALS_DIR.iterdir()):
        if not d.is_dir() or not _DIR_PATTERN.match(d.name):
            continue

        for cfg_path in sorted(d.glob("**/*.yaml")):
            rel = cfg_path.relative_to(_TUTORIALS_DIR).with_suffix("")
            yield str(rel), cfg_path


_CONFIGS = list(_discover_configs())


@pytest.mark.skipif(
    not _TUTORIALS_DIR.is_dir(),
    reason=f"Tutorials directory not found: {_TUTORIALS_DIR}",
)
@pytest.mark.skipif(
    len(_CONFIGS) == 0,
    reason="No tutorial configs discovered.",
)
@pytest.mark.parametrize("name,config_path", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_tutorial(name, config_path, tmp_path):
    """Run a single tutorial config and verify it completes without error."""
    run(config_path, output_dir=tmp_path)