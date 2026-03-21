import argparse
import logging
import sys
from pathlib import Path

import yaml

from fem2geo.internal.logger import setup_logger

log = logging.getLogger("fem2geoLogger")

_JOBS = {
    "principal_directions": "fem2geo.jobs.principal_directions",
    "tendency_plot":        "fem2geo.jobs.tendency_plot",
    "fracture_analysis":    "fem2geo.jobs.fracture_analysis",
    "resolved_shear":       "fem2geo.jobs.resolved_shear",
    "kostrov_analysis":     "fem2geo.jobs.kostrov_analysis",
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "job" not in cfg:
        raise ValueError(
            f"Job config '{path}' is missing required key 'job'."
        )
    return cfg


def parse_config(cfg: dict, job_dir: Path) -> tuple:
    """
    Parse the shared top-level config keys.

    Returns the schema (constructed from the config) and the raw
    section dicts for zone, plot, and output. Creates the output
    directory if it doesn't exist.

    Parameters
    ----------
    cfg : dict
        Full job config as loaded from YAML.
    job_dir : Path
        Directory containing the config file (used as default
        output directory).

    Returns
    -------
    schema : ModelSchema
    zone : dict
    plot : dict
    out : dict
        Output config with ``dir`` resolved to an absolute Path.
    """
    from fem2geo.internal.schema import ModelSchema

    schema = ModelSchema.builtin(
        cfg.get("schema", "adeli"), units=cfg.get("units")
    )
    zone = cfg.get("zone", {})
    data = cfg.get("data", {})
    plot = cfg.get("plot", {})
    out = cfg.get("output", {})
    out["dir"] = Path(out.get("dir", job_dir)).resolve()
    out["dir"].mkdir(parents=True, exist_ok=True)
    return schema, zone, data, plot, out

def run(job_path: Path) -> None:
    """
    Load a job config file and dispatch to the appropriate job module.

    Parameters
    ----------
    job_path : Path
        Path to the job YAML file.

    Raises
    ------
    ValueError
        If the job type is unknown or the config is malformed.
    FileNotFoundError
        If the job file does not exist.
    """
    import importlib

    job_path = Path(job_path).resolve()
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {job_path}")

    cfg      = load_config(job_path)
    job_type = cfg["job"]

    if job_type not in _JOBS:
        raise ValueError(
            f"Unknown job type '{job_type}'. Available: {list(_JOBS)}"
        )

    log.info(f"Running job '{job_type}' from {job_path}")
    module = importlib.import_module(_JOBS[job_type])
    module.run(cfg, job_path.parent)


def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser(
        prog="fem2geo",
        description="Run a fem2geo analysis job from a YAML config file.",
    )
    parser.add_argument("job", type=Path, help="Path to job YAML file.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    if args.verbose:
        from fem2geo.internal.logger import set_console_log_level
        set_console_log_level(logging.DEBUG)

    try:
        run(args.job)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
    except Exception as e:
        log.error(f"Job failed: {e}")
        raise
