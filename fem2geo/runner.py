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
}


def _load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "job" not in cfg:
        raise ValueError(f"Job config '{path}' is missing required key 'job'.")
    return cfg


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

    cfg      = _load_config(job_path)
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