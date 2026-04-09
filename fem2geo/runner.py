import argparse
import logging
import sys
from pathlib import Path

import yaml

from fem2geo.internal.logger import setup_logger

log = logging.getLogger("fem2geoLogger")

_JOBS = {
    "principal_directions": "fem2geo.jobs.principal_directions",
    "tendency":             "fem2geo.jobs.tendency",
    "fracture":             "fem2geo.jobs.fracture",
    "resolved_shear":       "fem2geo.jobs.resolved_shear",
    "kostrov":              "fem2geo.jobs.kostrov",
    "project":              "fem2geo.jobs.project",
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "job" not in cfg:
        raise ValueError(
            f"Job config '{path}' is missing required key 'job'."
        )
    return cfg


def resolve_output(cfg: dict, job_dir: Path) -> dict:
    """
    Resolve the output directory and create it.

    Returns the ``output`` block with ``dir`` resolved to an absolute
    Path. Creates the directory if it doesn't exist.

    Parameters
    ----------
    cfg : dict
        Full job config as loaded from YAML.
    job_dir : Path
        Directory containing the config file (used as default
        output directory).

    Returns
    -------
    dict
        The output config with ``dir`` set to an absolute Path.
    """
    out = dict(cfg.get("output", {}))
    out["dir"] = Path(out.get("dir", job_dir)).resolve()
    out["dir"].mkdir(parents=True, exist_ok=True)
    return out


def run(job_path: Path, output_dir: Path = None) -> None:
    """
    Load a job config file and dispatch to the appropriate job module.

    Parameters
    ----------
    job_path : Path
        Path to the job YAML file.
    output_dir : Path, optional
        Override the output directory from the config. Useful for testing.

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
        raise ValueError(f"Unknown job type '{job_type}'. Available: {list(_JOBS)}")

    if output_dir is not None:
        cfg.setdefault("output", {})["dir"] = str(output_dir)

    log.info(f"Running job '{job_type}' from {job_path}")
    module = importlib.import_module(_JOBS[job_type])
    module.run(cfg, job_path.parent)


def main() -> None:
    setup_logger()

    if len(sys.argv) > 1 and sys.argv[1] == "download-tutorials":
        from fem2geo.internal.tutorials import run_download
        sys.exit(run_download() or 0)

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