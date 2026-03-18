"""
Job: tendency_plot
==================
Plots slip and/or dilation tendency contours on a stereonet for one model
at one extraction zone. Average stress principal directions are overlaid by
default. Individual cell directions can optionally be shown to visualise
the spread around the average.

Config reference
----------------
job: tendency_plot
schema: adeli               # built-in schema name (default: adeli)
units:                      # optional category-level unit overrides
  pressure: Pa

model: path/to/model.vtk    # relative to this config file

zone:
  type: sphere              # sphere | box
  center: [x, y, z]
  radius: r                 # sphere only
  # dim: [dx, dy, dz]       # box only

plot:
  tendency: both            # slip | dilation | both (default: both)
  avg_directions:           # average stress directions (default: show=true)
    show: true
    s1: { color: "white", marker: "o", markersize: 8 }
    s2: { color: "white", marker: "s", markersize: 8 }
    s3: { color: "white", marker: "v", markersize: 8 }
  cell_directions:          # per-cell directions to show spread (default: show=false)
    show: false
    s1: { color: "r", marker: "o" }
    s2: { color: "g", marker: "s" }
    s3: { color: "b", marker: "v" }

output:
  dir: results/             # optional, defaults to config file directory
  save_vtu: false           # save extracted sub-model for Paraview

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem2geo import plots
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.utils import transform as tr

log = logging.getLogger("fem2geoLogger")


def _show(direction_cfg, default: bool) -> bool:
    if isinstance(direction_cfg, bool):
        return direction_cfg
    return direction_cfg.get("show", default)


def _style(direction_cfg, key: str, defaults: dict) -> dict:
    if not isinstance(direction_cfg, dict):
        return defaults[key]
    return plots._resolve_style(direction_cfg.get(key, {}), defaults[key])


def run(cfg: dict, job_dir: Path) -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"), units=cfg.get("units"))
    plot_cfg = cfg.get("plot", {})
    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", job_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    tendency = plot_cfg.get("tendency", "both")
    save_vtu = out_cfg.get("save_vtu", False)

    avg_cfg = plot_cfg.get("avg_directions", True)
    cell_cfg = plot_cfg.get("cell_directions", False)

    show_avg = _show(avg_cfg, default=True)
    show_cell = _show(cell_cfg, default=False)

    if tendency not in ("slip", "dilation", "both"):
        raise ValueError(f"plot.tendency must be slip | dilation | both, got '{tendency}'.")
    if "model" not in cfg:
        raise ValueError("tendency_plot requires a 'model' key.")

    # ── Load and extract ──────────────────────────────────────────────────────
    path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {path}")
    full_model = Model.from_file(path, schema)

    zone_cfg = cfg["zone"]
    if zone_cfg["type"] == "sphere":
        sub = full_model.extract_sphere(zone_cfg["center"], zone_cfg["radius"])
    elif zone_cfg["type"] == "box":
        sub = full_model.extract_box(zone_cfg["center"], np.asarray(zone_cfg["dim"]))
    else:
        raise ValueError(f"Unknown zone type '{zone_cfg['type']}'.")

    log.info(f"  {sub.n_cells} cells in zone")

    if save_vtu:
        sub.save(out_dir / "extract.vtu")

    # ── Average stress and principal directions ───────────────────────────────
    avg_stress = sub.avg_dev_stress()
    val, vec = np.linalg.eigh(avg_stress)
    order = np.argsort(val)
    val, vec = val[order], vec[:, order]
    phi = float((val[1] - val[2]) / (val[0] - val[2]))

    s1_avg = tr.line_enu2sphe(vec[:, 0])
    s2_avg = tr.line_enu2sphe(vec[:, 1])
    s3_avg = tr.line_enu2sphe(vec[:, 2])

    # ── Figure and axes ───────────────────────────────────────────────────────
    figsize = (16, 7) if tendency == "both" else (8, 8)
    fig = plt.figure(figsize=figsize)

    if tendency == "both":
        ax_slip = fig.add_subplot(121, projection="stereonet")
        ax_dil = fig.add_subplot(122, projection="stereonet")
        axes = {"slip": ax_slip, "dilation": ax_dil}
    else:
        axes = {tendency: fig.add_subplot(111, projection="stereonet")}

    # ── Tendency contours ─────────────────────────────────────────────────────
    for kind, ax in axes.items():
        ax.grid(True)
        plots.stereo_contour(ax, avg_stress, kind=kind)

    # ── Cell directions (spread) ──────────────────────────────────────────────
    if show_cell:
        s1_cell = np.array([tr.line_enu2sphe(v) for v in sub.dir_s1])
        s2_cell = np.array([tr.line_enu2sphe(v) for v in sub.dir_s2])
        s3_cell = np.array([tr.line_enu2sphe(v) for v in sub.dir_s3])

        for ax in axes.values():
            plots.stereo_line(ax, s1_cell, label=r"$\sigma_1$",
                              **_style(cell_cfg, "s1", plots._CELL_DEFAULTS))
            plots.stereo_line(ax, s2_cell, label=r"$\sigma_2$",
                              **_style(cell_cfg, "s2", plots._CELL_DEFAULTS))
            plots.stereo_line(ax, s3_cell, label=r"$\sigma_3$",
                              **_style(cell_cfg, "s3", plots._CELL_DEFAULTS))

    # ── Average directions ────────────────────────────────────────────────────
    if show_avg:
        for ax in axes.values():
            plots.stereo_line(ax, s1_avg, label=r"Avg $\sigma_1$",
                              **_style(avg_cfg, "s1", plots._LINE_DEFAULTS))
            plots.stereo_line(ax, s2_avg, label=r"Avg $\sigma_2$",
                              **_style(avg_cfg, "s2", plots._LINE_DEFAULTS))
            plots.stereo_line(ax, s3_avg, label=r"Avg $\sigma_3$",
                              **_style(avg_cfg, "s3", plots._LINE_DEFAULTS))

    # ── Titles and legends ────────────────────────────────────────────────────
    titles = {"slip": "Slip tendency", "dilation": "Dilation tendency"}
    suffix = f"\n$\\sigma_1={val[0]:.3f}$, $\\sigma_3={val[2]:.3f}$, $\\phi={phi:.2f}$"

    for kind, ax in axes.items():
        ax.set_title(titles[kind] + suffix, y=1.05)
        ax.legend(fontsize=7)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / "tendency_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")
