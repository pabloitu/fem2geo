"""
Job: fracture_analysis
======================
Compares fracture (joint, vein, dyke) orientation measurements with the stress
state predicted by a FEM model at a given extraction zone. Plots fracture poles
and model principal directions together on a stereonet.

Structural data is read from CSV files via
:func:`fem2geo.internal.io.load_structural_csv`. Only ``strike, dip`` columns
are supported (:class:`FractureData`). Fault data (``strike, dip, rake``) is
skipped — use the ``wallace_bott`` job for fault analysis.

Multiple datasets can be provided — each gets a distinct colour and legend
entry. The model's average σ1/σ2/σ3 directions are overlaid with the standard
circle/square/triangle markers.

Config reference
----------------
job: fracture_analysis
schema: adeli                       # built-in schema name (default: adeli)
units:                              # optional category-level unit overrides
  pressure: Pa

model: path/to/model.vtu            # relative to this config file

zone:
  type: sphere                      # sphere | box
  center: [x, y, z]
  radius: r                         # sphere only
  # dim: [dx, dy, dz]               # box only

data:                               # one or more named structural datasets
  fractures:
    file: path/to/fractures.csv
  faults:                           # loaded but not yet plotted
    file: path/to/faults.csv

plot:
  title: "Model vs field data"
  figsize: [8, 8]
  dpi: 200
  avg_directions:                   # model average σ1/σ2/σ3 (default: show=true)
    show: true
    color: "k"
    markersize: 8
  cell_directions:                  # per-cell model directions (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "grey"
    markersize: 3
    alpha: 0.3

output:
  dir: results/                     # optional, defaults to config file directory
  save_vtu: false                   # save extracted sub-model for Paraview

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fem2geo.data import FractureData, FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import (
    PlotConfig, MODEL_COLORS,
    stereo_line, stereo_pole, stereo_contour,
)
from fem2geo.utils.transform import line_enu2sphe

log = logging.getLogger("fem2geoLogger")


def run(cfg: dict, job_dir: Path) -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"), units=cfg.get("units"))
    zone_cfg = cfg["zone"]
    plot_cfg = cfg.get("plot", {})
    out_cfg = cfg.get("output", {})

    title = plot_cfg.get("title", "Model vs structural data")
    figsize = plot_cfg.get("figsize", [8, 8])
    dpi = plot_cfg.get("dpi", 200)
    out_dir = Path(out_cfg.get("dir", job_dir))
    save_vtu = out_cfg.get("save_vtu", False)

    avg_cfg = plot_cfg.get("avg_directions", {})
    cell_cfg = plot_cfg.get("cell_directions", {})
    show_avg = avg_cfg.get("show", True)
    show_cell = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")

    avg_style = PlotConfig.avg().update(avg_cfg)
    cell_style_cfg = (PlotConfig.density() if cell_style == "contour"
                      else PlotConfig.cell()).update(cell_cfg)

    if "data" not in cfg or not cfg["data"]:
        raise ValueError("Config must contain a non-empty 'data' section.")
    if "model" not in cfg:
        raise ValueError("Config must contain a 'model' key.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model and extract zone ───────────────────────────────────────────
    model_path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {model_path}")
    model = Model.from_file(model_path, schema)

    if zone_cfg["type"] == "sphere":
        sub = model.extract_sphere(zone_cfg["center"], zone_cfg["radius"])
    elif zone_cfg["type"] == "box":
        sub = model.extract_box(zone_cfg["center"], np.asarray(zone_cfg["dim"]))
    else:
        raise ValueError(f"Unknown zone type '{zone_cfg['type']}'.")

    log.info(f"  {sub.n_cells} cells in zone")

    if save_vtu:
        sub.save(out_dir / "extract.vtu")

    # ── Load structural datasets ──────────────────────────────────────────────
    data_entries = cfg["data"]
    datasets = {}
    for name, entry in data_entries.items():
        file_path = entry if isinstance(entry, str) else entry.get("file")
        if file_path is None:
            raise ValueError(f"Dataset '{name}' must have a 'file' key.")

        sd = load_structural_csv((job_dir / file_path).resolve())

        if isinstance(sd, FaultData):
            log.warning(
                f"  '{name}' is fault data (strike/dip/rake). "
                f"Fault comparison requires strain tensor derivation and is not "
                f"yet implemented — skipping."
            )
            continue

        datasets[name] = sd

    if not datasets:
        log.warning("No plottable structural datasets after filtering. Nothing to compare.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    legend_elements = [
        Line2D([0], [0], color="k", linewidth=0, marker="o", label=r"$\sigma_1$"),
        Line2D([0], [0], color="k", linewidth=0, marker="s", label=r"$\sigma_2$"),
        Line2D([0], [0], color="k", linewidth=0, marker="v", label=r"$\sigma_3$"),
    ]

    # ── Cell directions (model spread) ────────────────────────────────────────
    if show_cell:
        p1, a1 = line_enu2sphe(sub.dir_s1)
        p2, a2 = line_enu2sphe(sub.dir_s2)
        p3, a3 = line_enu2sphe(sub.dir_s3)
        cell_pc = cell_style_cfg

        if cell_style == "contour":
            stereo_contour(ax, p1, a1, **cell_pc.contour_kwargs())
            stereo_contour(ax, p2, a2, **cell_pc.contour_kwargs())
            stereo_contour(ax, p3, a3, **cell_pc.contour_kwargs())
        else:
            stereo_line(ax, p1, a1, **cell_pc.scatter_kwargs("o"))
            stereo_line(ax, p2, a2, **cell_pc.scatter_kwargs("s"))
            stereo_line(ax, p3, a3, **cell_pc.scatter_kwargs("v"))

    # ── Average principal directions ──────────────────────────────────────────
    if show_avg:
        _, vec = sub.avg_principal()
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])
        stereo_line(ax, p1, a1, label=r"$\sigma_1$",
                    **avg_style.scatter_kwargs("o"))
        stereo_line(ax, p2, a2, label=r"$\sigma_2$",
                    **avg_style.scatter_kwargs("s"))
        stereo_line(ax, p3, a3, label=r"$\sigma_3$",
                    **avg_style.scatter_kwargs("v"))

    # ── Fracture datasets (poles) ─────────────────────────────────────────────
    data_colors = MODEL_COLORS[:len(datasets)]

    for color, (name, fd) in zip(data_colors, datasets.items()):
        entry = data_entries[name]
        data_pc = PlotConfig.cell(color=color).update(
            entry if isinstance(entry, dict) else {}
        )
        data_pc = PlotConfig.from_cfg(data_pc, {"markersize": 6, "alpha": 0.8})

        stereo_pole(ax, fd.planes, label=f"{name} (poles)",
                    color=color, marker="+",
                    markersize=data_pc.markersize, alpha=data_pc.alpha)
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=0, marker="+",
                   label=f"{name} (poles)"))

    # ── Finalise ──────────────────────────────────────────────────────────────
    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(title, y=1.08)

    out_path = out_dir / "fracture_analysis.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")