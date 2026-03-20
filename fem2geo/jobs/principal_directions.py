"""
Job: principal_directions
=========================
Probes a model at a given location by plotting principal stress directions
on a stereonet. By default shows the average stress directions only.
Cell directions can be enabled to visualise the spread around the average.
Multiple models can be compared at the same zone — each gets a distinct colour.

Config reference
----------------
job: principal_directions
schema: adeli                   # built-in schema name (default: adeli)
units:                          # optional category-level unit overrides
  pressure: Pa

model: path/to/model.vtk        # single model — relative to this config file
# OR
models:                         # multiple models for comparison
  model_a: path/to/model_a.vtu
  model_b: path/to/model_b.vtu

zone:
  type: sphere                  # sphere | box
  center: [x, y, z]
  radius: r                     # sphere only
  # dim: [dx, dy, dz]           # box only

plot:
  title: "Principal stress directions"
  figsize: [8, 8]
  dpi: 200
  avg_directions:               # average stress directions (default: show=true)
    show: true
    color: "white"              # overridden per model in multi-model mode
    markersize: 8
    alpha: 1.0
  cell_directions:              # per-cell directions, shows spread (default: show=false)
    show: false
    style: scatter              # scatter | contour
    color: "k"                  # overridden per model in multi-model mode
    markersize: 3
    alpha: 0.4
    # contour options: levels, sigma, linewidth

output:
  dir: results/                 # optional, defaults to config file directory
  save_vtu: false               # save extracted sub-model(s) for Paraview

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import PlotConfig, MODEL_COLORS, stereo_line, stereo_contour
from fem2geo.utils.transform import line_enu2sphe

log = logging.getLogger("fem2geoLogger")


def run(cfg: dict, job_dir: Path) -> None:
    # Config
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"), units=cfg.get("units"))
    zone_cfg = cfg["zone"]
    plot_cfg = cfg.get("plot", {})
    out_cfg = cfg.get("output", {})

    title = plot_cfg.get("title", "Principal stress directions")
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

    if "models" in cfg:
        model_paths = cfg["models"]
    elif "model" in cfg:
        model_paths = {"model": cfg["model"]}
    else:
        raise ValueError("Config must contain 'model' or 'models'.")
    model_names = list(model_paths.keys())

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = MODEL_COLORS[:len(model_paths)]

    # Figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    legend_elements = [
        Line2D([0], [0], color="k", linewidth=0, marker="o", label=r"$\sigma_1$"),
        Line2D([0], [0], color="k", linewidth=0, marker="s", label=r"$\sigma_2$"),
        Line2D([0], [0], color="k", linewidth=0, marker="v", label=r"$\sigma_3$"),
    ]

    # Per-model loop
    for color, name in zip(colors, model_names):

        path = (job_dir / model_paths[name]).resolve()
        log.info(f"Loading {name}: {path}")

        model = Model.from_file(path, schema)

        model = model.extract(zone_cfg)

        log.info(f"  {model.n_cells} cells in zone")

        if save_vtu:
            model.save(out_dir / f"{name}_extract.vtu")

        # Cell directions (batched — single call per principal direction)
        if show_cell:
            p1, a1 = line_enu2sphe(model.dir_s1)
            p2, a2 = line_enu2sphe(model.dir_s2)
            p3, a3 = line_enu2sphe(model.dir_s3)
            cell_pc = PlotConfig.from_cfg(cell_style_cfg, {"color": color})

            if cell_style == "contour":
                stereo_contour(ax, p1, a1, **cell_pc.contour_kwargs())
                stereo_contour(ax, p2, a2, **cell_pc.contour_kwargs())
                stereo_contour(ax, p3, a3, **cell_pc.contour_kwargs())
            else:
                stereo_line(ax, p1, a1, **cell_pc.scatter_kwargs("o"))
                stereo_line(ax, p2, a2, **cell_pc.scatter_kwargs("s"))
                stereo_line(ax, p3, a3, **cell_pc.scatter_kwargs("v"))

        # Average stress principal directions
        if show_avg:
            _, vec = model.avg_principal()
            avg_pc = PlotConfig.from_cfg(avg_style, {"color": color})
            label = name if len(model_paths) > 1 else None

            p1, a1 = line_enu2sphe(vec[:, 0])
            p2, a2 = line_enu2sphe(vec[:, 1])
            p3, a3 = line_enu2sphe(vec[:, 2])
            stereo_line(ax, p1, a1, label=label, **avg_pc.scatter_kwargs("o"))
            stereo_line(ax, p2, a2, **avg_pc.scatter_kwargs("s"))
            stereo_line(ax, p3, a3, **avg_pc.scatter_kwargs("v"))

        if len(model_paths) > 1:
            legend_elements.append(Patch(facecolor=color, edgecolor="k", label=name))

    # Finalise
    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(title, y=1.08)

    out_path = out_dir / "principal_directions.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")
