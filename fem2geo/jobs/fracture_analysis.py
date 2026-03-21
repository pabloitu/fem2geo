"""
Job: fracture_analysis
======================
Compares fracture (joint, vein, dyke) orientation measurements with the stress
state predicted by a FEM model at a given extraction zone. Plots fracture poles
and model principal directions together on a stereonet.

Structural data is read from CSV files via
:func:`fem2geo.internal.io.load_structural_csv`. Only ``strike, dip`` columns
are supported (:class:`FractureData`).

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

data:                               # one or more named fracture datasets
  set_a: path/to/fractures_a.csv
  set_b: path/to/fractures_b.csv

plot:
  title: "Model vs field data"
  figsize: [8, 8]
  dpi: 200
  avg_directions:                   # model average σ1/σ2/σ3 (default: show=true)
    show: true
    markersize: 8
  cell_directions:                  # per-cell model directions (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "grey"
    markersize: 3
    alpha: 0.3
  data:                             # shared style for all fracture pole datasets
    markersize: 6
    alpha: 0.8

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
from matplotlib.lines import Line2D

from fem2geo.internal.io import load_structural_csv
from fem2geo.model import Model
from fem2geo.plots import (
    PlotConfig,
    MODEL_COLORS,
    stereo_line,
    stereo_pole,
    stereo_contour,
)
from fem2geo.runner import parse_config
from fem2geo.utils.transform import line_enu2sphe

log = logging.getLogger("fem2geoLogger")


# Plot defaults

AVG_STYLE = PlotConfig(color=MODEL_COLORS[0], markersize=8, markeredgecolor="k")
CELL_STYLE = PlotConfig(color="grey", markersize=3, alpha=0.3)
CONTOUR_STYLE = PlotConfig(color="grey", levels=4, sigma=2.0, linewidth=1.0)
DATA_STYLE = PlotConfig(markersize=6, alpha=0.8, marker="+")


# Main job


def run(cfg: dict, job_dir: Path) -> None:

    # Load configs
    schema, zone, data, plot, out = parse_config(cfg, job_dir)
    out_dir = Path(out.get("dir", job_dir))

    avg_cfg = plot.get("avg_directions", {})
    show_avg = avg_cfg.get("show", True)
    avg_style = AVG_STYLE.update(avg_cfg)

    cell_cfg = plot.get("cell_directions", {})
    show_cell = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")
    cell_pc = (CONTOUR_STYLE if cell_style == "contour" else CELL_STYLE).update(
        cell_cfg
    )

    data_style = DATA_STYLE.update(plot.get("data", {}))

    # load model and extract zone
    model_path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {model_path}")
    model = Model.from_file(model_path, schema)
    sub = model.extract(zone)
    log.info(f"  {sub.n_cells} cells in zone")

    # load structural datasets
    datasets = {
        name: load_structural_csv((job_dir / path).resolve())
        for name, path in cfg["data"].items()
    }

    # figure
    fig = plt.figure(figsize=plot.get("figsize", [8, 8]))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=avg_style.color,
            linewidth=0,
            marker="o",
            label=r"$\sigma_1$",
        ),
        Line2D(
            [0],
            [0],
            color=avg_style.color,
            linewidth=0,
            marker="s",
            label=r"$\sigma_2$",
        ),
        Line2D(
            [0],
            [0],
            color=avg_style.color,
            linewidth=0,
            marker="v",
            label=r"$\sigma_3$",
        ),
    ]

    # cell directions
    if show_cell:
        p1, a1 = line_enu2sphe(sub.dir_s1)
        p2, a2 = line_enu2sphe(sub.dir_s2)
        p3, a3 = line_enu2sphe(sub.dir_s3)
        if cell_style == "contour":
            kw = cell_pc.kwargs()
            stereo_contour(ax, p1, a1, **kw)
            stereo_contour(ax, p2, a2, **kw)
            stereo_contour(ax, p3, a3, **kw)
        else:
            stereo_line(ax, p1, a1, **cell_pc.update(marker="o").kwargs())
            stereo_line(ax, p2, a2, **cell_pc.update(marker="s").kwargs())
            stereo_line(ax, p3, a3, **cell_pc.update(marker="v").kwargs())

    # average principal directions
    if show_avg:
        _, vec = sub.avg_principals("stress")
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])
        stereo_line(
            ax, p1, a1, label=r"$\sigma_1$", **avg_style.update(marker="o").kwargs()
        )
        stereo_line(
            ax, p2, a2, label=r"$\sigma_2$", **avg_style.update(marker="s").kwargs()
        )
        stereo_line(
            ax, p3, a3, label=r"$\sigma_3$", **avg_style.update(marker="v").kwargs()
        )

    # fracture datasets — colors start at MODEL_COLORS[1] to avoid clashing with avg
    data_colors = MODEL_COLORS[1 : len(datasets) + 1]
    for color, (name, fd) in zip(data_colors, datasets.items()):
        stereo_pole(
            ax,
            fd.planes[:, 0],
            fd.planes[:, 1],
            label=f"{name}",
            **data_style.update(color=color).kwargs(),
        )
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=0, marker="+", label=name)
        )

    if out.get("vtu", False):
        model.save(out_dir / out["vtu"])

    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(plot.get("title", "Model vs Fracture Data"), y=1.08)
    fig.savefig(
        out_dir / out.get("figure", "fracture_analysis.png"),
        dpi=plot.get("dpi", 200),
        bbox_inches="tight",
    )
    plt.close(fig)
    log.info(f"Saved results: {out_dir}")
