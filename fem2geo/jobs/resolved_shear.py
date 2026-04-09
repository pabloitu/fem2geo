"""
Job: resolved_shear
=================
Visual Resolved Shear stress analysis: compares observed fault slip directions with
the shear traction predicted by the model stress tensor, plotted together
on a stereonet. Assumes the Wallace-Bott hypothesis of having the slip direction
close to the resolved shear stress.

For each fault plane (strike/dip/rake, Aki & Richards convention), the job:

1. Plots the fault plane as a great circle or pole.
2. Plots the observed slip direction as an arrow on the stereonet (colour A).
3. Resolves the shear traction from the model's average stress tensor on
   that fault plane, and plots the predicted slip direction as an arrow
   (colour B).

If the model stress correctly predicts the fault kinematics, the two
arrows for each fault should overlap in direction and sense. Systematic
divergence indicates a stress field inconsistent with the observed faulting.

Rake convention (Aki & Richards):
  rake > 0  : reverse/thrust component (hanging wall up)
  rake < 0  : normal component (hanging wall down)
  rake = 0  : pure left-lateral
  rake = 180: pure right-lateral

Config reference
----------------
job: resolved_shear
schema: adeli                       # built-in schema name (default: adeli)

model: path/to/model.vtu            # relative to this config file

zone:
  type: sphere                      # sphere | box
  center: [x, y, z]
  radius: r                         # sphere only
  # dim: [dx, dy, dz]               # box only

data:
  faults: path/to/faults.csv        # columns: strike, dip, rake (signed)

plot:
  title: "Resolved shear analysis"
  figsize: [8, 8]
  dpi: 200
  fault_planes:
    show: true
    style: planes                   # poles | planes
    color: "grey"
    alpha: 0.5
    linewidth: 0.8
  observed_slip:
    color: "#E63946"                # red
    arrowsize: 1.0
    linewidth: 1.5
  predicted_slip:
    color: "#2196F3"                # blue
    arrowsize: 1.0
    linewidth: 1.5
  avg_directions:
    show: true
    color: "k"
    markersize: 8

output:
  dir: results/                     # optional, defaults to config file directory
  vtu: extract.vtu                  # optional, saves extracted sub-model

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from fem2geo.data import FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import (
    get_style, stereo_axes, stereo_pole, stereo_plane, stereo_slip_arrow,
)
from fem2geo.runner import resolve_output
from fem2geo.utils.tensor import resolved_rakes

log = logging.getLogger("fem2geoLogger")

AVG_STYLE = {"color": "k", "markersize": 8, "markeredgecolor": "k"}
OBS_STYLE = {"color": "#E63946", "arrowsize": 1.0, "linewidth": 1.5}
PRED_STYLE = {"color": "#2196F3", "arrowsize": 1.0, "linewidth": 1.5}
PLANE_STYLE = {"color": "grey", "alpha": 0.5, "linewidth": 0.8}


def run(cfg: dict, job_dir: Path) -> None:
    out = resolve_output(cfg, job_dir)
    out_dir = out["dir"]
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"))
    zone = cfg.get("zone", {})
    plot = cfg.get("plot", {})

    # plot style options
    plane_cfg = plot.get("fault_planes", {})
    obs_cfg = plot.get("observed_slip", {})
    pred_cfg = plot.get("predicted_slip", {})
    avg_cfg = plot.get("avg_directions", {})

    show_planes = plane_cfg.get("show", True)
    plane_style = plane_cfg.get("style", "planes")
    show_avg = avg_cfg.get("show", True)
    avg_style = get_style(AVG_STYLE, avg_cfg)

    obs_color = obs_cfg.get("color", OBS_STYLE["color"])
    obs_arrowsize = obs_cfg.get("arrowsize", OBS_STYLE["arrowsize"])
    obs_lw = obs_cfg.get("linewidth", OBS_STYLE["linewidth"])
    pred_color = pred_cfg.get("color", PRED_STYLE["color"])
    pred_arrowsize = pred_cfg.get("arrowsize", PRED_STYLE["arrowsize"])
    pred_lw = pred_cfg.get("linewidth", PRED_STYLE["linewidth"])
    plane_color = plane_cfg.get("color", PLANE_STYLE["color"])
    plane_alpha = plane_cfg.get("alpha", PLANE_STYLE["alpha"])
    plane_lw = plane_cfg.get("linewidth", PLANE_STYLE["linewidth"])

    # load model and extract zone
    model_path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {model_path}")
    model = Model.from_file(model_path, schema)
    sub = model.extract(zone)
    log.info(f"  {sub.n_cells} cells in zone")

    if "vtu" in out:
        sub.save(out_dir / out["vtu"])

    avg_stress = sub.avg_tensor("stress")

    # load fault datasets
    fault_datasets = {}
    for name, entry in cfg["data"].items():
        file_path = entry if isinstance(entry, str) else entry.get("file")
        sd = load_structural_csv((job_dir / file_path).resolve())
        if not isinstance(sd, FaultData):
            log.warning(f"  '{name}' has no rake column — skipping.")
            continue
        fault_datasets[name] = sd

    if not fault_datasets:
        raise ValueError("No fault datasets found. CSV files must have strike, dip, rake columns.")

    # figure
    fig = plt.figure(figsize=plot.get("figsize", [8, 8]))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    # plot fault data
    for name, fd in fault_datasets.items():
        strikes, dips, rakes = fd.planes[:, 0], fd.planes[:, 1], fd.rakes

        # fault planes or poles
        if show_planes:
            if plane_style == "planes":
                stereo_plane(ax, strikes, dips, color=plane_color,
                             alpha=plane_alpha, linewidth=plane_lw)
            else:
                stereo_pole(ax, strikes, dips, color=plane_color,
                            marker="+", markersize=6, alpha=plane_alpha)

        # observed slip arrows
        stereo_slip_arrow(ax, strikes, dips, rakes,
                          color=obs_color, arrowsize=obs_arrowsize, linewidth=obs_lw)

        # predicted slip arrows (resolved shear traction)
        pred = resolved_rakes(avg_stress, strikes, dips)
        stereo_slip_arrow(ax, strikes, dips, pred,
                          color=pred_color, arrowsize=pred_arrowsize, linewidth=pred_lw)

    # average principal directions
    if show_avg:
        _, vec = sub.avg_principals("stress")
        stereo_axes(ax, vec, avg_style)

    # legend
    legend_elements = [
        FancyArrowPatch((0, 0), (0.02, 0), arrowstyle="->",
                        color=obs_color, lw=obs_lw, label="Observed slip"),
        FancyArrowPatch((0, 0), (0.02, 0), arrowstyle="->",
                        color=pred_color, lw=pred_lw, label="Predicted slip"),
    ]
    if show_planes:
        if plane_style == "planes":
            legend_elements.append(
                Line2D([0], [0], color=plane_color, linewidth=plane_lw,
                       alpha=plane_alpha, label="Fault planes"))
        else:
            legend_elements.append(
                Line2D([0], [0], color=plane_color, linewidth=0, marker="+",
                       markersize=6, alpha=plane_alpha, label="Fault poles"))
    if show_avg:
        legend_elements.extend([
            Line2D([0], [0], color=avg_style["color"], linewidth=0, marker="o", label=r"$\sigma_1$"),
            Line2D([0], [0], color=avg_style["color"], linewidth=0, marker="s", label=r"$\sigma_2$"),
            Line2D([0], [0], color=avg_style["color"], linewidth=0, marker="v", label=r"$\sigma_3$"),
        ])

    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(plot.get("title", "Resolved shear analysis"), y=1.08)
    fig.savefig(out_dir / out.get("figure", "resolved_shear.png"),
                dpi=plot.get("dpi", 200), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved results: {out_dir}")