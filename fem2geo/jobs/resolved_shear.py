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
units:                              # optional category-level unit overrides
  pressure: Pa

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
from matplotlib.patches import FancyArrowPatch

from fem2geo.data import FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.model import Model
from fem2geo.plots import (
    PlotConfig,
    stereo_line,
    stereo_pole,
    stereo_plane,
    stereo_slip_arrow,
)
from fem2geo.runner import parse_config
from fem2geo.utils.tensor import resolved_shear_enu
from fem2geo.utils.transform import line_enu2sphe, slip_enu2rake

log = logging.getLogger("fem2geoLogger")

AVG_STYLE = PlotConfig(color="k", markersize=8, markeredgecolor="k")
OBS_STYLE = {"color": "#E63946", "arrowsize": 1.0, "linewidth": 1.5}
PRED_STYLE = {"color": "#2196F3", "arrowsize": 1.0, "linewidth": 1.5}
PLANE_STYLE = {"color": "grey", "alpha": 0.5, "linewidth": 0.8}


def run(cfg: dict, job_dir: Path) -> None:
    schema, zone, data, plot, out = parse_config(cfg, job_dir)
    out_dir = Path(out.get("dir", job_dir))

    plane_cfg = plot.get("fault_planes", {})
    obs_cfg = plot.get("observed_slip", {})
    pred_cfg = plot.get("predicted_slip", {})
    avg_cfg = plot.get("avg_directions", {})

    show_planes = plane_cfg.get("show", True)
    plane_style = plane_cfg.get("style", "planes")
    show_avg = avg_cfg.get("show", True)
    avg_style = AVG_STYLE.update(avg_cfg)

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

    if out.get("save_vtu", False):
        sub.save(out_dir / "extract.vtu")

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
        raise ValueError(
            "No fault datasets found. CSV files must have strike, dip, rake columns."
        )

    # figure
    fig = plt.figure(figsize=plot.get("figsize", [8, 8]))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    # plot fault data
    for name, fd in fault_datasets.items():
        for i in range(len(fd)):
            strike, dip, rk = fd.planes[i, 0], fd.planes[i, 1], fd.rakes[i]

            # fault plane
            if show_planes:
                if plane_style == "planes":
                    stereo_plane(
                        ax,
                        strike,
                        dip,
                        color=plane_color,
                        alpha=plane_alpha,
                        linewidth=plane_lw,
                    )
                else:
                    stereo_pole(
                        ax,
                        strike,
                        dip,
                        color=plane_color,
                        marker="+",
                        markersize=6,
                        alpha=plane_alpha,
                    )

            # observed slip arrow
            stereo_slip_arrow(
                ax,
                strike,
                dip,
                rk,
                color=obs_color,
                arrowsize=obs_arrowsize,
                linewidth=obs_lw,
            )

            # predicted slip arrow (resolved shear traction)
            _, tau_hat = resolved_shear_enu(avg_stress, plane=[strike, dip])
            if np.linalg.norm(tau_hat) > 1e-12:
                pred_rake = slip_enu2rake(tau_hat, strike, dip)
                stereo_slip_arrow(
                    ax,
                    strike,
                    dip,
                    pred_rake,
                    color=pred_color,
                    arrowsize=pred_arrowsize,
                    linewidth=pred_lw,
                )

    # average principal directions
    if show_avg:
        _, vec = sub.avg_principals("stress")
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])
        stereo_line(ax, p1, a1, **avg_style.update(marker="o").kwargs())
        stereo_line(ax, p2, a2, **avg_style.update(marker="s").kwargs())
        stereo_line(ax, p3, a3, **avg_style.update(marker="v").kwargs())

    # legend
    legend_elements = [
        FancyArrowPatch(
            (0, 0),
            (0.02, 0),
            arrowstyle="->",
            color=obs_color,
            lw=obs_lw,
            label="Observed slip",
        ),
        FancyArrowPatch(
            (0, 0),
            (0.02, 0),
            arrowstyle="->",
            color=pred_color,
            lw=pred_lw,
            label="Predicted slip",
        ),
    ]
    if show_planes:
        if plane_style == "planes":
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=plane_color,
                    linewidth=plane_lw,
                    alpha=plane_alpha,
                    label="Fault planes",
                )
            )
        else:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=plane_color,
                    linewidth=0,
                    marker="+",
                    markersize=6,
                    alpha=plane_alpha,
                    label="Fault poles",
                )
            )
    if show_avg:
        legend_elements.extend(
            [
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
        )

    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(plot.get("title", "Resolved shear analysis"), y=1.08)
    fig.savefig(
        out_dir / out.get("figure", "resolved_shear.png"),
        dpi=plot.get("dpi", 200),
        bbox_inches="tight",
    )
    plt.close(fig)
    log.info(f"Saved results: {out_dir}")
