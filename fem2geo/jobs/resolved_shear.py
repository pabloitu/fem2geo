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
  faults:
    file: path/to/faults.csv        # columns: strike, dip, rake (signed)

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
import mplstereonet
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from fem2geo.data import FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import (
    PlotConfig,
    stereo_line,
    stereo_pole,
    stereo_plane,
    stereo_arrow,
)
from fem2geo.utils.tensor import resolved_shear_enu
from fem2geo.utils.transform import line_enu2sphe, line_rake2sphe, slip_enu2rake

log = logging.getLogger("fem2geoLogger")

# constant arrow length in projected stereonet coordinates (radians)
_ARROW_LENGTH = 0.08


def _slip_arrow(strike, dip, signed_rake, arrow_length=_ARROW_LENGTH):
    """
    Compute projected arrow start/end for a slip vector on a fault plane.

    The arrow starts at the slip line's stereonet position (from the
    unsigned rake) and points toward the plane's pole (reverse sense,
    rake > 0) or away from it (normal sense, rake < 0).

    Parameters
    ----------
    strike, dip : float
        Fault plane orientation in degrees.
    signed_rake : float
        Signed rake in degrees (Aki & Richards convention, (-180, 180]).
    arrow_length : float
        Arrow length in projected (radians) coordinates.

    Returns
    -------
    from_xy, to_xy : tuple of float
        (x, y) in projected stereonet coordinates for arrow tail and head.
    """
    # stereonet position from unsigned rake
    plunge, azm = line_rake2sphe(strike, dip, abs(signed_rake))

    # slip position in projected coords
    sx, sy = mplstereonet.line(plunge, azm)
    sx, sy = sx.item(), sy.item()

    # pole position in projected coords
    px, py = mplstereonet.pole(strike, dip)
    px, py = px.item(), py.item()

    # direction from slip position toward pole
    dx, dy = px - sx, py - sy
    dist = np.hypot(dx, dy)
    if dist < 1e-12:
        return (sx, sy), (sx, sy)

    dx, dy = dx / dist, dy / dist

    # reverse sense (rake > 0): arrow toward pole
    # normal sense (rake < 0): arrow away from pole
    if signed_rake < 0:
        dx, dy = -dx, -dy

    to_x = sx + arrow_length * dx
    to_y = sy + arrow_length * dy

    return (sx, sy), (to_x, to_y)


def run(cfg: dict, job_dir: Path) -> None:
    # config
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"), units=cfg.get("units"))
    zone_cfg = cfg["zone"]
    plot_cfg = cfg.get("plot", {})
    out_cfg = cfg.get("output", {})

    title = plot_cfg.get("title", "Resolved shear analysis")
    figsize = plot_cfg.get("figsize", [8, 8])
    dpi = plot_cfg.get("dpi", 200)
    out_dir = Path(out_cfg.get("dir", job_dir))
    save_vtu = out_cfg.get("save_vtu", False)

    plane_cfg = plot_cfg.get("fault_planes", {})
    obs_cfg = plot_cfg.get("observed_slip", {})
    pred_cfg = plot_cfg.get("predicted_slip", {})
    avg_cfg = plot_cfg.get("avg_directions", {})

    show_planes = plane_cfg.get("show", True)
    plane_style = plane_cfg.get("style", "planes")
    show_avg = avg_cfg.get("show", True)

    if "data" not in cfg or not cfg["data"]:
        raise ValueError("Config must contain a non-empty 'data' section.")
    if "model" not in cfg:
        raise ValueError("Config must contain a 'model' key.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # load model and extract zone
    model_path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {model_path}")
    model = Model.from_file(model_path, schema)

    sub = model.extract(zone_cfg)

    log.info(f"  {sub.n_cells} cells in zone")

    if save_vtu:
        sub.save(out_dir / "extract.vtu")

    # average stress tensor
    avg_stress = sub.avg_tensor("stress")

    # load fault datasets
    data_entries = cfg["data"]
    fault_datasets = {}
    for name, entry in data_entries.items():
        file_path = entry if isinstance(entry, str) else entry.get("file")
        if file_path is None:
            raise ValueError(f"Dataset '{name}' must have a 'file' key.")

        sd = load_structural_csv((job_dir / file_path).resolve())
        if not isinstance(sd, FaultData):
            log.warning(
                f"  '{name}' is not fault data (needs strike/dip/rake columns) — skipping."
            )
            continue
        fault_datasets[name] = sd

    if not fault_datasets:
        raise ValueError(
            "No fault datasets found. CSV files must have strike, dip, rake columns."
        )

    # figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    # style
    obs_color = obs_cfg.get("color", "#E63946")
    obs_arrowsize = obs_cfg.get("arrowsize", 1.0)
    obs_lw = obs_cfg.get("linewidth", 1.5)
    pred_color = pred_cfg.get("color", "#2196F3")
    pred_arrowsize = pred_cfg.get("arrowsize", 1.0)
    pred_lw = pred_cfg.get("linewidth", 1.5)
    plane_color = plane_cfg.get("color", "grey")
    plane_alpha = plane_cfg.get("alpha", 0.5)
    plane_lw = plane_cfg.get("linewidth", 0.8)

    # plot fault data
    for name, fd in fault_datasets.items():
        for i in range(len(fd)):
            strike, dip, rk = fd.strikes[i], fd.dips[i], fd.rakes[i]

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

            # observed slip arrow (signed rake from data)
            from_xy, to_xy = _slip_arrow(strike, dip, rk)
            stereo_arrow(
                ax, from_xy, to_xy, color=obs_color, arrowsize=obs_arrowsize, linewidth=obs_lw
            )

            # predicted slip arrow (signed rake from resolved shear traction)
            _, tau_hat = resolved_shear_enu(avg_stress, plane=[strike, dip])
            if np.linalg.norm(tau_hat) > 1e-12:
                pred_rake = slip_enu2rake(tau_hat, strike, dip)
                from_xy, to_xy = _slip_arrow(strike, dip, pred_rake)
                stereo_arrow(
                    ax,
                    from_xy,
                    to_xy,
                    color=pred_color,
                    arrowsize=pred_arrowsize,
                    linewidth=pred_lw,
                )

    # average principal directions
    if show_avg:
        avg_style = PlotConfig.avg().update(avg_cfg)
        _, vec = sub.avg_principal()
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])
        stereo_line(ax, p1, a1, **avg_style.scatter_kwargs("o"))
        stereo_line(ax, p2, a2, **avg_style.scatter_kwargs("s"))
        stereo_line(ax, p3, a3, **avg_style.scatter_kwargs("v"))

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
                Line2D([0], [0], color="k", linewidth=0, marker="o", label=r"$\sigma_1$"),
                Line2D([0], [0], color="k", linewidth=0, marker="s", label=r"$\sigma_2$"),
                Line2D([0], [0], color="k", linewidth=0, marker="v", label=r"$\sigma_3$"),
            ]
        )

    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(title, y=1.08)

    # save
    out_path = out_dir / "resolved_shear.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")
