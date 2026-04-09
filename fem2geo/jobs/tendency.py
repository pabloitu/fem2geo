"""
Job: tendency
==================
Plots slip, dilation, combined, or paired tendency fields on a stereonet
for one model at one extraction zone. Average stress principal directions
are overlaid by default. Individual cell directions can optionally be
shown to visualise the spread around the average.

Tendency types:

- ``slip``: normalized slip tendency Ts' (Morris et al., 1996), range [0, 1].
- ``dilation``: dilation tendency Td (Ferrill et al., 1999), range [0, 1].
- ``combined``: Ts' + Td (Ferrill et al., 2020), range [0, 2].
- ``both``: side-by-side slip and dilation panels.

Config reference
----------------
job: tendency
schema: adeli               # built-in schema name (default: adeli)

model: path/to/model.vtk    # relative to this config file

zone:
  type: sphere              # sphere | box
  center: [x, y, z]
  radius: r                 # sphere only
  # dim: [dx, dy, dz]       # box only

data:                               # optional fracture datasets overlaid as poles
  set_a: path/to/fractures_a.csv
  set_b: path/to/fractures_b.csv

plot:
  title: ""                 # optional, auto-generated if empty
  figsize: [16, 8]          # default for both; [8, 8] for single
  dpi: 200
  tendency: both            # slip | dilation | combined | both
  n_strikes: 180            # stereonet grid resolution
  n_dips: 45
  avg_directions:           # average stress directions (default: show=true)
    show: true
    color: "white"
    markersize: 8
  cell_directions:          # per-cell directions to show spread (default: show=false)
    show: false
    style: scatter          # scatter | contour
    color: "k"
    markersize: 3
    alpha: 0.4
  data:                     # shared style for fracture pole datasets
    markersize: 5
    alpha: 0.7

output:
  dir: results/             # optional, defaults to config file directory
  vtu: extract.vtu          # optional, saves extracted sub-model

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem2geo.internal.io import load_structural_csv
from fem2geo.model import Model
from fem2geo.plots import (
    MODEL_COLORS,
    get_style,
    stereo_field,
    stereo_axes,
    stereo_axes_contour,
    stereo_pole,
)
from fem2geo.runner import parse_config
from fem2geo.utils.tensor import slip_tendency, dilation_tendency, combined_tendency
from fem2geo.utils.transform import grid_nodes, grid_centers

log = logging.getLogger("fem2geoLogger")

_VALID_TENDENCIES = ("slip", "dilation", "combined", "both")

# Plot defaults
AVG_STYLE = {"color": "white", "markersize": 8, "markeredgecolor": "k"}
CELL_STYLE = {"color": "k", "markersize": 3, "alpha": 0.4}
CONTOUR_STYLE = {"color": "k", "levels": 4, "sigma": 2.0, "linewidth": 1.0}
DATA_STYLE = {"markersize": 5, "alpha": 0.7, "marker": "+"}

_CBAR_LABELS = {
    "slip": r"Slip tendency $T'_s$",
    "dilation": r"Dilation tendency $T_d$",
    "combined": r"Combined tendency $T'_s + T_d$",
}

_TITLES = {
    "slip": "Slip tendency",
    "dilation": "Dilation tendency",
    "combined": "Combined tendency",
}

TENDENCY_FUNCTIONS = {
    "slip": slip_tendency,
    "dilation": dilation_tendency,
    "combined": combined_tendency,
}


_VMAX = {"slip": 1.0, "dilation": 1.0, "combined": 2.0}


def run(cfg: dict, job_dir: Path) -> None:

    # load config
    schema, zone, data, plot, out = parse_config(cfg, job_dir)
    out_dir = Path(out.get("dir", job_dir))

    tendency = plot.get("tendency", "both")
    n_strikes = plot.get("n_strikes", 180)
    n_dips = plot.get("n_dips", 45)
    stress_values = plot.get("stress_values", False)

    avg_cfg = plot.get("avg_directions", {})
    show_avg = avg_cfg.get("show", True)
    avg_style = get_style(AVG_STYLE, avg_cfg)

    cell_cfg = plot.get("cell_directions", {})
    show_cell = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")
    cell_base = CONTOUR_STYLE if cell_style == "contour" else CELL_STYLE
    cell_pc = get_style(cell_base, cell_cfg)

    # load model and extract zone
    path = (job_dir / cfg["model"]).resolve()

    # model
    log.info(f"Loading model: {path}")
    model = Model.from_file(path, schema)
    sub = model.extract(zone)
    log.info(f"  {sub.n_cells} cells in zone")

    avg_stress = sub.avg_tensor("stress")
    val, vec = sub.avg_principals("stress")
    phi = float((val[1] - val[2]) / (val[0] - val[2]))

    # stereonet grid
    mesh_strike, mesh_dip = grid_nodes(n_strikes, n_dips)
    centers_strike, centers_dip = grid_centers(mesh_strike, mesh_dip)

    # figure
    is_double = tendency == "both"
    figsize = plot.get("figsize", [16, 8] if is_double else [8, 8])
    fig = plt.figure(figsize=figsize)

    if is_double:
        panels = {
            "slip": fig.add_subplot(121, projection="stereonet"),
            "dilation": fig.add_subplot(122, projection="stereonet"),
        }
    else:
        panels = {tendency: fig.add_subplot(111, projection="stereonet")}

    for ax in panels.values():
        ax.grid(True)
        ax.set_azimuth_ticks([])

    # tendency fields
    for kind, ax in panels.items():
        vals = TENDENCY_FUNCTIONS[kind](
            avg_stress, centers_strike.ravel(), centers_dip.ravel()
        ).reshape(centers_strike.shape)
        stereo_field(
            ax,
            mesh_strike,
            mesh_dip,
            vals,
            vmin=0.0,
            vmax=_VMAX[kind],
            cbar_label=_CBAR_LABELS[kind],
        )

    # cell directions
    if show_cell:
        cell_vecs = np.stack([sub.dir_s1, sub.dir_s2, sub.dir_s3], axis=-1)
        for ax in panels.values():
            if cell_style == "contour":
                stereo_axes_contour(ax, cell_vecs, cell_pc)
            else:
                stereo_axes(ax, cell_vecs, cell_pc)

    # average principal directions
    if show_avg:
        labels = (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$")
        for ax in panels.values():
            stereo_axes(ax, vec, avg_style, labels=labels)

    # fracture data overlays
    if cfg.get("data"):
        data_style = get_style(DATA_STYLE, plot.get("data", {}))
        data_colors = MODEL_COLORS[1 : len(cfg["data"]) + 1]
        for color, (name, path) in zip(data_colors, cfg["data"].items()):
            fd = load_structural_csv((job_dir / path).resolve())
            for ax in panels.values():
                stereo_pole(
                    ax,
                    fd.planes[:, 0],
                    fd.planes[:, 1],
                    label=name,
                    **get_style(data_style, color=color),
                )

    # plots
    if stress_values:
        suffix = (
            f"\n$\\sigma_1={val[0]:.3f}$, $\\sigma_3={val[2]:.3f}$, $\\phi={phi:.2f}$"
        )
    else:
        suffix = ""
    custom_title = plot.get("title", "")
    for kind, ax in panels.items():
        ax.set_title(custom_title + suffix if custom_title else _TITLES[kind] + suffix,
                     y=1.05)
        ax.legend(fontsize=7)

    fig.savefig(
        out_dir / out.get("figure", "tendency.png"),
        dpi=plot.get("dpi", 200),
        bbox_inches="tight",
    )
    plt.close(fig)
    log.info(f"Saved results: {out_dir}")