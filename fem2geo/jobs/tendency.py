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
units:                      # optional category-level unit overrides
  pressure: Pa

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
    PlotConfig,
    MODEL_COLORS,
    stereo_field,
    stereo_line,
    stereo_contour,
    stereo_pole,
)
from fem2geo.runner import parse_config
from fem2geo.utils.tensor import slip_tendency, dilation_tendency, combined_tendency
from fem2geo.utils.transform import line_enu2sphe, grid_nodes, grid_centers

log = logging.getLogger("fem2geoLogger")

_VALID_TENDENCIES = ("slip", "dilation", "combined", "both")

# Plot defaults
AVG_STYLE = PlotConfig(color="white", markersize=8, markeredgecolor="k")
CELL_STYLE = PlotConfig(color="k", markersize=3, alpha=0.4)
CONTOUR_STYLE = PlotConfig(color="k", levels=4, sigma=2.0, linewidth=1.0)
DATA_STYLE = PlotConfig(markersize=5, alpha=0.7, marker="+")

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

    avg_cfg = plot.get("avg_directions", {})
    show_avg = avg_cfg.get("show", True)
    avg_style = AVG_STYLE.update(avg_cfg)

    cell_cfg = plot.get("cell_directions", {})
    show_cell = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")
    cell_pc = (CONTOUR_STYLE if cell_style == "contour" else CELL_STYLE).update(
        cell_cfg
    )

    path = (job_dir / cfg["model"]).resolve()

    # model
    log.info(f"Loading model: {path}")
    model = Model.from_file(path, schema)
    sub = model.extract(zone)
    log.info(f"  {sub.n_cells} cells in zone")

    avg_stress = sub.avg_tensor("stress")
    val, vec = sub.avg_principals("stress")
    phi = float((val[1] - val[2]) / (val[0] - val[2]))

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

    for kind, ax in panels.items():

        mesh_s, mesh_d = grid_nodes(n_strikes, n_dips)
        cs, cd = grid_centers(mesh_s, mesh_d)
        # planes = np.column_stack([cs.ravel(), cd.ravel()])
        vals = TENDENCY_FUNCTIONS[kind](avg_stress, planes=[cs, cd]).reshape(cs.shape)

        stereo_field(
            ax,
            mesh_s,
            mesh_d,
            vals,
            vmin=0.0,
            vmax=_VMAX[kind],
            cbar_label=_CBAR_LABELS[kind],
        )

    if show_cell:
        p1, a1 = line_enu2sphe(sub.dir_s1)
        p2, a2 = line_enu2sphe(sub.dir_s2)
        p3, a3 = line_enu2sphe(sub.dir_s3)
        for ax in panels.values():
            if cell_style == "contour":
                kw = cell_pc.kwargs()
                stereo_contour(ax, p1, a1, **kw)
                stereo_contour(ax, p2, a2, **kw)
                stereo_contour(ax, p3, a3, **kw)
            else:
                stereo_line(ax, p1, a1, **cell_pc.update(marker="o").kwargs())
                stereo_line(ax, p2, a2, **cell_pc.update(marker="s").kwargs())
                stereo_line(ax, p3, a3, **cell_pc.update(marker="v").kwargs())

    if show_avg:
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])
        for ax in panels.values():
            stereo_line(
                ax, p1, a1, label=r"$\sigma_1$", **avg_style.update(marker="o").kwargs()
            )
            stereo_line(
                ax, p2, a2, label=r"$\sigma_2$", **avg_style.update(marker="s").kwargs()
            )
            stereo_line(
                ax, p3, a3, label=r"$\sigma_3$", **avg_style.update(marker="v").kwargs()
            )

    if cfg.get("data"):
        data_style = DATA_STYLE.update(plot.get("data", {}))
        data_colors = MODEL_COLORS[1 : len(cfg["data"]) + 1]
        for color, (name, path) in zip(data_colors, cfg["data"].items()):
            fd = load_structural_csv((job_dir / path).resolve())
            for ax in panels.values():
                stereo_pole(
                    ax,
                    fd.planes[:, 0],
                    fd.planes[:, 1],
                    label=name,
                    **data_style.update(color=color).kwargs(),
                )

    suffix = f"\n$\\sigma_1={val[0]:.3f}$, $\\sigma_3={val[2]:.3f}$, $\\phi={phi:.2f}$"
    custom_title = plot.get("title", "")
    for kind, ax in panels.items():
        ax.set_title(custom_title if custom_title else _TITLES[kind] + suffix, y=1.05)
        ax.legend(fontsize=7)

    # out

    if "vtu" in out:
        sub.save(out_dir / out.get("vtu", "extract.vtu"))
    fig.savefig(
        out_dir / out.get("figure", "tendency.png"),
        dpi=plot.get("dpi", 200),
        bbox_inches="tight",
    )
    plt.close(fig)
    log.info(f"Saved results: {out_dir}")
