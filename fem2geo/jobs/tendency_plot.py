"""
Job: tendency_plot
==================
Plots slip and/or dilation tendency fields on a stereonet for one model
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
  title: ""                 # optional, auto-generated if empty
  figsize: [16, 7]          # default for both; [8, 8] for single
  dpi: 200
  tendency: both            # slip | dilation | both (default: both)
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

from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import PlotConfig, stereo_field, stereo_line, stereo_contour
from fem2geo.utils.transform import line_enu2sphe

log = logging.getLogger("fem2geoLogger")


def run(cfg: dict, job_dir: Path) -> None:
    # config
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"), units=cfg.get("units"))
    zone_cfg = cfg["zone"]
    plot_cfg = cfg.get("plot", {})
    out_cfg = cfg.get("output", {})

    tendency = plot_cfg.get("tendency", "both")
    dpi = plot_cfg.get("dpi", 200)
    out_dir = Path(out_cfg.get("dir", job_dir))
    save_vtu = out_cfg.get("save_vtu", False)

    avg_cfg = plot_cfg.get("avg_directions", {})
    cell_cfg = plot_cfg.get("cell_directions", {})
    show_avg = avg_cfg.pop("show", True) if isinstance(avg_cfg, dict) else avg_cfg
    show_cell = cell_cfg.pop("show", False) if isinstance(cell_cfg, dict) else cell_cfg
    cell_style = cell_cfg.pop("style", "scatter") if isinstance(cell_cfg, dict) else "scatter"

    avg_style = PlotConfig.avg(color="white").update(
        avg_cfg if isinstance(avg_cfg, dict) else {})
    cell_style_cfg = (PlotConfig.density() if cell_style == "contour"
                      else PlotConfig.cell()).update(
        cell_cfg if isinstance(cell_cfg, dict) else {})

    if tendency not in ("slip", "dilation", "both"):
        raise ValueError(
            f"plot.tendency must be slip | dilation | both, got '{tendency}'.")
    if "model" not in cfg:
        raise ValueError("tendency_plot requires a 'model' key.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # load and extract
    path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {path}")
    model = Model.from_file(path, schema)

    if zone_cfg["type"] == "sphere":
        sub = model.extract_sphere(zone_cfg["center"], zone_cfg["radius"])
    elif zone_cfg["type"] == "box":
        sub = model.extract_box(zone_cfg["center"], np.asarray(zone_cfg["dim"]))
    else:
        raise ValueError(f"Unknown zone type '{zone_cfg['type']}'.")

    log.info(f"  {sub.n_cells} cells in zone")

    if save_vtu:
        sub.save(out_dir / "extract.vtu")

    # average stress and principal directions
    avg_stress = sub.avg_tensor("stress")
    val, vec = np.linalg.eigh(avg_stress)
    order = np.argsort(val)
    val, vec = val[order], vec[:, order]
    phi = float((val[1] - val[2]) / (val[0] - val[2]))

    # figure and axes
    if tendency == "both":
        figsize = plot_cfg.get("figsize", [16, 7])
    else:
        figsize = plot_cfg.get("figsize", [8, 8])

    fig = plt.figure(figsize=figsize)

    if tendency == "both":
        ax_slip = fig.add_subplot(121, projection="stereonet")
        ax_dil = fig.add_subplot(122, projection="stereonet")
        axes = {"slip": ax_slip, "dilation": ax_dil}
    else:
        axes = {tendency: fig.add_subplot(111, projection="stereonet")}

    # tendency fields
    for kind, ax in axes.items():
        ax.grid(True)
        stereo_field(ax, avg_stress, kind=kind)

    # cell directions (spread)
    if show_cell:
        p1, a1 = line_enu2sphe(sub.dir_s1)
        p2, a2 = line_enu2sphe(sub.dir_s2)
        p3, a3 = line_enu2sphe(sub.dir_s3)

        for ax in axes.values():
            if cell_style == "contour":
                stereo_contour(ax, p1, a1, **cell_style_cfg.contour_kwargs())
                stereo_contour(ax, p2, a2, **cell_style_cfg.contour_kwargs())
                stereo_contour(ax, p3, a3, **cell_style_cfg.contour_kwargs())
            else:
                stereo_line(ax, p1, a1, **cell_style_cfg.scatter_kwargs("o"))
                stereo_line(ax, p2, a2, **cell_style_cfg.scatter_kwargs("s"))
                stereo_line(ax, p3, a3, **cell_style_cfg.scatter_kwargs("v"))

    # average directions
    if show_avg:
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])

        for ax in axes.values():
            stereo_line(ax, p1, a1, label=r"$\sigma_1$",
                        **avg_style.scatter_kwargs("o"))
            stereo_line(ax, p2, a2, label=r"$\sigma_2$",
                        **avg_style.scatter_kwargs("s"))
            stereo_line(ax, p3, a3, label=r"$\sigma_3$",
                        **avg_style.scatter_kwargs("v"))

    # titles and legends
    titles = {"slip": "Slip tendency", "dilation": "Dilation tendency"}
    suffix = (f"\n$\\sigma_1={val[0]:.3f}$, $\\sigma_3={val[2]:.3f}$,"
              f" $\\phi={phi:.2f}$")
    custom_title = plot_cfg.get("title", "")

    for kind, ax in axes.items():
        t = custom_title if custom_title else titles[kind] + suffix
        ax.set_title(t, y=1.05)
        ax.legend(fontsize=7)

    # save
    out_path = out_dir / "tendency_plot.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")