"""
Job: principal_directions
=========================
Probes a model at a given location by plotting principal stress directions
on a stereonet. By default, shows the average stress directions only. Cell
directions can be enabled to visualise the spread around the average.


Config reference
----------------
job: principal_directions
schema: adeli
units:
  pressure: Pa

model: path/to/model.vtk
# OR
models:
  model_a: path/to/model_a.vtu
  model_b: path/to/model_b.vtu

zone:
  type: sphere
  center: [x, y, z]
  radius: r

plot:
  title: "Principal stress directions"
  figsize: [8, 8]
  dpi: 200
  avg_directions:
    show: true
    color: "white"
    markersize: 8
  cell_directions:
    show: false
    style: scatter
    color: "k"
    markersize: 3
    alpha: 0.4

output:
  dir: results/
  figure: principal_directions.png
  vtu: extract.vtu

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

from fem2geo.model import Model
from fem2geo.plots import PlotConfig, MODEL_COLORS, stereo_axes, stereo_axes_contour
from fem2geo.runner import parse_config

log = logging.getLogger("fem2geoLogger")

# Default Plot Properties
AVG_STYLE = PlotConfig(color="red", markersize=8, markeredgecolor="k")
CELL_STYLE = PlotConfig(color="red", markersize=3, markeredgecolor="none", alpha=0.4)
CONTOUR_STYLE = PlotConfig(color="red", levels=4, sigma=2.0, linewidth=1.0)


def run(cfg: dict, job_dir: Path) -> None:

    # read and parse different config segments
    schema, zone, data, plot, out = parse_config(cfg, job_dir)
    out_dir = out["dir"]

    # model paths
    models = cfg.get("models", {"model": cfg.get("model")})

    # plot options
    avg_cfg = plot.get("avg_directions", {})    # Config for model average
    avg_show = avg_cfg.get("show", True)        # Flag to show model average
    avg_pc = AVG_STYLE.update(avg_cfg)          # Plot config, default at top of module

    cell_cfg = plot.get("cell_directions", {})      # Config for per-cell plot
    cell_show = cell_cfg.get("show", False)         # Flag to show per-cell
    cell_style = cell_cfg.get("style", "scatter")   # scatter or contour
    cell_pc = CONTOUR_STYLE.update(cell_cfg) if cell_style == "contour" \
        else CELL_STYLE.update(cell_cfg)

    # figure
    fig = plt.figure(figsize=plot.get("figsize", [8, 8]))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)
    legend = [
        Line2D([0], [0], color="k", lw=0, marker="o", label=r"$\sigma_1$"),
        Line2D([0], [0], color="k", lw=0, marker="s", label=r"$\sigma_2$"),
        Line2D([0], [0], color="k", lw=0, marker="v", label=r"$\sigma_3$"),
    ]

    colors = MODEL_COLORS[: len(models)]  # per model color
    # Model loop
    for color, name in zip(colors, models):
        path = (job_dir / models[name]).resolve()
        log.info(f"Loading {name}: {path}")

        model = Model.from_file(path, schema)       # Load model
        model = model.extract(zone)                 # Extract region of interest

        log.info("Processing results...")
        if cell_show:
            cell_vecs = np.stack(
                [model.dir_s1, model.dir_s2, model.dir_s3], axis=-1
            )
            cpc = cell_pc.update(color=color)
            if cell_style == "contour":
                stereo_axes_contour(ax, cell_vecs, cpc)
            else:
                stereo_axes(ax, cell_vecs, cpc)

        if avg_show:
            _, vec = model.avg_principals()
            label = name if len(models) > 1 else None
            apc = avg_pc.update(color=color)
            stereo_axes(ax, vec, apc, labels=(label, None, None))

        if len(models) > 1:
            legend.append(Patch(facecolor=color, edgecolor="k", label=name))
    log.info("Done")

    # save
    if "vtu" in out:
        model.save(out_dir / out["vtu"])
    if legend:
        ax.legend(handles=legend, fontsize=7)
    ax.set_title(plot.get("title", "Principal stress directions"), y=1.08)
    fig.savefig(
        out_dir / out.get("figure", "principal_directions.png"),
        dpi=plot.get("dpi", 200), bbox_inches="tight",
    )
    plt.close(fig)
    log.info(f"Saved results in: {out_dir}")