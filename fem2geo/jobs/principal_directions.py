"""
Job: principal_directions
=========================
Probes a model at a given location by plotting principal stress directions
on a stereonet. By default, shows the average stress directions only. Cell
directions can be enabled to visualise the spread around the average.
Multiple models can be compared at the same zone — each gets a distinct
colour.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fem2geo.model import Model
from fem2geo.plots import PlotConfig, MODEL_COLORS, stereo_line, stereo_contour
from fem2geo.runner import parse_config
from fem2geo.utils.transform import line_enu2sphe

log = logging.getLogger("fem2geoLogger")

# default styles
AVG_STYLE = PlotConfig(
    color="red", markersize=8, markeredgecolor="k",
)
CELL_STYLE = PlotConfig(
    color="red", markersize=3, markeredgecolor="none", alpha=0.4,
)
CONTOUR_STYLE = PlotConfig(
    color="red", levels=4, sigma=2.0, linewidth=1.0,
)


# main job
def run(cfg: dict, job_dir: Path) -> None:

    # parse configuration file
    # schema: map file fields to internal (see fem2geo/internal/schemas)
    # zone, plot, data and out: groups in config file
    schema, zone, data, plot, out = parse_config(cfg, job_dir)
    out_dir = out["dir"]

    # model paths
    models = cfg.get("models", {"model": cfg.get("model")})

    # plot options
    avg_cfg = plot.get("avg_directions", {})    # Config for model average eigenvectors
    avg_show = avg_cfg.get("show", True)
    avg_plot_config = AVG_STYLE.update(avg_cfg)

    cell_cfg = plot.get("cell_directions", {}) # Config for each cell eigenvectors
    cell_show = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")  # plotted as scatter or contour
    cell_plot_config = CONTOUR_STYLE.update(cell_cfg) if cell_style == "contour" \
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

    # per-model loop
    colors = MODEL_COLORS[: len(models)]
    for color, name in zip(colors, models):
        path = (job_dir / models[name]).resolve() # get model's filepath
        log.info(f"Loading {name}: {path}")

        model = Model.from_file(path, schema)   # Load model
        model = model.extract(zone)             # Select zone to probe

        log.info(f"Processing results...")
        if cell_show:
            plunge1, azimuth1 = line_enu2sphe(model.dir_s1)  # ENU to Spherical coords
            plunge2, azimuth2 = line_enu2sphe(model.dir_s2)
            plunge3, azimuth3 = line_enu2sphe(model.dir_s3)

            cpc = cell_plot_config.update(color=color)  # Update by model's color

            if cell_style == "contour":
                stereo_contour(ax, plunge1, azimuth1, **cpc.kwargs())
                stereo_contour(ax, plunge2, azimuth2, **cpc.kwargs())
                stereo_contour(ax, plunge3, azimuth3, **cpc.kwargs())
            else:
                stereo_line(ax, plunge1, azimuth1, **cpc.update(marker="o").kwargs())
                stereo_line(ax, plunge2, azimuth2, **cpc.update(marker="s").kwargs())
                stereo_line(ax, plunge3, azimuth3, **cpc.update(marker="v").kwargs())

        if avg_show:
            _, vec = model.avg_principal()  # get eigenvectors of averaged tensor

            label = name if len(models) > 1 else None
            plunge1, azimuth1 = line_enu2sphe(vec[:, 0])
            plunge2, azimuth2 = line_enu2sphe(vec[:, 1])
            plunge3, azimuth3 = line_enu2sphe(vec[:, 2])

            apc = avg_plot_config.update(color=color)
            stereo_line(
                ax,
                plunge1,
                azimuth1,
                label=label,
                **apc.update(marker="o").kwargs(),
            )
            stereo_line(ax, plunge2, azimuth2, **apc.update(marker="s").kwargs())
            stereo_line(ax, plunge3, azimuth3, **apc.update(marker="v").kwargs())

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
