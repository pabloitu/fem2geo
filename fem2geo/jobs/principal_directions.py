"""
Job: principal_directions
=========================
Probes a model at a given site by plotting principal stress directions on a
stereonet. By default, shows the model's average σ1/σ2/σ3 directions only.
Per-cell directions can be enabled to visualise the spread around the average.

Config reference
----------------
job: principal_directions
schema: adeli                       # built-in schema name (default: adeli)

model: path/to/model.vtk            # relative to this config file

site:
  center: [x, y, z]
  radius: r

plot:
  title: "Principal stress directions"
  figsize: [8, 8]
  dpi: 200
  avg_directions:                   # model average σ1/σ2/σ3 (default: show=true)
    show: true
    color: "red"
    marker: "o"
    markersize: 8
  cell_directions:                  # per-cell model directions (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "red"
    markersize: 3
    alpha: 0.4

output:
  dir: results/                     # optional, defaults to config file directory
  figure: principal_directions.png
  vtu: extract.vtu                  # save extracted sub-model for Paraview

Example
-------
fem2geo config.yaml
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fem2geo.model import Model
from fem2geo.internal.schema import ModelSchema
from fem2geo.plots import get_style, stereo_axes, stereo_axes_contour
from fem2geo.runner import resolve_output

log = logging.getLogger("fem2geoLogger")

AVG = {"color": "red", "markersize": 8, "markeredgecolor": "k"}
CELL = {"color": "red", "markersize": 3, "markeredgecolor": "none", "alpha": 0.4}
CONTOUR = {"color": "red", "levels": 4, "sigma": 2.0, "linewidth": 1.0}

LEGEND = [
    Line2D([0], [0], color="k", lw=0, marker="o", label=r"$\sigma_1$"),
    Line2D([0], [0], color="k", lw=0, marker="s", label=r"$\sigma_2$"),
    Line2D([0], [0], color="k", lw=0, marker="v", label=r"$\sigma_3$"),
]


def parse_common(cfg, job_dir):
    plot = cfg.get("plot", {})
    avg = plot.get("avg_directions", {})
    cell = plot.get("cell_directions", {})
    cell_style = cell.get("style", "scatter")
    cell_base = CONTOUR if cell_style == "contour" else CELL

    return {
        "schema": ModelSchema.builtin(cfg.get("schema", "adeli")),
        "model_path": (job_dir / cfg["model"]).resolve(),
        "title": plot.get("title", "Principal stress directions"),
        "figsize": plot.get("figsize", [8, 8]),
        "dpi": plot.get("dpi", 200),
        "avg_show": avg.get("show", True),
        "avg_style": get_style(AVG, avg),
        "cell_show": cell.get("show", False),
        "cell_style": cell_style,
        "cell_props": get_style(cell_base, cell),
        "out": resolve_output(cfg, job_dir),
    }


def parse_site(entry, job_dir):
    site = dict(entry)
    site["center"] = np.asarray(site["center"], dtype=float)
    return site


def parse(cfg, job_dir):
    params = parse_common(cfg, job_dir)
    params["site"] = parse_site(cfg["site"], job_dir)
    return params


def compute(ax, model, site, params):
    if params["cell_show"]:
        vecs = np.stack([model.dir_s1, model.dir_s2, model.dir_s3], axis=-1)
        if params["cell_style"] == "contour":
            stereo_axes_contour(ax, vecs, params["cell_props"])
        else:
            stereo_axes(ax, vecs, params["cell_props"])

    if params["avg_show"]:
        _, vec = model.avg_principals("stress")
        stereo_axes(ax, vec, params["avg_style"],
                    labels=(r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"))


def draw(model, site, params):
    fig = plt.figure(figsize=params["figsize"])
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    compute(ax, model, site, params)

    ax.legend(handles=LEGEND, fontsize=7)
    ax.set_title(params["title"], y=1.08)

    out = params["out"]
    fig.savefig(out["dir"] / out.get("figure", "principal_directions.png"),
                dpi=params["dpi"], bbox_inches="tight")
    plt.close(fig)
    return fig


def run(cfg, job_dir):
    params = parse(cfg, job_dir)
    site = params["site"]

    log.info(f"Loading {params['model_path']}")
    model = Model.from_file(params["model_path"], params["schema"])
    sub = model.extract(site["center"], site["radius"])
    log.info(f"  {sub.n_cells} cells in site")

    draw(sub, site, params)

    out = params["out"]
    if "vtu" in out:
        sub.save(out["dir"] / out["vtu"])

    log.info(f"Saved results in: {out['dir']}")