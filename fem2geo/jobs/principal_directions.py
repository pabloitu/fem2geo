"""
Job: principal_directions
=========================
Probes a model at a given site by plotting principal stress directions on a
stereonet. Shows the model's average principal directions, with optional
per-cell spread visualisation.

Config reference
----------------
job: principal_directions
schema: adeli                       # built-in schema name (default: adeli)
tensor: stress                      # stress | stress_dev | strain | strain_rate
                                    # strain_plastic | strain_elastic

model: path/to/model.vtk            # relative to this config file

site:
  center: [x, y, z]
  radius: r

plot:
  title: "Principal stress directions"
  figsize: [8, 8]
  dpi: 200
  legend_size: 8                    # scales legend
  legend_loc: "best"                # best | 1 (upper right) | 2 | 3 | 4
  principals:                       # model average
    color: "red"
    markersize: 8
  cell_principals:                  # per-cell directions (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "red"
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fem2geo.model import Model
from fem2geo.internal.schema import ModelSchema
from fem2geo.plots import get_style, stereo_axes, stereo_axes_contour
from fem2geo.runner import resolve_output
from fem2geo.utils.tensor import TENSOR_LABELS

log = logging.getLogger("fem2geoLogger")

AVG = {"color": "red", "markersize": 8, "markeredgecolor": "k"}
CELL = {"color": "red", "markersize": 3, "markeredgecolor": "k", "alpha": 0.4}
CONTOUR = {"color": "red", "levels": 4, "sigma": 2.0, "linewidth": 1.0}

MARKERS = ("o", "s", "v")


def parse_common(cfg, job_dir):
    plot = cfg.get("plot", {})
    avg = plot.get("principals", {})
    cell = plot.get("cell_principals", {})
    cell_style = cell.get("style", "scatter")
    cell_base = CONTOUR if cell_style == "contour" else CELL
    which = cfg.get("tensor", "stress")

    return {
        "schema": ModelSchema.builtin(cfg.get("schema", "adeli")),
        "model_path": (job_dir / cfg["model"]).resolve(),
        "which": which,
        "labels": TENSOR_LABELS[which],
        "title": plot.get("title", "Principal directions"),
        "figsize": plot.get("figsize", [8, 8]),
        "dpi": plot.get("dpi", 200),
        "legend_size": plot.get("legend_size", 8),
        "legend_loc": plot.get("legend_loc", "best"),
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
    legend = []
    labels = params["labels"]

    if params["cell_show"]:
        vecs = np.stack([model.dir_s1, model.dir_s2, model.dir_s3], axis=-1)
        if params["cell_style"] == "contour":
            stereo_axes_contour(ax, vecs, params["cell_props"])
        else:
            stereo_axes(ax, vecs, params["cell_props"])

    _, vec = model.avg_principals(params["which"])
    stereo_axes(ax, vec, params["avg_style"], labels=labels)

    legend.extend([
        Line2D([0], [0], color="k", lw=0, marker=MARKERS[i], label=labels[i])
        for i in range(3)
    ])

    if params["cell_show"]:
        color = params["cell_props"].get("color", "red")
        legend.append(
            Line2D([0], [0], color=color, lw=0, marker=".",
                   alpha=0.4, label="Cell Directions")
        )

    return legend


def draw(model, site, params):
    fig = plt.figure(figsize=params["figsize"])
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    legend = compute(ax, model, site, params)

    ax.legend(handles=legend, prop={"size": params["legend_size"]},
              loc=params["legend_loc"])
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
        print(out["dir"] / out["vtu"])
        sub.save(out["dir"] / out["vtu"])

    log.info(f"Saved results in: {out['dir']}")