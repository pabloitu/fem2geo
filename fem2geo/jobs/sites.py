"""
Job: sites
==========
Dispatcher for running any atomic job over multiple sites. Reuses the atomic
job's ``parse_common``, ``parse_site``, and ``compute`` functions, and assembles
the per-site panels into a single grid figure.

Config reference
----------------
job: sites.<inner>                  # e.g. sites.principal_directions
schema: adeli
model: path/to/model.vtu

sites:
  name_a:
    center: [x, y, z]
    radius: r
    # data: path/to/data.csv        # if the inner job uses data
    # title: "Custom panel title"   # optional, defaults to the site key
  name_b:
    center: [x, y, z]
    radius: r

plot:
  title: "Figure-level suptitle"
  figsize: [12, 8]
  dpi: 200
  grid: [rows, cols]                # optional, auto closest-to-square
  # all other plot keys are forwarded to the inner job and applied uniformly
  # to every panel (e.g. avg_directions, cell_directions)

output:
  dir: results/
  figure: sites.png

Example
-------
fem2geo config.yaml
"""

import logging
import math

import matplotlib.pyplot as plt

from fem2geo.jobs import (
    principal_directions, fracture, resolved_shear, kostrov, tendency,
)
from fem2geo.model import Model

log = logging.getLogger("fem2geoLogger")

INNER = {
    "principal_directions": principal_directions,
    "fracture":              fracture,
    "resolved_shear":        resolved_shear,
    "kostrov":               kostrov,
    "tendency":              tendency,
}


def parse(cfg, job_dir):
    name = cfg["job"].split(".", 1)[1]
    if name not in INNER:
        raise ValueError(f"Unknown inner job '{name}'")
    inner = INNER[name]

    common = inner.parse_common(cfg, job_dir)
    sites = {
        key: inner.parse_site(entry, job_dir)
        for key, entry in cfg["sites"].items()
    }

    plot = cfg.get("plot", {})
    return {
        **common,
        "inner": inner,
        "inner_name": name,
        "sites": sites,
        "grid": plot.get("grid"),
    }


def grid_shape(n, override=None):
    if override:
        return tuple(override)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def draw(model, params):
    sites = params["sites"]
    rows, cols = grid_shape(len(sites), params["grid"])

    fig = plt.figure(figsize=params["figsize"])
    inner = params["inner"]
    legend = getattr(inner, "LEGEND", None)

    for i, (name, site) in enumerate(sites.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection="stereonet")
        ax.grid(True)

        sub = model.extract(site["center"], site["radius"])
        log.info(f"  [{name}] {sub.n_cells} cells")

        inner.compute(ax, sub, site, params)

        if legend is not None:
            ax.legend(handles=legend, fontsize=6)
        ax.set_title(site.get("title", name), y=1.08)

    if params["title"]:
        fig.suptitle(params["title"])

    out = params["out"]
    figname = out.get("figure", f"sites_{params['inner_name']}.png")
    fig.savefig(out["dir"] / figname, dpi=params["dpi"], bbox_inches="tight")
    plt.close(fig)
    return fig


def run(cfg, job_dir):
    params = parse(cfg, job_dir)

    log.info(f"Loading {params['model_path']}")
    model = Model.from_file(params["model_path"], params["schema"])

    draw(model, params)

    log.info(f"Saved results in: {params['out']['dir']}")