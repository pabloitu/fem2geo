"""
Job: sites
==========
Dispatcher for running any atomic job over multiple sites, assembling per-site
panels into a single figure. Optionally, each site can also be saved as its own
figure by setting ``output.per_site: true``.

Config reference
----------------
job: sites.<inner>                  # e.g. sites.principal_directions
schema: adeli
model: path/to/model.vtu

sites:
  name_a:
    center: [x, y, z]
    radius: r
    title: "Custom panel title"     # optional, defaults to site key
  name_b:
    center: [x, y, z]
    radius: r

plot:
  title: "Figure-level suptitle"
  figsize: [12, 8]
  dpi: 200
  legend_size: 14
  grid: [rows, cols]                # optional, auto closest-to-square

output:
  dir: results/
  figure: sites.png
  per_site: false                   #  write each site as its own file

Example
-------
fem2geo config.yaml
"""

import logging
import math

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh

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
        "legend_size": plot.get("legend_size", 14),
        "inner": inner,
        "inner_name": name,
        "sites": sites,
        "grid": plot.get("grid"),
        "per_site": cfg.get("output", {}).get("per_site", False),
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
    inner = params["inner"]

    fig = plt.figure(figsize=params["figsize"], layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.0, wspace=0.0)

    sup = None
    if params["title"]:
        sup = fig.suptitle(params["title"], fontsize=12)

    legend_handles = None
    mappable = None

    for i, (name, site) in enumerate(sites.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection="stereonet")
        ax.grid(True)
        ax.set_azimuth_ticks([])

        sub = model.extract(site["center"], site["radius"])
        log.info(f"  [{name}] {sub.n_cells} cells")

        handles = inner.compute(ax, sub, site, params, cbar=False)

        if legend_handles is None:
            legend_handles = handles

        for c in ax.collections:
            if isinstance(c, QuadMesh):
                mappable = c

        ax.set_title(site.get("title", name), fontsize=9, y=1.02)

    # shared legend (top right)
    leg = None
    if legend_handles:
        leg = fig.legend(
            handles=legend_handles,
            prop={"size": params["legend_size"]},
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
        )

    # shared colorbar (tendency only)
    if mappable is not None:
        cb = params.get("cbar_opts", {})
        panel_axes = [a for a in fig.axes if a.get_subplotspec() is not None]
        fig.colorbar(
            mappable,
            ax=panel_axes,
            label=tendency.CBAR_LABELS.get(params.get("variant", ""), ""),
            shrink=cb.get("shrink", 0.6),
            pad=cb.get("pad", 0.04),
            orientation=cb.get("orientation", "vertical"),
            ticks=cb["levels"] if isinstance(cb.get("levels"), (list, tuple)) else None,
        )

    out = params["out"]
    figname = out.get("figure", f"sites_{params['inner_name']}.png")
    extras = [a for a in (leg, sup) if a is not None]
    fig.savefig(out["dir"] / figname, dpi=params["dpi"],
                bbox_inches="tight", bbox_extra_artists=extras)
    plt.close(fig)

    if params["per_site"]:
        save_per_site(model, params)

    return fig


def save_per_site(model, params):
    """Render each site as its own standalone figure via the inner job's draw."""
    inner = params["inner"]
    sites = params["sites"]
    out = params["out"]
    original_figname = out.get("figure")

    for name, site in sites.items():
        sub = model.extract(site["center"], site["radius"])
        out["figure"] = f"{name}.png"
        inner.draw(sub, site, params)

    if original_figname is None:
        out.pop("figure", None)
    else:
        out["figure"] = original_figname


def run(cfg, job_dir):
    params = parse(cfg, job_dir)

    log.info(f"Loading {params['model_path']}")
    model = Model.from_file(params["model_path"], params["schema"])

    draw(model, params)

    log.info(f"Saved results in: {params['out']['dir']}")