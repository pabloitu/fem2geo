"""
Job: tendency
=============
Plots slip, dilation, or summarized reactivation tendency on a stereonet for
one model and one site. The model's average principal directions are placed
on the tendency field. An optional dataset (fractures or faults) can be drawn
as poles on top.

Tendency variants:

- ``slip``: normalized slip tendency Ts' (Morris and Ferrill, 1996), range [0, 1].
- ``dilation``: dilation tendency Td (Ferrill et al., 2020), range [0, 1].
- ``summarized``: Ts' + Td (Jolie et al., 2016), range [0, 2].

Config reference
----------------
job: tendency
schema: adeli                       # built-in schema name (default: adeli)
variant: slip                       # slip | dilation | summarized

model: path/to/model.vtu            # relative to this config file

site:
  center: [x, y, z]
  radius: r
  data: path/to/data.csv            # optional fracture or fault CSV

plot:
  title: "Slip tendency"            # optional, auto-generated if empty
  figsize: [8, 8]
  dpi: 200
  legend_size: 8
  legend_loc: "best"
  stress_values: false              # append σ1, σ3, φ to the title
  principals:                       # model average (always shown)
    color: "white"
    markersize: 8
  poles:                            # data pole overlay (fractures or faults)
    color: "black"
    marker: "+"
    markersize: 5
    alpha: 0.7
  colorbar:
    cmap: "viridis"
    levels: 10                      # int (n bins) or list of bin edges
    shrink: 0.6
    pad: 0.08
    orientation: "vertical"

output:
  dir: results/
  figure: tendency.png
  vtu: extract.vtu

Example
-------
fem2geo config.yaml
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fem2geo.internal.io import load_structural_csv
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import get_style, stereo_field, stereo_axes, stereo_pole
from fem2geo.runner import resolve_output
from fem2geo.utils.tensor import (
    slip_tendency, dilation_tendency, summarized_tendency,
)
from fem2geo.utils.transform import grid_nodes, grid_centers

log = logging.getLogger("fem2geoLogger")

AVG = {"color": "white", "markersize": 8, "markeredgecolor": "k"}
POLES = {"color": "black", "markersize": 5, "alpha": 0.7, "marker": "+"}
CBAR = {"cmap": "magma", "levels": None, "shrink": 0.6, "pad": 0.08,
        "orientation": "vertical"}

LABELS = (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$")
MARKERS = ("o", "s", "v")

FUNCTIONS = {
    "slip": slip_tendency,
    "dilation": dilation_tendency,
    "summarized": summarized_tendency,
}

CBAR_LABELS = {
    "slip": r"Slip tendency $T'_s$",
    "dilation": r"Dilation tendency $T_d$",
    "summarized": r"Summarized tendency $T'_s + T_d$",
}

TITLES = {
    "slip": "Slip tendency",
    "dilation": "Dilation tendency",
    "summarized": "Summarized tendency",
}

VMAX = {"slip": 1.0, "dilation": 1.0, "summarized": 2.0}


def parse_common(cfg, job_dir):
    plot = cfg.get("plot", {})
    avg = plot.get("principals", {})
    poles = plot.get("poles", {})
    cbar = plot.get("colorbar", {})
    variant = cfg.get("variant", "slip")
    if variant not in FUNCTIONS:
        raise ValueError(
            f"Unknown variant '{variant}'. Valid: {list(FUNCTIONS)}"
        )

    merged_cbar = {**CBAR, **cbar}

    return {
        "schema": ModelSchema.builtin(cfg.get("schema", "adeli")),
        "model_path": (job_dir / cfg["model"]).resolve(),
        "job_dir": job_dir,
        "variant": variant,
        "title": plot.get("title", ""),
        "figsize": plot.get("figsize", [8, 8]),
        "dpi": plot.get("dpi", 200),
        "legend_size": plot.get("legend_size", 8),
        "legend_loc": plot.get("legend_loc", "best"),
        "stress_values": plot.get("stress_values", False),
        "n_strikes": plot.get("n_strikes", 180),
        "n_dips": plot.get("n_dips", 45),
        "avg_style": get_style(AVG, avg),
        "pole_style": get_style(POLES, poles),
        "cbar_opts": merged_cbar,
        "out": resolve_output(cfg, job_dir),
    }


def parse_site(entry, job_dir):
    site = dict(entry)
    site["center"] = np.asarray(site["center"], dtype=float)
    if "data" in entry:
        site["poles"] = load_structural_csv(
            (job_dir / entry["data"]).resolve()
        )
    else:
        site["poles"] = None
    return site


def parse(cfg, job_dir):
    params = parse_common(cfg, job_dir)
    params["site"] = parse_site(cfg["site"], job_dir)
    return params


def compute(ax, model, site, params, cbar=True):
    legend = []
    variant = params["variant"]
    ps = params["pole_style"]
    cb = params["cbar_opts"]

    avg_stress = model.avg_tensor("stress")
    ms, md = grid_nodes(params["n_strikes"], params["n_dips"])
    cs, cd = grid_centers(ms, md)
    vals = FUNCTIONS[variant](
        avg_stress, cs.ravel(), cd.ravel()
    ).reshape(cs.shape)

    stereo_field(
        ax, ms, md, vals,
        cmap=cb["cmap"],
        vmin=0.0, vmax=VMAX[variant],
        levels=cb["levels"],
        cbar=cbar,
        cbar_label=CBAR_LABELS[variant],
        cbar_kwargs={"shrink": cb["shrink"], "pad": cb["pad"],
                     "orientation": cb["orientation"]},
    )

    _, vec = model.avg_principals("stress")
    stereo_axes(ax, vec, params["avg_style"], labels=LABELS)

    legend.extend([
        Line2D([0], [0], color=params["avg_style"].get("color", "white"),
               markeredgecolor="k", lw=0, marker=MARKERS[i], label=LABELS[i])
        for i in range(3)
    ])

    if site["poles"] is not None:
        fd = site["poles"]
        stereo_pole(ax, fd.planes[:, 0], fd.planes[:, 1], **ps)
        legend.append(
            Line2D([0], [0], color=ps.get("color", "black"), lw=0,
                   marker=ps.get("marker", "+"), label="poles")
        )

    return legend


def draw(model, site, params):
    fig = plt.figure(figsize=params["figsize"])
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)
    ax.set_azimuth_ticks([])

    legend = compute(ax, model, site, params, cbar=True)

    val, _ = model.avg_principals("stress")
    title = params["title"] or TITLES[params["variant"]]
    if params["stress_values"]:
        phi = float((val[1] - val[2]) / (val[0] - val[2]))
        title = (
            f"{title}\n"
            rf"$\sigma_1={val[0]:.3f}$, "
            rf"$\sigma_3={val[2]:.3f}$, "
            rf"$\phi={phi:.2f}$"
        )
    ax.set_title(title, y=1.05)

    ax.legend(handles=legend, prop={"size": params["legend_size"]},
              loc=params["legend_loc"])

    out = params["out"]
    fig.savefig(out["dir"] / out.get("figure", "tendency.png"),
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