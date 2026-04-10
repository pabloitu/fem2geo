"""
Job: resolved_shear
===================
Compares observed fault slip directions with the shear traction predicted by
the model's average stress tensor. Assumes the Wallace-Bott hypothesis.

For each fault plane (strike/dip/rake, Aki & Richards convention), the job
plots the fault plane as a great circle, the observed slip direction as an
arrow, and the predicted slip direction (resolved from the model) as a second
arrow. Overlap indicates consistency; divergence indicates a stress field
inconsistent with the observed faulting.

Config reference
----------------
job: resolved_shear
schema: adeli                       # built-in schema name (default: adeli)

model: path/to/model.vtu            # relative to this config file

site:
  center: [x, y, z]
  radius: r
  data: path/to/faults.csv          # columns: strike, dip, rake (signed)

plot:
  title: "Resolved shear analysis"
  figsize: [8, 8]
  dpi: 200
  legend_size: 8
  legend_loc: "best"
  principals:                       # model average σ1/σ2/σ3 (always shown)
    color: "k"
    markersize: 8
  planes:                           # fault great circles
    color: "grey"
    linewidth: 0.8
    alpha: 0.5
  observed:                         # observed slip arrows
    color: "#E63946"
    linewidth: 1.5
    arrowsize: 1.0
  predicted:                        # predicted slip arrows
    color: "#2196F3"
    linewidth: 1.5
    arrowsize: 1.0

output:
  dir: results/
  figure: resolved_shear.png
  vtu: extract.vtu

Example
-------
fem2geo config.yaml
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from fem2geo.data import FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import get_style, stereo_axes, stereo_plane, stereo_slip_arrow
from fem2geo.runner import resolve_output
from fem2geo.utils.tensor import resolved_rake

log = logging.getLogger("fem2geoLogger")

AVG = {"color": "k", "markersize": 8, "markeredgecolor": "k"}
OBS = {"color": "#E63946", "arrowsize": 1.0, "linewidth": 1.5}
PRED = {"color": "#2196F3", "arrowsize": 1.0, "linewidth": 1.5}
PLANE = {"color": "grey", "alpha": 0.5, "linewidth": 0.8}

LABELS = (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$")
MARKERS = ("o", "s", "v")


def parse_common(cfg, job_dir):
    plot = cfg.get("plot", {})
    avg = plot.get("principals", {})
    planes = plot.get("planes", {})
    obs = plot.get("observed", {})
    pred = plot.get("predicted", {})

    return {
        "schema": ModelSchema.builtin(cfg.get("schema", "adeli")),
        "model_path": (job_dir / cfg["model"]).resolve(),
        "job_dir": job_dir,
        "title": plot.get("title", "Resolved shear analysis"),
        "figsize": plot.get("figsize", [8, 8]),
        "dpi": plot.get("dpi", 200),
        "legend_size": plot.get("legend_size", 8),
        "legend_loc": plot.get("legend_loc", "best"),
        "avg_style": get_style(AVG, avg),
        "plane_style": get_style(PLANE, planes),
        "obs_style": get_style(OBS, obs),
        "pred_style": get_style(PRED, pred),
        "out": resolve_output(cfg, job_dir),
    }


def parse_site(entry, job_dir):
    site = dict(entry)
    site["center"] = np.asarray(site["center"], dtype=float)
    sd = load_structural_csv((job_dir / site["data"]).resolve())
    if not isinstance(sd, FaultData):
        raise ValueError(
            f"Fault data requires strike, dip, rake columns: {site['data']}"
        )
    site["faults"] = sd
    return site


def parse(cfg, job_dir):
    params = parse_common(cfg, job_dir)
    params["site"] = parse_site(cfg["site"], job_dir)
    return params


def compute(ax, model, site, params):
    legend = []
    fd = site["faults"]
    strikes, dips, rakes = fd.planes[:, 0], fd.planes[:, 1], fd.rakes
    ps = params["plane_style"]
    os = params["obs_style"]
    prs = params["pred_style"]

    # fault planes
    stereo_plane(ax, strikes, dips, **ps)

    # observed slip arrows
    stereo_slip_arrow(ax, strikes, dips, rakes, **os)

    # predicted slip arrows
    avg_stress = model.avg_tensor("stress")
    pred = resolved_rake(avg_stress, strikes, dips)
    stereo_slip_arrow(ax, strikes, dips, pred, **prs)

    # principal directions
    _, vec = model.avg_principals("stress")
    stereo_axes(ax, vec, params["avg_style"], labels=LABELS)

    legend.extend([
        FancyArrowPatch((0, 0), (0.02, 0), arrowstyle="->",
                        color=os["color"], lw=os["linewidth"],
                        label="Observed slip"),
        FancyArrowPatch((0, 0), (0.02, 0), arrowstyle="->",
                        color=prs["color"], lw=prs["linewidth"],
                        label="Predicted slip"),
        Line2D([0], [0], color=ps["color"], linewidth=ps["linewidth"],
               alpha=ps.get("alpha", 0.5), label="Fault planes"),
    ])
    legend.extend([
        Line2D([0], [0], color="k", lw=0, marker=MARKERS[i], label=LABELS[i])
        for i in range(3)
    ])

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
    fig.savefig(out["dir"] / out.get("figure", "resolved_shear.png"),
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