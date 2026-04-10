"""
Job: kostrov
============
Computes the Kostrov (1974) summed moment tensor from a fault population and
compares its principal axes with the model's tensor principal directions. The
Kostrov tensor represents the bulk kinematic strain implied by the observed
fault slip.

Config reference
----------------
job: kostrov
schema: adeli                       # built-in schema name (default: adeli)
tensor: strain                      # strain | strain_rate | strain_plastic
                                    # strain_elastic | stress_dev

model: path/to/model.vtu            # relative to this config file

site:
  center: [x, y, z]
  radius: r
  data: path/to/faults.csv          # columns: strike, dip, rake (signed)

plot:
  title: "Kostrov analysis"
  figsize: [8, 8]
  dpi: 200
  legend_size: 8
  legend_loc: "best"
  principals:                       # model tensor axes (always shown)
    color: "#2196F3"
    markersize: 10
  kostrov:                          # Kostrov tensor axes (always shown)
    color: "#E63946"
    markersize: 10
  fault_axes:                       # per-fault P/B/T (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "#E63946"
    markersize: 3
    alpha: 0.3

output:
  dir: results/
  figure: kostrov.png
  vtu: extract.vtu

Example
-------
fem2geo config.yaml
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fem2geo.data import FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import get_style, stereo_axes, stereo_axes_contour
from fem2geo.runner import resolve_output
from fem2geo.utils import tensor
from fem2geo.utils.tensor import (
    TENSOR_LABELS, TENSOR_SYMBOL, kostrov_tensor, axes_misfit,
)

log = logging.getLogger("fem2geoLogger")

AVG = {"color": "#2196F3", "markersize": 10, "markeredgecolor": "k"}
KOSTROV = {"color": "#E63946", "markersize": 10, "markeredgecolor": "k"}
FAULT = {"color": "#E63946", "markersize": 3, "alpha": 0.3}
FAULT_CONTOUR = {"color": "#E63946", "levels": 4, "sigma": 2.0, "linewidth": 1.0}

K_LABELS = (r"$K_1$ (shortening)", r"$K_2$ (intermediate)", r"$K_3$ (extension)")
K_LOG = ("K1 (short.)", "K2 (int.)", "K3 (ext.)")
MARKERS = ("o", "s", "v")


def parse_common(cfg, job_dir):
    plot = cfg.get("plot", {})
    avg = plot.get("principals", {})
    k_cfg = plot.get("kostrov", {})
    fa = plot.get("fault_axes", {})
    fa_style = fa.get("style", "scatter")
    fa_base = FAULT_CONTOUR if fa_style == "contour" else FAULT
    which = cfg.get("tensor", "strain")

    return {
        "schema": ModelSchema.builtin(cfg.get("schema", "adeli")),
        "model_path": (job_dir / cfg["model"]).resolve(),
        "job_dir": job_dir,
        "which": which,
        "labels": TENSOR_LABELS[which],
        "symbol": TENSOR_SYMBOL[which],
        "title": plot.get("title", "Kostrov analysis"),
        "figsize": plot.get("figsize", [8, 8]),
        "dpi": plot.get("dpi", 200),
        "legend_size": plot.get("legend_size", 8),
        "legend_loc": plot.get("legend_loc", "best"),
        "avg_style": get_style(AVG, avg),
        "k_style": get_style(KOSTROV, k_cfg),
        "fa_show": fa.get("show", False),
        "fa_style": fa_style,
        "fa_props": get_style(fa_base, fa),
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
    labels = params["labels"]
    fd = site["faults"]
    strikes, dips, rakes = fd.planes[:, 0], fd.planes[:, 1], fd.rakes

    # per-fault P/B/T axes
    if params["fa_show"]:
        dyads = np.array([
            kostrov_tensor(s, d, r) for s, d, r in zip(strikes, dips, rakes)
        ])
        per_vecs = tensor.eigenvectors(dyads)
        if params["fa_style"] == "contour":
            stereo_axes_contour(ax, per_vecs, params["fa_props"])
        else:
            stereo_axes(ax, per_vecs, params["fa_props"])

    # kostrov tensor
    K = kostrov_tensor(strikes, dips, rakes)
    k_vecs = tensor.eigenvectors(K[None, :])[0]
    stereo_axes(ax, k_vecs, params["k_style"], labels=K_LABELS)

    # model tensor
    m_vals, m_vecs = model.avg_principals(params["which"])
    stereo_axes(ax, m_vecs, params["avg_style"], labels=labels)

    # misfit
    angles, pairs = axes_misfit(k_vecs, m_vecs)
    for idx, (i, j) in enumerate(pairs):
        log.info(f"  {K_LOG[i]} <-> {labels[j]}: {angles[idx]:.1f} deg")

    # legend
    sym = params["symbol"]
    legend.extend([
        Line2D([0], [0], color=params["k_style"]["color"], lw=0,
               marker="o", markersize=8, label="Kostrov axes"),
        Line2D([0], [0], color=params["avg_style"]["color"], lw=0,
               marker="o", markersize=8, label=f"{sym} axes"),
    ])
    legend.extend([
        Line2D([0], [0], color="k", lw=0, marker=MARKERS[i],
               label=rf"$K_{i+1}$, {labels[i]}")
        for i in range(3)
    ])
    if params["fa_show"]:
        legend.append(
            Line2D([0], [0], color=params["fa_props"].get("color", "#E63946"),
                   lw=0, marker=".", alpha=0.3, label="Individual faults")
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
    fig.savefig(out["dir"] / out.get("figure", "kostrov.png"),
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