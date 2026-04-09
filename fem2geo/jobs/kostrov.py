"""
Job: kostrov
============
Computes the Kostrov (1974) summed moment tensor from a fault population
and compares its principal axes with the model's average deviatoric stress
principal directions on a stereonet.

The Kostrov tensor represents the bulk kinematic strain implied by the
observed fault slip. If the model stress field is mechanically consistent
with the faulting, the Kostrov shortening axis should align with σ1 and
the extension axis with σ3.

Structural data is read from CSV files via
:func:`fem2geo.internal.io.load_structural_csv`. Only fault data
(``strike, dip, rake`` with signed rake, Aki & Richards convention) is
supported. Fracture-only data is skipped with a warning.

Config reference
----------------
job: kostrov
schema: adeli                       # built-in schema name (default: adeli)
units:                              # optional category-level unit overrides
  pressure: Pa

model: path/to/model.vtu            # relative to this config file

zone:
  type: sphere                      # sphere | box
  center: [x, y, z]
  radius: r                         # sphere only
  # dim: [dx, dy, dz]               # box only

data:
  faults: path/to/faults.csv        # columns: strike, dip, rake (signed)

tensor: strain                  #  strain | strain_rate | strain_plastic | strain_elastic

plot:
  title: "Kostrov analysis"
  figsize: [8, 8]
  dpi: 200
  kostrov:                          # Kostrov tensor principal axes
    color: "#E63946"                # red
    markersize: 10
  model:                            # model principal axes
    color: "#2196F3"                # blue
    markersize: 10
  cell_directions:                  # per-cell model directions (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "grey"
    markersize: 3
    alpha: 0.3
  data_spread:                      # per-fault P/T axes (default: show=false)
    show: false
    style: scatter                  # scatter | contour
    color: "#E63946"
    markersize: 3
    alpha: 0.3

output:
  dir: results/
  vtu: extract.vtu                  # optional, saves extracted sub-model

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fem2geo.data import FaultData
from fem2geo.internal.io import load_structural_csv
from fem2geo.model import Model
from fem2geo.plots import PlotConfig, stereo_axes, stereo_axes_contour
from fem2geo.runner import parse_config
from fem2geo.utils import tensor
from fem2geo.utils.tensor import kostrov_tensor, axes_misfit

log = logging.getLogger("fem2geoLogger")

KOSTROV_STYLE = PlotConfig(color="#E63946", markersize=10, markeredgecolor="k")
MODEL_STYLE = PlotConfig(color="#2196F3", markersize=10, markeredgecolor="k")
CELL_STYLE = PlotConfig(color="grey", markersize=3, alpha=0.3)
CONTOUR_STYLE = PlotConfig(color="grey", levels=4, sigma=2.0, linewidth=1.0)
DATA_STYLE = PlotConfig(color="#E63946", markersize=3, alpha=0.3)
DATA_CONTOUR_STYLE = PlotConfig(color="#E63946", levels=4, sigma=2.0, linewidth=1.0)

_TENSOR_LABELS = {
    "stress_dev": (r"$\sigma^{\mathrm{dev}}_1$", r"$\sigma^{\mathrm{dev}}_2$",
                   r"$\sigma^{\mathrm{dev}}_3$"),
    "strain": (r"$\epsilon_1$", r"$\epsilon_2$", r"$\epsilon_3$"),
    "strain_rate": (r"$\dot{\epsilon}_1$", r"$\dot{\epsilon}_2$",
                    r"$\dot{\epsilon}_3$"),
    "strain_plastic": (r"$\epsilon^p_1$", r"$\epsilon^p_2$", r"$\epsilon^p_3$"),
    "strain_elastic": (r"$\epsilon^e_1$", r"$\epsilon^e_2$", r"$\epsilon^e_3$"),
}

_TENSOR_SYMBOL = {
    "stress_dev": r"$\sigma^{\mathrm{dev}}$",
    "strain": r"$\epsilon$",
    "strain_rate": r"$\dot{\epsilon}$",
    "strain_plastic": r"$\epsilon^p$",
    "strain_elastic": r"$\epsilon^e$",
}

_K_LABELS = [r"$K_1$ (shortening)", r"$K_2$ (intermediate)", r"$K_3$ (extension)"]
_K_LOG = ["K1 (short.)", "K2 (int.)", "K3 (ext.)"]


def run(cfg: dict, job_dir: Path) -> None:
    schema, zone, data, plot, out = parse_config(cfg, job_dir)
    out_dir = Path(out.get("dir", job_dir))
    which = cfg.get("tensor", "strain")

    kostrov_cfg = plot.get("kostrov", {})
    model_cfg = plot.get("model", {})
    cell_cfg = plot.get("cell_directions", {})
    data_cfg = plot.get("data_spread", {})

    k_style = KOSTROV_STYLE.update(kostrov_cfg)
    m_style = MODEL_STYLE.update(model_cfg)
    show_cell = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")
    cell_pc = (CONTOUR_STYLE if cell_style == "contour"
               else CELL_STYLE).update(cell_cfg)
    show_data = data_cfg.get("show", False)
    data_style = data_cfg.get("style", "scatter")
    data_pc = (DATA_CONTOUR_STYLE if data_style == "contour"
               else DATA_STYLE).update(data_cfg)

    # load model and extract zone
    model_path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {model_path}")
    model = Model.from_file(model_path, schema)
    sub = model.extract(zone)
    log.info(f"  {sub.n_cells} cells in zone")

    # load fault datasets
    all_strikes, all_dips, all_rakes = [], [], []
    for name, entry in cfg["data"].items():
        file_path = entry if isinstance(entry, str) else entry.get("file")
        sd = load_structural_csv((job_dir / file_path).resolve())

        log.info(f"  '{name}': {len(sd)} faults")
        all_strikes.append(sd.planes[:, 0])
        all_dips.append(sd.planes[:, 1])
        all_rakes.append(sd.rakes)

    strikes = np.concatenate(all_strikes)
    dips = np.concatenate(all_dips)
    rakes = np.concatenate(all_rakes)
    log.info(f"  Total: {len(strikes)} faults for Kostrov summation")

    # Kostrov tensor principals
    K = kostrov_tensor(strikes, dips, rakes)
    k_vecs = tensor.eigenvectors(K[None, :])[0]

    # model tensor principals
    axis_labels = _TENSOR_LABELS[which]
    m_vals, m_vecs = sub.avg_principals(which)

    # angular misfit between Kostrov and model axes
    log.info(f"  Misfit between Kostrov and Model tensors ({which}): {m_vals}")

    angles, pairs = axes_misfit(k_vecs, m_vecs)
    for idx, (i, j) in enumerate(pairs):
        log.info(f"  {_K_LOG[i]} <-> {axis_labels[j]}: {angles[idx]:.1f} deg")

    # figure
    fig = plt.figure(figsize=plot.get("figsize", [8, 8]))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    # cell-level model directions
    if show_cell:
        cell_vecs = np.stack([sub.dir_s1, sub.dir_s2, sub.dir_s3], axis=-1)
        if cell_style == "contour":
            stereo_axes_contour(ax, cell_vecs, cell_pc)
        else:
            stereo_axes(ax, cell_vecs, cell_pc)

    # per-fault P/B/T axes via individual Kostrov dyads
    if show_data:
        dyads = np.array(
            [kostrov_tensor(s, d, r) for s, d, r in zip(strikes, dips, rakes)])
        per_vecs = tensor.eigenvectors(dyads)
        if data_style == "contour":
            stereo_axes_contour(ax, per_vecs, data_pc)
        else:
            stereo_axes(ax, per_vecs, data_pc)

    # Kostrov principal axes
    stereo_axes(ax, k_vecs, k_style, labels=_K_LABELS)

    # model principal axes
    stereo_axes(ax, m_vecs, m_style, labels=axis_labels)

    # legend
    sym = _TENSOR_SYMBOL[which]
    legend_elements = [
        Line2D([0], [0], color=k_style.color, linewidth=0, marker="o", markersize=8,
               label="Kostrov axes"),
        Line2D([0], [0], color=m_style.color, linewidth=0, marker="o", markersize=8,
               label=f"{sym} axes"),
        Line2D([], [], color="k", linewidth=0, marker="o", markersize=6,
               label=rf"$K_1$, {axis_labels[0]}"),
        Line2D([], [], color="k", linewidth=0, marker="s", markersize=6,
               label=rf"$K_2$, {axis_labels[1]}"),
        Line2D([], [], color="k", linewidth=0, marker="v", markersize=6,
               label=rf"$K_3$, {axis_labels[2]}"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(plot.get("title", "Kostrov analysis"), y=1.08)

    # save
    if "vtu" in out:
        sub.save(out_dir / out.get("vtu", "extract.vtu"))
    fig.savefig(out_dir / out.get("figure", "kostrov.png"),
                dpi=plot.get("dpi", 200), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved results: {out_dir}")