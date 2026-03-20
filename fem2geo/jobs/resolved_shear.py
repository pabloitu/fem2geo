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
  faults:
    file: path/to/faults.csv        # columns: strike, dip, rake (signed)

compare: stress_dev                 # stress_dev | strain | strain_rate | strain_plastic | strain_elastic

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
  save_vtu: false

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
from fem2geo.internal.schema import ModelSchema
from fem2geo.model import Model
from fem2geo.plots import PlotConfig, stereo_line, stereo_contour
from fem2geo.utils.tensor import kostrov_tensor
from fem2geo.utils import transform as tr
from fem2geo.utils.transform import line_enu2sphe

log = logging.getLogger("fem2geoLogger")


def run(cfg: dict, job_dir: Path) -> None:
    # config
    schema = ModelSchema.builtin(cfg.get("schema", "adeli"), units=cfg.get("units"))
    zone_cfg = cfg["zone"]
    plot_cfg = cfg.get("plot", {})
    out_cfg = cfg.get("output", {})

    title = plot_cfg.get("title", "Kostrov analysis")
    figsize = plot_cfg.get("figsize", [8, 8])
    dpi = plot_cfg.get("dpi", 200)
    out_dir = Path(out_cfg.get("dir", job_dir))
    save_vtu = out_cfg.get("save_vtu", False)
    compare = cfg.get("compare", "stress_dev")

    kostrov_cfg = plot_cfg.get("kostrov", {})
    model_cfg = plot_cfg.get("model", {})
    cell_cfg = plot_cfg.get("cell_directions", {})
    k_style = PlotConfig.avg(color=kostrov_cfg.get("color", "#E63946")).update(kostrov_cfg)
    m_style = PlotConfig.avg(color=model_cfg.get("color", "#2196F3")).update(model_cfg)

    show_cell = cell_cfg.get("show", False) if isinstance(cell_cfg, dict) else cell_cfg
    cell_style = cell_cfg.get("style", "scatter") if isinstance(cell_cfg, dict) else "scatter"
    cell_pc = (PlotConfig.density() if cell_style == "contour"
               else PlotConfig.cell(color="grey")).update(
        cell_cfg if isinstance(cell_cfg, dict) else {})

    data_cfg = plot_cfg.get("data_spread", {})
    show_data = data_cfg.get("show", False) if isinstance(data_cfg, dict) else data_cfg
    data_style = data_cfg.get("style", "scatter") if isinstance(data_cfg, dict) else "scatter"
    data_pc = (PlotConfig.density() if data_style == "contour"
               else PlotConfig.cell(color="#E63946")).update(
        data_cfg if isinstance(data_cfg, dict) else {})

    if "data" not in cfg or not cfg["data"]:
        raise ValueError("Config must contain a non-empty 'data' section.")
    if "model" not in cfg:
        raise ValueError("Config must contain a 'model' key.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # load model and extract zone
    model_path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {model_path}")
    model = Model.from_file(model_path, schema)

    if zone_cfg["type"] == "sphere":
        sub = model.extract_sphere(zone_cfg["center"], zone_cfg["radius"])
    elif zone_cfg["type"] == "box":
        sub = model.extract_box(zone_cfg["center"], np.asarray(zone_cfg["dim"]))
    else:
        raise ValueError(f"Unknown zone type '{zone_cfg['type']}'.")

    log.info(f"  {sub.n_cells} cells in zone")

    if save_vtu:
        sub.save(out_dir / "extract.vtu")

    # load fault datasets
    data_entries = cfg["data"]
    all_strikes, all_dips, all_rakes = [], [], []

    for name, entry in data_entries.items():
        file_path = entry if isinstance(entry, str) else entry.get("file")
        if file_path is None:
            raise ValueError(f"Dataset '{name}' must have a 'file' key.")

        sd = load_structural_csv((job_dir / file_path).resolve())

        if not isinstance(sd, FaultData):
            log.warning(
                f"  '{name}' has no rake column — skipping (Kostrov requires "
                f"strike/dip/rake)."
            )
            continue

        log.info(f"  '{name}': {len(sd)} faults")
        all_strikes.append(sd.strikes)
        all_dips.append(sd.dips)
        all_rakes.append(sd.rakes)

    if not all_strikes:
        raise ValueError(
            "No fault datasets found. CSV files must have strike, dip, rake columns.")

    strikes = np.concatenate(all_strikes)
    dips = np.concatenate(all_dips)
    rakes = np.concatenate(all_rakes)
    log.info(f"  Total: {len(strikes)} faults for Kostrov summation")

    # Kostrov tensor
    K = kostrov_tensor(strikes, dips, rakes)
    k_vals, k_vecs = np.linalg.eigh(K)
    k_order = np.argsort(k_vals)
    k_vals, k_vecs = k_vals[k_order], k_vecs[:, k_order]

    log.info(f"  Kostrov eigenvalues: {k_vals}")

    # model tensor
    _COMPARE_LABELS = {
        "stress_dev":     (r"\sigma'_1", r"\sigma'_2", r"\sigma'_3"),
        "strain":         (r"\epsilon_1", r"\epsilon_2", r"\epsilon_3"),
        "strain_rate":    (r"\dot\epsilon_1", r"\dot\epsilon_2", r"\dot\epsilon_3"),
        "strain_plastic": (r"\epsilon^p_1", r"\epsilon^p_2", r"\epsilon^p_3"),
        "strain_elastic": (r"\epsilon^e_1", r"\epsilon^e_2", r"\epsilon^e_3"),
    }
    if compare not in _COMPARE_LABELS:
        raise ValueError(
            f"compare='{compare}' not supported. "
            f"Use: {', '.join(_COMPARE_LABELS)}.")

    M = sub.avg_tensor(compare)
    axis_labels = _COMPARE_LABELS[compare]

    m_vals, m_vecs = np.linalg.eigh(M)
    m_order = np.argsort(m_vals)
    m_vals, m_vecs = m_vals[m_order], m_vecs[:, m_order]

    log.info(f"  Model eigenvalues ({compare}): {m_vals}")

    # match model axes to Kostrov axes by direction (best dot product),
    # not by eigenvalue rank — eigenvalue ordering can differ between
    # stress, strain, and elastic strain
    dots = np.abs(k_vecs.T @ m_vecs)  # (3, 3) matrix of |dot products|
    k_labels = ["K1 (short.)", "K2 (int.)", "K3 (ext.)"]
    used = set()
    for i in range(3):
        # find best matching model axis not yet used
        order = np.argsort(-dots[i])
        for j in order:
            if j not in used:
                used.add(j)
                angle = np.rad2deg(np.arccos(np.clip(dots[i, j], 0, 1)))
                log.info(f"  {k_labels[i]} <-> ${axis_labels[j]}$: "
                         f"{angle:.1f} deg")
                break

    # figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid(True)

    # cell-level model directions (spread)
    if show_cell:
        p1, a1 = line_enu2sphe(sub.dir_s1)
        p2, a2 = line_enu2sphe(sub.dir_s2)
        p3, a3 = line_enu2sphe(sub.dir_s3)
        fn = stereo_contour if cell_style == "contour" else stereo_line

        fn(ax, p1, a1, **cell_pc.as_kwargs(cell_style, "o"))
        fn(ax, p2, a2, **cell_pc.as_kwargs(cell_style, "s"))
        fn(ax, p3, a3, **cell_pc.as_kwargs(cell_style, "v"))

    # per-fault P/B/T axes (data spread)
    if show_data:
        slips = tr.slip_rake2enu(strikes, dips, rakes)
        normals = tr.plane_sphe2enu(strikes, dips)
        if slips.ndim == 1:
            slips, normals = slips[None, :], normals[None, :]

        # each dyad: ½(s⊗n + n⊗s), shape (N, 3, 3)
        dyads = 0.5 * (slips[:, :, None] * normals[:, None, :]
                        + normals[:, :, None] * slips[:, None, :])
        per_vals, per_vecs = np.linalg.eigh(dyads)

        # P (shortening), B (intermediate), T (extension)
        p_p, p_a = line_enu2sphe(per_vecs[:, :, 0])
        b_p, b_a = line_enu2sphe(per_vecs[:, :, 1])
        t_p, t_a = line_enu2sphe(per_vecs[:, :, 2])
        fn = stereo_contour if data_style == "contour" else stereo_line

        fn(ax, p_p, p_a, **data_pc.as_kwargs(data_style, "o"))
        fn(ax, b_p, b_a, **data_pc.as_kwargs(data_style, "s"))
        fn(ax, t_p, t_a, **data_pc.as_kwargs(data_style, "v"))

    # Kostrov axes
    for i, marker in enumerate(["o", "s", "v"]):
        p, a = line_enu2sphe(k_vecs[:, i])
        label = [r"$K_1$ (shortening)", r"$K_2$ (intermediate)",
                 r"$K_3$ (extension)"][i] if i < 3 else None
        stereo_line(ax, p, a, label=label, **k_style.scatter_kwargs(marker))

    # model axes
    for i, marker in enumerate(["o", "s", "v"]):
        p, a = line_enu2sphe(m_vecs[:, i])
        stereo_line(ax, p, a, label=f"${axis_labels[i]}$",
                    **m_style.scatter_kwargs(marker))

    # legend
    legend_elements = [
        Line2D([0], [0], color=k_style.color, linewidth=0, marker="o",
               markersize=8, label="Kostrov axes"),
        Line2D([0], [0], color=m_style.color, linewidth=0, marker="o",
               markersize=8, label="Model axes"),
        Line2D([], [], color="k", linewidth=0, marker="o", markersize=6,
               label=r"$1$ (min)"),
        Line2D([], [], color="k", linewidth=0, marker="s", markersize=6,
               label=r"$2$ (int)"),
        Line2D([], [], color="k", linewidth=0, marker="v", markersize=6,
               label=r"$3$ (max)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(title, y=1.08)

    # save
    out_path = out_dir / "kostrov.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")