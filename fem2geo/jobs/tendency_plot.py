"""
Job: tendency_plot
==================
Plots slip, dilation, combined, or paired tendency fields on a stereonet
for one model at one extraction zone. Average stress principal directions
are overlaid by default. Individual cell directions can optionally be
shown to visualise the spread around the average.

Tendency types:

- ``slip``: normalized slip tendency Ts' (Morris et al., 1996), range [0, 1].
- ``dilation``: dilation tendency Td (Ferrill et al., 1999), range [0, 1].
- ``combined``: Ts' + Td (Ferrill et al., 2020), range [0, 2].
- ``both``: side-by-side slip and dilation panels.

Config reference
----------------
job: tendency_plot
schema: adeli               # built-in schema name (default: adeli)
units:                      # optional category-level unit overrides
  pressure: Pa

model: path/to/model.vtk    # relative to this config file

zone:
  type: sphere              # sphere | box
  center: [x, y, z]
  radius: r                 # sphere only
  # dim: [dx, dy, dz]       # box only

plot:
  title: ""                 # optional, auto-generated if empty
  figsize: [16, 7]          # default for both; [8, 8] for single
  dpi: 200
  tendency: both            # slip | dilation | combined | both
  n_strikes: 180            # stereonet grid resolution
  n_dips: 45
  avg_directions:           # average stress directions (default: show=true)
    show: true
    color: "white"
    markersize: 8
  cell_directions:          # per-cell directions to show spread (default: show=false)
    show: false
    style: scatter          # scatter | contour
    color: "k"
    markersize: 3
    alpha: 0.4

output:
  dir: results/             # optional, defaults to config file directory
  save_vtu: false           # save extracted sub-model for Paraview

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem2geo import runner
from fem2geo.model import Model
from fem2geo.plots import PlotConfig, stereo_field, stereo_line, stereo_contour
from fem2geo.runner import parse_config
from fem2geo.utils.tensor import slip_tendency, dilation_tendency, combined_tendency
from fem2geo.utils.transform import line_enu2sphe, grid_nodes, grid_centers

log = logging.getLogger("fem2geoLogger")

_VALID_TENDENCIES = ("slip", "dilation", "combined", "both")

_TENDENCY_FNS = {
    "slip": slip_tendency,
    "dilation": dilation_tendency,
    "combined": combined_tendency,
}

_CBAR_LABELS = {
    "slip": r"Slip tendency $T'_s$",
    "dilation": r"Dilation tendency $T_d$",
    "combined": r"Combined tendency $T'_s + T_d$",
}

_TITLES = {
    "slip": "Slip tendency",
    "dilation": "Dilation tendency",
    "combined": "Combined tendency",
}

_VMAX = {"slip": 1.0, "dilation": 1.0, "combined": 2.0}


def _compute_field(sigma, kind, n_strikes, n_dips):
    """
    Discretize the stereonet and compute a tendency field.

    Returns the node grids (for pcolormesh edges) and the cell-center
    values ready for :func:`~fem2geo.plots.stereo_field`.
    """
    mesh_s, mesh_d = grid_nodes(n_strikes, n_dips)
    cs, cd = grid_centers(mesh_s, mesh_d)
    planes = np.column_stack([cs.ravel(), cd.ravel()])
    vals = _TENDENCY_FNS[kind](sigma, planes=planes).reshape(cs.shape)
    return mesh_s, mesh_d, vals


def run(cfg: dict, job_dir: Path) -> None:
    # config
    schema, zone, data, plot, out = parse_config(cfg, job_dir)

    tendency = plot.get("tendency", "both")
    dpi = plot.get("dpi", 200)
    n_strikes = plot.get("n_strikes", 180)
    n_dips = plot.get("n_dips", 45)
    out_dir = Path(out.get("dir", job_dir))
    save_vtu = out.get("save_vtu", False)

    avg_cfg = plot.get("avg_directions", {})
    cell_cfg = plot.get("cell_directions", {})
    show_avg = avg_cfg.get("show", True)
    show_cell = cell_cfg.get("show", False)
    cell_style = cell_cfg.get("style", "scatter")

    avg_style = PlotConfig.avg(color="white").update(
        avg_cfg if isinstance(avg_cfg, dict) else {})
    cell_pc = (PlotConfig.density() if cell_style == "contour"
               else PlotConfig.cell()).update(
        cell_cfg if isinstance(cell_cfg, dict) else {})

    if tendency not in _VALID_TENDENCIES:
        raise ValueError(
            f"plot.tendency must be one of {_VALID_TENDENCIES}, "
            f"got '{tendency}'.")
    if "model" not in cfg:
        raise ValueError("tendency_plot requires a 'model' key.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # load and extract
    path = (job_dir / cfg["model"]).resolve()
    log.info(f"Loading model: {path}")
    model = Model.from_file(path, schema)

    sub = model.extract(zone)

    log.info(f"  {sub.n_cells} cells in zone")

    if save_vtu:
        sub.save(out_dir / "extract.vtu")

    # average stress and principal directions
    avg_stress = sub.avg_tensor("stress")
    val, vec = np.linalg.eigh(avg_stress)
    order = np.argsort(val)
    val, vec = val[order], vec[:, order]
    phi = float((val[1] - val[2]) / (val[0] - val[2]))

    # figure layout
    is_double = tendency == "both"
    if is_double:
        figsize = plot.get("figsize", [16, 7])
    else:
        figsize = plot.get("figsize", [8, 8])

    fig = plt.figure(figsize=figsize)

    if is_double:
        ax_slip = fig.add_subplot(121, projection="stereonet")
        ax_dil = fig.add_subplot(122, projection="stereonet")
        panels = {"slip": ax_slip, "dilation": ax_dil}
    else:
        panels = {tendency: fig.add_subplot(111, projection="stereonet")}

    # tendency fields
    for kind, ax in panels.items():
        ax.grid(True)
        mesh_s, mesh_d, vals = _compute_field(
            avg_stress, kind, n_strikes, n_dips)
        stereo_field(ax, mesh_s, mesh_d, vals,
                     vmin=0.0, vmax=_VMAX[kind],
                     cbar_label=_CBAR_LABELS[kind])

    # cell directions (spread)
    if show_cell:
        p1, a1 = line_enu2sphe(sub.dir_s1)
        p2, a2 = line_enu2sphe(sub.dir_s2)
        p3, a3 = line_enu2sphe(sub.dir_s3)
        fn = stereo_contour if cell_style == "contour" else stereo_line

        for ax in panels.values():
            fn(ax, p1, a1, **cell_pc.as_kwargs(cell_style, "o"))
            fn(ax, p2, a2, **cell_pc.as_kwargs(cell_style, "s"))
            fn(ax, p3, a3, **cell_pc.as_kwargs(cell_style, "v"))

    # average directions
    if show_avg:
        p1, a1 = line_enu2sphe(vec[:, 0])
        p2, a2 = line_enu2sphe(vec[:, 1])
        p3, a3 = line_enu2sphe(vec[:, 2])

        for ax in panels.values():
            stereo_line(ax, p1, a1, label=r"$\sigma_1$",
                        **avg_style.scatter_kwargs("o"))
            stereo_line(ax, p2, a2, label=r"$\sigma_2$",
                        **avg_style.scatter_kwargs("s"))
            stereo_line(ax, p3, a3, label=r"$\sigma_3$",
                        **avg_style.scatter_kwargs("v"))

    # titles and legends
    suffix = (f"\n$\\sigma_1={val[0]:.3f}$, $\\sigma_3={val[2]:.3f}$,"
              f" $\\phi={phi:.2f}$")
    custom_title = plot.get("title", "")

    for kind, ax in panels.items():
        t = custom_title if custom_title else _TITLES[kind] + suffix
        ax.set_title(t, y=1.05)
        ax.legend(fontsize=7)

    # save
    out_path = out_dir / "tendency_plot.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")