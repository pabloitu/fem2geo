# fem2geo

**Physics-Based Modeling to Structural Geology and Seismology**

A Python library for structural geology and seismology analyses on outputs from finite element
(FEM) or boundary element (BEM) models

`fem2geo` bridges numerical simulations and field
observations: it loads model outputs, extracts regions of interest, computes geomechanical
variables, and produces publication-quality stereonet figures.

Key capabilities:

- Load and normalise FEM/BEM simulation outputs (VTK/VTU) via a flexible solver schema system
- Extract spatial subsets (spheres, bounding boxes) from large models
- Reconstruct stress tensors and compute volume-weighted average stress, principal directions, and stress invariants
- Slip and dilation tendency analysis on stereonets
- Compare multiple model realisations at the same location
- Run analyses from a YAML config file with sensible defaults and optional customisation

---

## Table of Contents

- [Installation](#installation)
- [Run](#run)
- [Tutorials](#tutorials)
- [Documentation](https://fem2geo.readthedocs.io)
- [Roadmap / Known Issues](#roadmap--known-issues)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Latest version

```shell
sudo apt install git python3-pip python3-venv
git clone https://github.com/pabloitu/fem2geo
cd fem2geo
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Stable version (from PyPI)

```shell
python3 -m venv venv
source venv/bin/activate
pip install fem2geo
```

Tutorials are not included in the PyPI package. Download them separately:

```shell
fem2geo download-tutorials
```

---

## Run

Each tutorial folder contains a `config.yaml` for the basic case and a
`config_advanced.yaml` with full plot customisation. Run either with:

```shell
cd tutorials/1_probing_model
fem2geo config.yaml
fem2geo config_advanced.yaml
```

The `job` key in the config selects the analysis. Currently supported jobs:

| Job | Description |
|-----|-------------|
| `principal_directions` | Stereonet of average (and optional cell-level) principal stress directions for one or multiple models |
| `tendency` | Slip and/or dilation tendency contour with average principal directions overlaid |

---

## Tutorials

| # | Folder | Job                    | Description |
|---|--------|------------------------|-------------|
| 1 | `1_probing_model` | `principal_directions` | Probe average stress state at a location in a single model |
| 2 | `2_model_comparison` | `principal_directions` | Compare principal stress directions across multiple models |
| 3 | `3_structural_data` | `compare_structural`   | Overlay field structural measurements on model predictions |
| 4 | `4_tendency` | `tendency`        | Slip and dilation tendency with average directions |

---

## Roadmap / Known Issues

- Borehole module: sample model variables along a borehole trajectory
- Seismic catalog module: import focal mechanisms and compare with model stress orientations
- Raster module: compare model plan-view outputs with surface observations
- Inversion module: find the model configuration that best fits a dataset
- Schema: add a `lithostatic` flag to mark whether outputs include lithostatic pressure
- Stress reconstruction warning fires on every `stress` access — should fire once per instance