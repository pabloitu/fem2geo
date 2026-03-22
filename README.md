# fem2geo

[![tests](https://github.com/pabloitu/fem2geo/actions/workflows/build-test.yml/badge.svg)](https://github.com/pabloitu/fem2geo/actions/workflows/build-test.yml)
[![Docs](https://readthedocs.org/projects/fem2geo/badge/?version=latest)](https://fem2geo.readthedocs.io)
[![codecov](https://codecov.io/gh/pabloitu/fem2geo/graph/badge.svg?token=Q55UKQGTY0)](https://codecov.io/gh/pabloitu/fem2geo)
[![PyPI - Version](https://img.shields.io/pypi/v/fem2geo)](https://pypi.org/project/fem2geo)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fem2geo)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19096967.svg)](https://doi.org/10.5281/zenodo.19096967)

**Physics-Based Modeling to Structural Geology and Seismology**

A Python library for structural geology and seismology analyses on outputs from finite
element (FEM) or boundary element (BEM) models. `fem2geo` bridges numerical simulations
and field observations: it loads model outputs, extracts regions of interest, computes
geomechanical variables, and produces publication-quality stereonet figures.

Key capabilities:

- Load and normalise FEM/BEM outputs (VTK/VTU)
- Extract spatial subsets from large models
- Reconstruct stress tensors; compute volume-weighted averages, principal directions, and stress
  invariants
- Compare fracture and fault orientation data with model stress predictions
- Slip tendency, dilation tendency, and combined reactivation tendency on stereonets
- Resolved shear stress analysis: compare observed and predicted fault slip
- Kostrov summed moment tensor from fault populations, compared with model kinematics
- Run any analysis from a single YAML config file

---

## Table of Contents

- [Installation](#installation)
- [Run](#run)
- [Tutorials](#tutorials)
- [Documentation](https://fem2geo.readthedocs.io)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### From source

```shell
sudo apt install git python3-pip python3-venv
git clone https://github.com/pabloitu/fem2geo
cd fem2geo
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### From PyPI

```shell
python3 -m venv venv
source venv/bin/activate
pip install fem2geo
```

Tutorials are not bundled in the PyPI package. Download them separately after installing:

```shell
fem2geo download-tutorials
```

This fetches the tutorials bundle from the latest GitHub release and extracts it
into `./tutorials`.

---

## Run

Jobs are run by pointing `fem2geo` at a YAML config file:

```shell
fem2geo config.yaml
fem2geo config_advanced.yaml --verbose
```

The `job` key in the config selects the analysis. All other keys are
job-specific — see each tutorial for a documented example.

Supported jobs:

| Job                    | Description                                                                    |
|------------------------|--------------------------------------------------------------------------------|
| `principal_directions` | Average and per-cell principal stress directions for one or more models        |
| `tendency`             | Slip, dilation, or combined tendency fields with principal directions overlaid |
| `fracture_analysis`    | Fracture pole data overlaid on model principal stress directions               |
| `resolved_shear`       | Observed vs predicted slip directions (Wallace-Bott) for each fault plane      |
| `kostrov`              | Kostrov summed moment tensor compared with model average strain or stress axes |

---

## Tutorials

Each tutorial folder contains a minimal `config.yaml` and a fully annotated
`config_advanced.yaml`. Run either with `fem2geo config.yaml` from inside the folder.

| # | Folder                   | Job                    | Description                                                                      |
|---|--------------------------|------------------------|----------------------------------------------------------------------------------|
| 1 | `1_principal_directions` | `principal_directions` | Probe the average stress state at a point in a single model                      |
| 2 | `2_model_comparison`     | `principal_directions` | Compare principal stress directions across multiple models                       |
| 3 | `3_fracture_analysis`    | `fracture_analysis`    | Overlay fracture orientation measurements on model stress predictions            |
| 4 | `4_tendency`             | `tendency`             | Slip and dilation tendency fields with optional fracture data                    |
| 5 | `5_resolved_shear`       | `resolved_shear`       | Test Wallace-Bott consistency between model stress and observed fault kinematics |
| 6 | `6_kostrov`              | `kostrov`              | Compare Kostrov bulk kinematics with model strain or deviatoric stress axes      |

---

## Roadmap

- Borehole module: sample model variables along a borehole trajectory
- Seismic catalog module: import focal mechanisms and compare with model stress orientations
- Raster module: compare model plan-view outputs with surface observations
- Inversion module: find the model configuration that best fits a structural dataset
- Schema: add a `lithostatic` flag to distinguish outputs with and without lithostatic pressure

---

## Contributing

Bug reports, feature requests, and pull requests are welcome at
[github.com/pabloitu/fem2geo](https://github.com/pabloitu/fem2geo).

---

## License

See `LICENSE`.