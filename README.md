<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/fem2geo_logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/fem2geo_logo.png">
  <img alt="fem2geo" src="docs/source/_static/fem2geo_logo.svg" width="400">
</picture>


[![tests](https://github.com/pabloitu/fem2geo/actions/workflows/build-test.yml/badge.svg)](https://github.com/pabloitu/fem2geo/actions/workflows/build-test.yml)
[![Docs](https://readthedocs.org/projects/fem2geo/badge/?version=latest)](https://fem2geo.readthedocs.io)
[![codecov](https://codecov.io/gh/pabloitu/fem2geo/graph/badge.svg?token=Q55UKQGTY0)](https://codecov.io/gh/pabloitu/fem2geo)
[![PyPI - Version](https://img.shields.io/pypi/v/fem2geo)](https://pypi.org/project/fem2geo)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fem2geo)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19096967.svg)](https://doi.org/10.5281/zenodo.19096967)

**Physics-Based Modeling to Structural Geology**

A Python library for structural geology analyses on outputs from finite
element (FEM) or boundary element (BEM) models. `fem2geo` bridges numerical
simulations and field observations: it loads model outputs, extracts regions
of interest, computes geomechanical variables, and produces stereonet figures.

Key capabilities:

- Extract spatial subsets from large models
- Compare fracture and fault orientation data with model predictions
- Slip and dilation tendencies on stereonets
- Compare model to Kostrov moment tensor derived from fault populations
- Re-project datasets (catalogs, meshes, rasters) into a model's reference frame
- Run any analysis from reproducible config files

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
```

Supported jobs, each shown in ``tutorials/``:

| Job                    | Description                                                                       |
|------------------------|-----------------------------------------------------------------------------------|
| `principal_directions` | Average and per-cell principal directions at a site                               |
| `fracture`             | Fracture pole data overlaid on model principal directions                         |
| `resolved_shear`       | Observed vs predicted slip directions for each fault plane                        |
| `kostrov`              | Kostrov summed moment tensor compared with model average strain or stress axes    |
| `tendency`             | Slip, dilation, or summarized tendency fields                                     |
| `sites.<inner>`        | Run any of the above over multiple sites in one figure                            |
| `project`              | Project georeferenced catalogs, meshes, or rasters into a model's reference frame |

---


## Roadmap

- Borehole module: sample model variables along a borehole trajectory
- Seismic catalog module: import focal mechanisms and compare with model stress orientations
- Inversion module: find the model configuration that best fits a structural dataset

---

## Contributing

Bug reports and feature requests are welcome at
[github.com/pabloitu/fem2geo/issues](https://github.com/pabloitu/fem2geo/issues).

