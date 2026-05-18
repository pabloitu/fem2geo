# Contributing to fem2geo

Thank you for contributing to `fem2geo`! This document describes how to report issues, suggest changes, and contribute
code or documentation.

## Reporting bugs

Search [existing issues](https://github.com/pabloitu/fem2geo/issues) first.
If the problem is not already tracked, open a new issue with:

- A short description of expected vs actual behaviour
- The minimal config or script that reproduces it
- Python version, OS, and `fem2geo` version (`pip show fem2geo`)
- The full traceback, when applicable

For security-sensitive reports, email the maintainers directly instead of
opening a public issue.

## Suggesting enhancements

Open an issue describing the feature, its use case, and a sketch of the API
or configuration syntax when possible. Check that the idea fits the scope
of the project.

## Contributing code

### Setup

Fork the repository, then:

```console
$ git clone https://github.com/<your-fork>/fem2geo
$ cd fem2geo
$ python -m venv venv
$ source venv/bin/activate
$ pip install -e ".[dev]"
```

The `dev` extras install testing, linting, and documentation dependencies.

### Workflow

1. Branch off `master`: `git checkout -b {feature-name}`
2. Make changes, with tests covering the new behaviour
3. Run the test suite: `pytest tests/`
4. Format with `black .`
5. Open a pull request against `master` and link the related issue

Keep pull requests focused. A small PR with one concern is easier to review
than a sprawling one.


### Adding a new analysis

Analyses live under `fem2geo/jobs/`. Each one:

- Declares its YAML schema in `fem2geo/internal/schemas/`
- Reads inputs through `fem2geo.model.Model`
- Writes outputs in one of the supported formats
- Ships with a tutorial under `docs/source/tutorials/` and an example YAML
  under `tutorials/`

See `fracture`, `kostrov`, and `tendency` for reference implementations.


## Contributing documentation

Documentation lives in `docs/source/`:

- `intro/` — installation, theory, user guide
- `tutorials/` — config-driven tutorials matching the `tutorials/` folder
  at the repository root
- `reference/` — API reference, mostly auto-generated from docstrings

### Building locally

With the dev environment active, from `docs/`:

```console
$ make clean
$ make html
```

The result is at `docs/build/html/index.html`.


### Adding a tutorial

A tutorial entry consists of:

1. A YAML configuration under `tutorials/<N>_<name>/`
2. An documentation RST page under `docs/source/tutorials/<name>.rst`
3. A link added to the toctree in `docs/source/index.rst`
4. Any referenced figures under `docs/source/_static/tutorials/`

## Attribution

Adapted from the [contributing.md](https://contributing.md/) template.