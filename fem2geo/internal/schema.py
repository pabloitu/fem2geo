import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


SI_FACTORS: dict[str, float] = {
    "pa": 1.0,      "kpa": 1e3,     "mpa": 1e6,     "gpa": 1e9,
    "mm": 1e-3,     "cm":  1e-2,    "m":   1.0,     "km":  1e3,
    "m/s": 1.0,     "cm/a": 1e-2 / 3.156e7,
    "k":  1.0,      "c":   1.0,
    "s":  1.0,      "ma":  3.156e13,
    "pa.s": 1.0,    "1/s": 1.0,     "mw/m2": 1e-3,
}


def resolve_unit(category, active_units):
    """Return (unit_str, si_factor) for a category, or (None, None)."""
    if not category:
        return None, None
    unit = active_units.get(category)
    if not unit:
        return None, None
    return unit, SI_FACTORS.get(unit.lower())


@dataclass
class ScalarEntry:
    """
    Single scalar or vector field mapping.

    Parameters
    ----------
    canonical : str
        Internal fem2geo name for this field.
    solver_key : str
        Array name used by the solver output file.
    category : str, optional
        Unit category (e.g. ``"pressure"``, ``"velocity"``).
    unit : str, optional
        Unit string resolved from the active unit set.
    si_factor : float, optional
        Multiplicative factor to convert to SI units.
    """
    canonical:  str
    solver_key: str
    category:   Optional[str]   = None
    unit:       Optional[str]   = None
    si_factor:  Optional[float] = None


@dataclass
class TensorEntry:
    """
    Symmetric 3x3 tensor mapping.

    Either ``voigt6`` (single packed array, Voigt order
    [xx, yy, zz, xy, yz, zx]) or ``components`` (dict of component
    labels to solver array names) must be set, not both.

    Parameters
    ----------
    canonical : str
        Internal fem2geo name for this tensor.
    voigt6 : str, optional
        Solver array name for the packed (N, 6) representation.
    components : dict[str, str], optional
        Map from component label to solver array name.
    category : str, optional
        Unit category (e.g. ``"pressure"``).
    unit : str, optional
        Unit string resolved from the active unit set.
    si_factor : float, optional
        Multiplicative factor to convert to SI units.
    """
    canonical:  str
    voigt6:     Optional[str]            = None
    components: Optional[dict[str, str]] = None
    category:   Optional[str]            = None
    unit:       Optional[str]            = None
    si_factor:  Optional[float]          = None

    @property
    def is_packed(self) -> bool:
        return self.voigt6 is not None


@dataclass
class ModelSchema:
    """
    Translates between solver-specific array names and canonical
    fem2geo names, including unit conversion factors.

    Parameters
    ----------
    name : str
        Schema identifier (e.g. solver name).
    fields : dict[str, ScalarEntry]
        Scalar and vector field mappings.
    tensors : dict[str, TensorEntry]
        Symmetric tensor mappings, keyed by canonical name.
    """
    name:    str
    fields:  dict[str, ScalarEntry]
    tensors: dict[str, TensorEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, units: Optional[dict] = None) -> "ModelSchema":
        """
        Build a schema from a raw config dictionary.

        Parameters
        ----------
        d : dict
            Schema definition, normally loaded from YAML.
        units : dict, optional
            Category-level unit overrides (e.g. ``{"pressure": "MPa"}``).

        Returns
        -------
        ModelSchema
        """
        active_units = {**d.get("units", {}), **(units or {})}

        fields = {}
        for canonical, spec in d.get("fields", {}).items():
            solver_key = spec["field"] if isinstance(spec, dict) else spec
            category = spec.get("category") if isinstance(spec, dict) else None
            unit, si = resolve_unit(category, active_units)
            fields[canonical] = ScalarEntry(canonical, solver_key, category, unit, si)

        tensors = {}
        for canonical, spec in d.get("tensors", {}).items():
            category = spec.get("category")
            unit, si = resolve_unit(category, active_units)
            tensors[canonical] = TensorEntry(
                canonical=canonical,
                voigt6=spec.get("voigt6"),
                components=spec.get("components"),
                category=category, unit=unit, si_factor=si,
            )

        return cls(name=d.get("solver", "custom"), fields=fields, tensors=tensors)

    @classmethod
    def from_yaml(cls, path, units: Optional[dict] = None) -> "ModelSchema":
        """
        Load a schema from a YAML file on disk.

        Parameters
        ----------
        path : str or Path
            Path to the YAML schema file.
        units : dict, optional
            Category-level unit overrides.

        Returns
        -------
        ModelSchema
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, units=units)

    @classmethod
    def builtin(cls, name: str, units: Optional[dict] = None) -> "ModelSchema":
        """
        Load one of the bundled schemas by name.

        Parameters
        ----------
        name : str
            Schema name (e.g. ``"adeli"``). Must match a YAML file
            under ``fem2geo/internal/schemas/``.
        units : dict, optional
            Category-level unit overrides.

        Returns
        -------
        ModelSchema

        Raises
        ------
        ValueError
            If no schema with that name exists.
        """
        p = Path(__file__).parent / "schemas" / f"{name}.yaml"
        if not p.exists():
            available = [
                f.stem for f in
                (Path(__file__).parent / "schemas").glob("*.yaml")
            ]
            raise ValueError(f"No built-in schema '{name}'. Available: {available}")
        return cls.from_yaml(p, units=units)