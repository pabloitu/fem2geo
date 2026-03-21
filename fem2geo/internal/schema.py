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
    """Mapping for a scalar or vector field (one array in the file)."""
    canonical:  str
    solver_key: str
    category:   Optional[str]   = None
    unit:       Optional[str]   = None
    si_factor:  Optional[float] = None


@dataclass
class TensorEntry:
    """
    Mapping for a symmetric 3x3 tensor.

    Exactly one of ``voigt6`` or ``components`` must be set:

    - ``voigt6``: solver array name for a single (N, 6) packed array,
      Voigt order [xx, yy, zz, xy, yz, zx].
    - ``components``: dict mapping component labels (xx, yy, zz, xy, yz, zx)
      to their solver array names.
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
    Maps canonical fem2geo names to solver-specific array names and units.

    ``fields`` holds scalar and vector mappings (one array each).
    ``tensors`` holds symmetric tensor mappings (packed or component-wise).
    """
    name:    str
    fields:  dict[str, ScalarEntry]
    tensors: dict[str, TensorEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, units: Optional[dict] = None) -> "ModelSchema":
        active_units = {**d.get("units", {}), **(units or {})}

        fields = {}
        for canonical, spec in d.get("fields", {}).items():
            solver_key = spec["field"] if isinstance(spec, dict) else spec
            category = spec.get("category") if isinstance(spec, dict) else None
            unit, si = resolve_unit(category, active_units)
            fields[canonical] = ScalarEntry(
                canonical, solver_key, category, unit, si
            )

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

        return cls(name=d.get("solver", "custom"), fields=fields,
                   tensors=tensors)

    @classmethod
    def from_yaml(cls, path, units: Optional[dict] = None) -> "ModelSchema":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, units=units)

    @classmethod
    def builtin(cls, name: str, units: Optional[dict] = None) -> "ModelSchema":
        """Load a built-in schema by name (e.g. ``"adeli"``)."""
        p = Path(__file__).parent / "schemas" / f"{name}.yaml"
        if not p.exists():
            available = [
                f.stem for f in
                (Path(__file__).parent / "schemas").glob("*.yaml")
            ]
            raise ValueError(
                f"No built-in schema '{name}'. Available: {available}"
            )
        return cls.from_yaml(p, units=units)