import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


SI_FACTORS: dict[str, float] = {
    "pa": 1.0,      "kpa": 1e3,     "mpa": 1e6,     "gpa": 1e9,
    "mm": 1e-3,     "cm":  1e-2,    "m":   1.0,     "km":  1e3,
    "m/s": 1.0,     "cm/a": 1e-2 / 3.156e7,
    "k":  1.0,      "c":   1.0,
    "s":  1.0,      "ma":  3.156e13,
}


@dataclass
class FieldEntry:
    canonical:  str
    solver_key: str
    category:   Optional[str]   = None
    unit:       Optional[str]   = None
    si_factor:  Optional[float] = None


@dataclass
class ModelSchema:
    """
    Maps canonical fem2geo field names to solver-specific array names and units.

    Parameters
    ----------
    name : str
        Solver identifier.
    fields : dict[str, FieldEntry]
        Canonical field name to entry mapping.
    """
    name:   str
    fields: dict[str, FieldEntry]

    @classmethod
    def _from_raw(cls, raw: dict, unit_overrides: Optional[dict] = None) -> "ModelSchema":
        active_units = {**raw.get("units", {}), **(unit_overrides or {})}
        fields = {}
        for canonical, spec in raw.get("fields", {}).items():
            solver_key = spec["field"] if isinstance(spec, dict) else spec
            category   = spec.get("category") if isinstance(spec, dict) else None
            unit       = active_units.get(category) if category else None
            si_factor  = SI_FACTORS.get(unit.lower()) if unit else None
            fields[canonical] = FieldEntry(canonical, solver_key, category, unit, si_factor)
        return cls(name=raw.get("solver", "custom"), fields=fields)

    @classmethod
    def from_dict(cls, d: dict, units: Optional[dict] = None) -> "ModelSchema":
        return cls._from_raw(d, unit_overrides=units)

    @classmethod
    def from_yaml(cls, path, units: Optional[dict] = None) -> "ModelSchema":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_raw(raw, unit_overrides=units)

    @classmethod
    def builtin(cls, name: str, units: Optional[dict] = None) -> "ModelSchema":
        """
        Load a schema shipped with the package.

        Parameters
        ----------
        name : str
            Schema name, e.g. ``"adeli"``.
        units : dict, optional
            Category-level unit overrides, e.g. ``{"pressure": "Pa"}``.

        Raises
        ------
        ValueError
            If no built-in schema with the given name exists.
        """
        p = Path(__file__).parent / "schemas" / f"{name}.yaml"
        print(p)
        if not p.exists():
            available = [f.stem for f in (Path(__file__).parent / "schemas").glob("*.yaml")]
            raise ValueError(f"No built-in schema '{name}'. Available: {available}")
        return cls.from_yaml(p, units=units)

    def solver_key(self, canonical: str) -> str:
        return self.fields[canonical].solver_key

    def si_factor(self, canonical: str) -> float:
        entry = self.fields.get(canonical)
        return entry.si_factor if (entry and entry.si_factor) else 1.0

    def has(self, canonical: str) -> bool:
        return canonical in self.fields