import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


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
    """
    canonical:  str
    solver_key: str


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
    """
    canonical:  str
    voigt6:     Optional[str]            = None
    components: Optional[dict[str, str]] = None

    @property
    def is_packed(self) -> bool:
        return self.voigt6 is not None


@dataclass
class ModelSchema:
    """
    Translates between solver-specific array names and canonical fem2geo names.

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
    def from_dict(cls, d: dict) -> "ModelSchema":
        """
        Build a schema from a raw config dictionary.

        Parameters
        ----------
        d : dict
            Schema definition, normally loaded from YAML.

        Returns
        -------
        ModelSchema
        """
        fields = {}
        for canonical, spec in d.get("fields", {}).items():
            solver_key = spec["field"] if isinstance(spec, dict) else spec
            fields[canonical] = ScalarEntry(canonical, solver_key)

        tensors = {}
        for canonical, spec in d.get("tensors", {}).items():
            tensors[canonical] = TensorEntry(
                canonical=canonical,
                voigt6=spec.get("voigt6"),
                components=spec.get("components"),
            )

        return cls(name=d.get("solver", "custom"), fields=fields, tensors=tensors)

    @classmethod
    def from_yaml(cls, path) -> "ModelSchema":
        """
        Load a schema from a YAML file on disk.

        Parameters
        ----------
        path : str or Path
            Path to the YAML schema file.

        Returns
        -------
        ModelSchema
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def builtin(cls, name: str) -> "ModelSchema":
        """
        Load one of the bundled schemas by name.

        Parameters
        ----------
        name : str
            Schema name (e.g. ``"adeli"``). Must match a YAML file
            under ``fem2geo/internal/schemas/``.

        Returns
        -------
        ModelSchema

        """
        p = Path(__file__).parent / "schemas" / f"{name}.yaml"
        if not p.exists():
            available = [
                f.stem for f in
                (Path(__file__).parent / "schemas").glob("*.yaml")
            ]
            raise ValueError(f"No built-in schema '{name}'. Available: {available}")
        return cls.from_yaml(p)