"""Structural geology and catalog data containers."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FractureData:
    """
    Orientation measurements for planar features (joints, veins, dykes).

    Parameters
    ----------
    planes : numpy.ndarray, shape (N, 2)
        Strike and dip pairs in degrees, right-hand rule.
    """

    planes: np.ndarray

    def __post_init__(self):
        self.planes = np.atleast_2d(np.asarray(self.planes, dtype=float))
        if self.planes.shape[1] != 2:
            raise ValueError("planes must have shape (N, 2): [strike, dip].")

    def __len__(self):
        return self.planes.shape[0]

    def __repr__(self):
        return f"FractureData({len(self)} measurements)"


@dataclass
class FaultData:
    """
    Fault slip data, represented as plane orientation plus rake.

    Rake follows the Aki & Richards convention, measured in the fault
    plane from the strike direction, positive toward hanging-wall
    up-dip. Range is (-180, 180].

    - rake > 0: reverse/thrust component (hanging wall up).
    - rake < 0: normal component (hanging wall down).
    - rake = 0: pure left-lateral.
    - rake = ±180: pure right-lateral.

    Parameters
    ----------
    planes : numpy.ndarray, shape (N, 2)
        Strike and dip pairs in degrees, right-hand rule.
    rakes : numpy.ndarray, shape (N,)
        Signed rake in degrees.
    """

    planes: np.ndarray
    rakes: np.ndarray

    def __post_init__(self):
        self.planes = np.atleast_2d(np.asarray(self.planes, dtype=float))
        self.rakes = np.asarray(self.rakes, dtype=float).ravel()
        if self.planes.shape[1] != 2:
            raise ValueError("planes must have shape (N, 2): [strike, dip].")
        if self.planes.shape[0] != self.rakes.shape[0]:
            raise ValueError("planes and rakes must have the same number of rows.")
        if np.any(np.abs(self.rakes) > 180.0):
            raise ValueError(
                "Rakes must be in (-180, 180] (Aki & Richards convention)."
            )

    def __len__(self):
        return self.planes.shape[0]

    def __repr__(self):
        return f"FaultData({len(self)} measurements)"


@dataclass
class CatalogData:
    """
    Point catalog with arbitrary per-point numeric attributes.

    Used for earthquake catalogs, sample locations, observation points,
    or any tabular dataset where each row has a position and a set of
    measured values.

    Parameters
    ----------
    x, y, z : numpy.ndarray, shape (N,)
        Point coordinates.
    attrs : dict[str, numpy.ndarray], optional
        Per-point numeric attributes. Each value must be a 1D array
        of length N.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.z = np.asarray(self.z, dtype=float).ravel()
        n = self.x.shape[0]
        if self.y.shape[0] != n or self.z.shape[0] != n:
            raise ValueError("x, y, z must have the same length.")
        clean = {}
        for name, arr in self.attrs.items():
            arr = np.asarray(arr).ravel()
            if arr.shape[0] != n:
                raise ValueError(
                    f"attr '{name}' has length {arr.shape[0]}, expected {n}."
                )
            clean[name] = arr
        self.attrs = clean

    def __len__(self):
        return self.x.shape[0]

    def __repr__(self):
        keys = list(self.attrs)
        return f"CatalogData({len(self)} points, attrs={keys})"