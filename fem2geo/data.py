"""Structural geology data containers."""

from dataclasses import dataclass

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
