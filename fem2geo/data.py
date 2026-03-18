"""
Structural geology data containers.

These dataclasses hold field measurements loaded from CSV files via
:func:`fem2geo.internal.io.load_structural_csv`. They are the user-facing
data objects for structural analysis jobs.

Dykes, joints, veins, and fractures all use :class:`FractureData` — any
planar feature described by strike and dip without slip information.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FractureData:
    """
    Fracture (or joint, vein, dyke) orientation measurements.

    Parameters
    ----------
    planes : numpy.ndarray, shape (N, 2)
        Strike/dip pairs in degrees (right-hand rule).
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

    @property
    def strikes(self) -> np.ndarray:
        return self.planes[:, 0]

    @property
    def dips(self) -> np.ndarray:
        return self.planes[:, 1]


@dataclass
class FaultData:
    """
    Fault slip measurements: plane orientation with rake.

    Parameters
    ----------
    planes : numpy.ndarray, shape (N, 2)
        Strike/dip pairs in degrees (right-hand rule).
    rakes : numpy.ndarray, shape (N,)
        Rake angles in degrees, measured within the fault plane from the
        strike direction.

    Notes
    -----
    Strain tensor derivation from fault populations (e.g. Kostrov summation)
    is not yet implemented.
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

    def __len__(self):
        return self.planes.shape[0]

    def __repr__(self):
        return f"FaultData({len(self)} measurements)"

    @property
    def strikes(self) -> np.ndarray:
        return self.planes[:, 0]

    @property
    def dips(self) -> np.ndarray:
        return self.planes[:, 1]