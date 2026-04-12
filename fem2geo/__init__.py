import os

try:
    from importlib.metadata import version
    __version__ = version("fem2geo")
except Exception:
    __version__ = "unknown"

from fem2geo.model import Model
from fem2geo.internal.schema import ModelSchema
from fem2geo.data import FractureData, FaultData, CatalogData

from fem2geo import plots
from fem2geo.utils import tensor
from fem2geo.utils import transform


dir_testdata = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tutorials", "data")
)

__all__ = [
    "__version__",
    "Model",
    "ModelSchema",
    "FractureData",
    "FaultData",
    "CatalogData",
    "plots",
    "tensor",
    "transform",
    "dir_testdata",
]