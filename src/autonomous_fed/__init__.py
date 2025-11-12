"""
This module initializes the autonomous_fed package.

Imports:
    - EnvironmentSolver from clients
    - EnvironmentHelpers from helpers
    - SVARResults from objects
"""

from autonomous_fed.__about__ import __title__, __version__, __author__, __description__, __url__

from .clients import EnvironmentSolver
from .helpers import EnvironmentHelpers
from .objects import SVARResults, MapMinMax
from .optimizers import LevenbergMarquardt
from .initializers import nguyen_widrow_
from .networks import SingleHiddenLayerNet

__all__ = [
    "EnvironmentSolver",
    "EnvironmentHelpers",
    "SVARResults",
    "MapMinMax",
    "LevenbergMarquardt",
    "nguyen_widrow_",
    "SingleHiddenLayerNet",
    "__title__",
    "__version__",
    "__author__",
    "__description__",
    "__url__",
]
