"""
This module initializes the autonomous_fed package.

Imports:
    - EnvironmentSolver from clients
    - EnvironmentHelpers from helpers
    - SVARResults from objects
"""

from autonomous_fed.__about__ import __title__, __version__, __author__, __description__, __url__

from . import clients
from . import helpers
from . import objects

from .clients import EnvironmentSolver
from .helpers import EnvironmentHelpers
from .objects import SVARResults

__all__ = [
    "EnvironmentSolver",
    "EnvironmentHelpers",
    "SVARResults",
    "__title__",
    "__version__",
    "__author__",
    "__description__",
    "__url__",
]
