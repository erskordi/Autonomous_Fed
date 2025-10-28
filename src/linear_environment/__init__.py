"""
This module initializes the linear_environment package.

Imports:
    - LinearEnvironmentSolver from clients
    - LinearEnvironmentHelpers from helpers
    - SVARResults from objects    
"""

from linear_environment.__about__ import __title__, __version__, __author__, __description__, __url__

from . import clients
from . import helpers
from . import objects

from .clients import LinearEnvironmentSolver
from .helpers import LinearEnvironmentHelpers
from .objects import SVARResults

__all__ = [
    "LinearEnvironmentSolver",
    "LinearEnvironmentHelpers",
    "SVARResults",
    "__title__",
    "__version__",
    "__author__",
    "__description__",
    "__url__",
]
