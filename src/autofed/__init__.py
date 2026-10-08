# MIT License
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""autofed: reinforcement learning for monetary policy.

Replication and extension of Hinterlang & Tänzer (2021), "Optimal Monetary Policy Using
Reinforcement Learning": estimated linear and non-linear economies, benchmark interest-rate rules,
and learned reaction functions.

The top level exposes what a research notebook needs first::

    import autofed as af

    nb = af.NotebookSession("01_linear_svar", number=1)
    panel = af.load_macro_panel(nb.artifacts("data"))
"""

from __future__ import annotations

from . import linear, plots, rules, style
from .artifacts import load_frame, save_frame
from .data import NAMES, MacroPanel, SampleConfig, fetch_macro_panel, load_macro_panel
from .exceptions import (
    ArtifactError,
    AutofedError,
    ConfigurationError,
    DataError,
    ExportError,
    SpecificationError,
)
from .export import notebook_to_pdf
from .project import Workspace
from .report import Exhibit, NotebookSession
from .rules import REFERENCE_RULES, PolicyRule

__version__ = "0.0.2"

__all__ = [
    "NAMES", "REFERENCE_RULES", "ArtifactError", "AutofedError", "ConfigurationError", "DataError", "Exhibit",
    "ExportError", "MacroPanel", "NotebookSession", "PolicyRule", "SampleConfig", "SpecificationError", "Workspace",
    "__version__", "fetch_macro_panel", "linear", "load_frame", "load_macro_panel", "notebook_to_pdf", "plots", "rules", "save_frame", "style",
]
