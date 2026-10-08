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

"""Matplotlib house style: one palette and one set of rcParams for every notebook."""

from __future__ import annotations

from typing import Final

import matplotlib as mpl
from cycler import cycler

__all__ = ["COLORS", "SERIES_COLORS", "apply"]

#: Colour-blind-safe palette (Okabe & Ito, 2008).
COLORS: Final[dict[str, str]] = {
    "black": "#000000",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}

#: Fixed colours of recurring series, so a rule or a model looks the same in every figure.
SERIES_COLORS: Final[dict[str, str]] = {
    "Actual": COLORS["black"],
    "TR93": COLORS["vermillion"],
    "NPP": COLORS["orange"],
    "BA": COLORS["green"],
    "SVAR": COLORS["blue"],
    "TVP-SVAR-SV": COLORS["purple"],
    "NARX": COLORS["sky"],
    "NSSM": COLORS["grey"],
}

_CYCLE: Final[tuple[str, ...]] = ("blue", "vermillion", "green", "purple", "orange", "sky", "grey")


def apply() -> None:
    """Install the house style into ``matplotlib.rcParams``.

    Figures use constrained layout, so callers never call ``tight_layout``; numbering, titles and
    notes are added by :meth:`autofed.report.NotebookSession.figure`.
    """
    mpl.rcParams.update(
        {
            "figure.figsize": (9.0, 4.2),
            "figure.dpi": 110,
            "figure.constrained_layout.use": True,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.titleweight": "regular",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=[COLORS[c] for c in _CYCLE]),
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "mathtext.fontset": "dejavusans",
        }
    )
