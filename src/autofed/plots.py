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

"""Recurring figure layouts. Every function returns a figure for ``NotebookSession.figure``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import COLORS, SERIES_COLORS

__all__ = ["VARIABLE_LABELS", "actual_vs_fit", "irf_grid", "policy_paths", "policy_surface_grid",
           "shade_window", "squared_errors", "surface", "timestamps", "training_curves"]

#: Axis labels of the model variables, with units.
VARIABLE_LABELS: Final[dict[str, str]] = {
    "y": "Output gap (percent)",
    "pi": "Inflation (percent, year over year)",
    "i": "Federal funds rate (percent)",
}
_EXTRA_COLORS: Final[tuple[str, ...]] = ("blue", "purple", "sky", "grey")
_LINESTYLES: Final[tuple[str, ...]] = ("--", ":", "-.", (0, (5, 1)))  # type: ignore[assignment]


def timestamps(index: pd.Index) -> pd.DatetimeIndex:
    """Quarter-start timestamps of a quarterly ``PeriodIndex`` (or the index itself if already dates)."""
    return index.to_timestamp() if isinstance(index, pd.PeriodIndex) else pd.DatetimeIndex(index)


def shade_window(ax: Axes, start: str, end: str, *, label: str | None = None) -> None:
    """Shade the quarters ``start`` to ``end`` (inclusive) in light grey."""
    lo = pd.Period(start, "Q").to_timestamp(how="start")
    hi = pd.Period(end, "Q").to_timestamp(how="end")
    ax.axvspan(lo, hi, color=COLORS["grey"], alpha=0.12, lw=0, label=label)


def actual_vs_fit(frame: pd.DataFrame, column: str, fits: Mapping[str, str], *,
                  bands: tuple[str, str] | None = None, band_label: str = "90% posterior band") -> Figure:
    """Realised series against one or more fitted columns.

    Args:
        frame: Data on a quarterly index holding ``column`` and every fitted column.
        column: ``"y"`` or ``"pi"``.
        fits: Legend label -> fitted column name.
        bands: Names of the lower and upper band columns.
        band_label: Legend label of the band.

    Raises:
        KeyError: If a requested column is missing.
    """
    needed = [column, *fits.values(), *(bands or ())]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise KeyError(f"frame lacks columns {missing}")
    x = timestamps(frame.index)
    fig, ax = plt.subplots()
    if bands is not None:
        ax.fill_between(x, frame[bands[0]], frame[bands[1]], color=COLORS["blue"], alpha=0.15, lw=0, label=band_label)
    ax.plot(x, frame[column], color=COLORS["black"], lw=1.6, label="Actual")
    palette = [COLORS["blue"], COLORS["vermillion"], COLORS["green"], COLORS["purple"]]
    for k, (label, fit_col) in enumerate(fits.items()):
        ax.plot(x, frame[fit_col], ls=_LINESTYLES[k % len(_LINESTYLES)], color=palette[k % len(palette)],
                lw=1.5 if k == 0 else 1.2, label=label)
    ax.axhline(0.0 if column == "y" else 2.0, color=COLORS["grey"], lw=0.6)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(VARIABLE_LABELS.get(column, column))
    ax.legend(loc="best")
    return fig


def squared_errors(series: Mapping[str, pd.Series], *, colors: Sequence[str] | None = None) -> Figure:
    """Squared one-step errors over time, one line per model (the layout of the paper's Figures 4-5)."""
    if not series:
        raise ValueError("at least one error series is required")
    fig, ax = plt.subplots()
    palette = list(colors) if colors is not None else [COLORS["vermillion"], COLORS["blue"], COLORS["green"]]
    for k, (label, values) in enumerate(series.items()):
        ax.plot(timestamps(values.index), values, color=palette[k % len(palette)], lw=1.4, label=label)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Squared error (percentage points squared)")
    ax.legend(loc="best")
    return fig


def irf_grid(responses: np.ndarray, names: Sequence[str], *, labels: Mapping[str, str] | None = None) -> Figure:
    """Grid of impulse responses: rows are responding variables, columns are shocks.

    Args:
        responses: Array of shape ``(horizon + 1, variable, shock)``.
        names: Variable names in model order.
        labels: Display names; defaults to the names.

    Raises:
        ValueError: If ``responses`` does not have shape ``(h, k, k)`` with ``k == len(names)``.
    """
    k = len(names)
    if responses.ndim != 3 or responses.shape[1:] != (k, k):
        raise ValueError(f"responses must have shape (h, {k}, {k}), got {responses.shape}")
    shown = dict(labels or {n: n for n in names})
    horizon = np.arange(responses.shape[0])
    fig, axes = plt.subplots(k, k, figsize=(10.5, 7.2), sharex=True)
    for r, response in enumerate(names):
        for c, shock in enumerate(names):
            ax = axes[r, c]
            ax.plot(horizon, responses[:, r, c], color=COLORS["blue"])
            ax.axhline(0.0, color=COLORS["black"], lw=0.5)
            ax.set_title(f"{shown[response]} to {shown[shock]} shock", fontsize=9)
            if c == 0:
                ax.set_ylabel("Percentage points")
            if r == k - 1:
                ax.set_xlabel("Quarters after shock")
    return fig


def policy_paths(paths: Mapping[str, pd.DataFrame], *, pi_star: float = 2.0,
                 shade: tuple[str, str] | None = None) -> Figure:
    """Three stacked panels (policy rate, inflation, output gap), one line per policy.

    Args:
        paths: Frames with columns ``i``, ``pi`` and ``y`` keyed by policy name; ``"Actual"`` is drawn
            in black on top.
        pi_star: Inflation target, drawn as the reference line of the inflation panel.
        shade: Quarters to shade, e.g. the estimation window.
    """
    if not paths:
        raise ValueError("at least one path is required")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    for ax, column, reference in zip(axes, ("i", "pi", "y"), (0.0, pi_star, 0.0)):
        if shade is not None:
            shade_window(ax, *shade)
        ax.axhline(reference, color=COLORS["grey"], lw=0.7)
        extra = 0
        for name, path in paths.items():
            known = name in SERIES_COLORS
            if known:
                color, style = SERIES_COLORS[name], "-" if name == "Actual" or len(paths) <= 4 else "--"
            else:                                   # learned policies: colours the fixed series do not use
                color = COLORS[_EXTRA_COLORS[extra % len(_EXTRA_COLORS)]]
                style = "-" if extra < len(_EXTRA_COLORS) else "-."
                extra += 1
            ax.plot(timestamps(path.index), path[column], color=color, ls=style,
                    lw=1.8 if name == "Actual" else 1.1 if known else 1.5, zorder=3 if name == "Actual" else 2, label=name)
        ax.set_ylabel(VARIABLE_LABELS[column])
    axes[-1].set_xlabel("Quarter")
    axes[0].legend(loc="upper right", ncol=min(len(paths), 4))
    return fig


def surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, xlabel: str, ylabel: str, zlabel: str,
            elev: float = 24.0, azim: float = -58.0) -> Figure:
    """Three-dimensional response surface on a rectangular grid.

    Args:
        X: Grid of the first input, shape ``(m, n)``.
        Y: Grid of the second input, same shape.
        Z: Response on the grid, same shape.
        xlabel: Label of the first input, with units.
        ylabel: Label of the second input, with units.
        zlabel: Label of the response, with units.
        elev: Elevation of the viewpoint in degrees.
        azim: Azimuth of the viewpoint in degrees.

    Raises:
        ValueError: If the three grids do not share one two-dimensional shape.
    """
    if not (X.ndim == 2 and X.shape == Y.shape == Z.shape):
        raise ValueError(f"X, Y and Z must share one 2-D shape, got {X.shape}, {Y.shape}, {Z.shape}")
    fig = plt.figure(figsize=(8.6, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="black", linewidth=0.35, alpha=0.95)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)
    return fig


def policy_surface_grid(panels: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], *, rate_max: float,
                        pi_star: float = 2.0, ncols: int = 2) -> Figure:
    """Filled contours of reaction functions over ``(inflation, output gap)``, one panel per policy.

    Args:
        panels: Panel title to ``(PI, Y, RATE)`` grids, e.g. from ``autofed.rl.policy_surface``.
        rate_max: Upper end of the common colour scale, percent.
        pi_star: Inflation target, drawn as a reference line.
        ncols: Number of columns of the panel grid.

    Raises:
        ValueError: If no panel is given.
    """
    if not panels:
        raise ValueError("at least one panel is required")
    n = len(panels)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols + 0.6, 3.9 * nrows), sharex=True, sharey=True, squeeze=False)
    levels = np.linspace(0.0, rate_max, 16)
    filled = None
    for ax, (title, (PI, Y, RATE)) in zip(axes.ravel(), panels.items()):
        filled = ax.contourf(PI, Y, RATE, levels=levels, cmap="viridis", extend="both")
        ax.axvline(pi_star, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(0.0, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Inflation (percent)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Output gap (percent)")
    if filled is not None:
        fig.colorbar(filled, ax=axes.ravel().tolist(), shrink=0.9, label="Policy rate (percent)")
    return fig


def training_curves(curves: Mapping[str, Sequence[float]], *, window: int = 25, ncols: int = 4) -> Figure:
    """Rolling-mean episode returns, one panel per training run.

    Args:
        curves: Panel title to the sequence of episode returns.
        window: Length of the rolling mean, in episodes.
        ncols: Number of columns of the panel grid.

    Raises:
        ValueError: If no curve is given.
    """
    if not curves:
        raise ValueError("at least one curve is required")
    n = len(curves)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols + 0.4, 2.3 * nrows + 0.3), squeeze=False)
    for ax, (title, returns) in zip(axes.ravel(), curves.items()):
        smooth = pd.Series(list(returns), dtype=float).rolling(window, min_periods=1).mean()
        ax.plot(smooth.to_numpy(), color=COLORS["blue"], lw=1.1)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Episode")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Return ({window}-episode mean)")
    return fig
