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

"""Rendering of tables and verbatim listings as Matplotlib figures (private)."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

Formatter = str | Callable[[Any], str]

_CHAR_EM = 0.56          # mean glyph width of the body font, in em
_ROW_EM = 2.0            # row height, in em
_PAD_CHARS = 2.5         # horizontal padding per column, in characters
_BOLD = 1.18             # bold headings run wider than body text
_MARGIN_IN = 0.12        # outer margin, inches
_MIN_WIDTH_IN = 3.6


def format_value(value: Any, fmt: Formatter) -> str:
    """Format one cell: strings pass through, missing values are blank, floats use ``fmt``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return fmt.format(number) if isinstance(fmt, str) else fmt(number)
    if value is pd.NaT:
        return ""
    return str(value)


def frame_to_cells(
    frame: pd.DataFrame,
    *,
    index: bool = True,
    formats: Mapping[str, Formatter] | None = None,
    float_format: Formatter = "{:.3f}",
) -> tuple[list[str], list[list[str]], list[str], list[int]]:
    """Convert a DataFrame to header, body rows, column alignments and group-row positions.

    A two-level row index is rendered with the outer level as bold group rows.

    Raises:
        TypeError: If ``frame`` is not a DataFrame.
        ValueError: If ``frame`` has no columns, or its row index has more than two levels.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(frame).__name__}")
    if frame.shape[1] == 0:
        raise ValueError("cannot render a table without columns")
    formats = dict(formats or {})
    grouped = index and isinstance(frame.index, pd.MultiIndex)
    if grouped and frame.index.nlevels != 2:
        raise ValueError("row MultiIndex must have exactly two levels (group, row)")

    columns = [" ".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
               for col in frame.columns]
    header = ([str(frame.index.names[-1] or "")] if index else []) + columns
    numeric = [pd.api.types.is_numeric_dtype(frame.iloc[:, j]) for j in range(frame.shape[1])]
    align = (["left"] if index else []) + ["right" if flag else "left" for flag in numeric]

    rows: list[list[str]] = []
    groups: list[int] = []
    previous_group: object = object()
    for position in range(frame.shape[0]):
        key = frame.index[position]
        if grouped:
            group, row_label = key
            if group != previous_group:
                groups.append(len(rows))
                rows.append([str(group)] + [""] * len(columns))
                previous_group = group
        else:
            row_label = key
        cells = [format_value(frame.iat[position, j], formats.get(frame.columns[j], formats.get(columns[j], float_format)))
                 for j in range(frame.shape[1])]
        rows.append(([str(row_label)] if index else []) + cells)
    return header, rows, align, groups


def _display_len(text: str) -> int:
    """Approximate rendered length; mathtext markup does not occupy space."""
    if "$" in text:
        for token in ("\\hat", "\\bar", "\\varepsilon", "\\sigma", "\\pi", "\\chi", "\\Delta", "\\alpha", "\\beta"):
            text = text.replace(token, "x")
        text = "".join(ch for ch in text if ch not in "$\\{}^_")
    return len(text)


def render_table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: Sequence[str],
    *,
    caption: str,
    note: str | None = None,
    group_rows: Sequence[int] = (),
    fontsize: float = 9.0,
) -> Figure:
    """Draw a booktabs-style table (top, header and bottom rules only) and return the figure.

    Args:
        header: Column headings.
        rows: Body rows of already formatted strings.
        align: ``"left"`` or ``"right"`` per column.
        caption: Numbered caption drawn above the table, e.g. ``"Table 2.1: ..."``.
        note: Optional note drawn below the bottom rule.
        group_rows: Positions in ``rows`` that are group headings (bold, first cell only).
        fontsize: Body font size in points.

    Raises:
        ValueError: If the rows are ragged or ``align`` does not match the header.
    """
    n_cols = len(header)
    if len(align) != n_cols or any(len(row) != n_cols for row in rows):
        raise ValueError("header, align and every row must have the same number of columns")
    groups = set(group_rows)

    em = fontsize / 72.0
    char_w, row_h = _CHAR_EM * em, _ROW_EM * em
    widths = [
        (max([_display_len(header[j]) * _BOLD] + [_display_len(row[j]) for k, row in enumerate(rows) if k not in groups or j > 0]
             + [4]) + _PAD_CHARS) * char_w
        for j in range(n_cols)
    ]
    if groups:
        widths[0] = max(widths[0], (max(_display_len(rows[k][0]) for k in groups) + _PAD_CHARS) * char_w * 0.6)
    scale = max(1.0, _MIN_WIDTH_IN / sum(widths))
    widths = [w * scale for w in widths]
    table_w = sum(widths)

    wrap_at = max(int(table_w / (char_w * 0.95)), 30)
    caption_lines = textwrap.wrap(caption, width=wrap_at) or [""]
    note_lines = textwrap.wrap(note, width=int(wrap_at * 1.15)) if note else []
    caption_h = (len(caption_lines) * 1.35 + 0.5) * em * 1.08
    note_h = (len(note_lines) * 1.3 + 0.6) * em * 0.9 if note_lines else 0.0
    body_h = (len(rows) + 1) * row_h

    fig_w, fig_h = table_w + 2 * _MARGIN_IN, caption_h + body_h + note_h + 2 * _MARGIN_IN
    fig = plt.figure(figsize=(fig_w, fig_h), layout="none")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, fig_w)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")

    x0, x1 = _MARGIN_IN, _MARGIN_IN + table_w
    y = fig_h - _MARGIN_IN
    for k, line in enumerate(caption_lines):
        ax.text(x0, y - (k + 0.75) * 1.35 * em * 1.08, line, fontsize=fontsize * 1.08, fontweight="bold",
                ha="left", va="center")
    top = y - caption_h

    edges = [x0]
    for width in widths:
        edges.append(edges[-1] + width)

    def put(col: int, y_mid: float, text: str, **style: Any) -> None:
        pad = 0.5 * _PAD_CHARS * char_w
        if align[col] == "right":
            ax.text(edges[col + 1] - pad, y_mid, text, ha="right", va="center", fontsize=fontsize, **style)
        else:
            ax.text(edges[col] + pad, y_mid, text, ha="left", va="center", fontsize=fontsize, **style)

    ax.plot([x0, x1], [top, top], color="black", lw=1.1)
    for j, text in enumerate(header):
        put(j, top - 0.5 * row_h, text, fontweight="bold")
    ax.plot([x0, x1], [top - row_h, top - row_h], color="black", lw=0.6)
    for k, row in enumerate(rows):
        y_mid = top - (k + 1.5) * row_h
        if k in groups:
            ax.text(x0 + 0.5 * _PAD_CHARS * char_w, y_mid, row[0], ha="left", va="center", fontsize=fontsize,
                    fontweight="bold", fontstyle="italic")
            continue
        for j, text in enumerate(row):
            put(j, y_mid, text)
    bottom = top - body_h
    ax.plot([x0, x1], [bottom, bottom], color="black", lw=1.1)
    for k, line in enumerate(note_lines):
        ax.text(x0, bottom - (k + 0.9) * 1.3 * em * 0.9, line, fontsize=fontsize * 0.9, fontstyle="italic",
                ha="left", va="center", color="#333333")
    return fig


def render_listing(text: str, *, caption: str, note: str | None = None, fontsize: float = 7.5) -> Figure:
    """Draw a verbatim, monospaced text block (for library-generated summaries) and return the figure."""
    lines = text.rstrip("\n").expandtabs(4).split("\n")
    em = fontsize / 72.0
    width = max(max((len(line) for line in lines), default=10), 40) * 0.602 * em
    wrap_at = max(int(width / (0.56 * em * 1.2)), 30)
    caption_lines = textwrap.wrap(caption, width=wrap_at) or [""]
    note_lines = textwrap.wrap(note, width=int(wrap_at * 1.15)) if note else []
    line_h = 1.32 * em
    caption_h = (len(caption_lines) * 1.35 + 0.6) * em * 1.25
    note_h = (len(note_lines) * 1.3 + 0.6) * em * 1.1 if note_lines else 0.0
    body_h = (len(lines) + 1) * line_h

    fig_w, fig_h = width + 2 * _MARGIN_IN, caption_h + body_h + note_h + 2 * _MARGIN_IN
    fig = plt.figure(figsize=(fig_w, fig_h), layout="none")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, fig_w)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")
    x0, y = _MARGIN_IN, fig_h - _MARGIN_IN
    for k, line in enumerate(caption_lines):
        ax.text(x0, y - (k + 0.75) * 1.35 * em * 1.25, line, fontsize=fontsize * 1.25, fontweight="bold",
                ha="left", va="center")
    top = y - caption_h
    ax.plot([x0, x0 + width], [top, top], color="black", lw=1.1)
    for k, line in enumerate(lines):
        ax.text(x0, top - (k + 1.0) * line_h, line, fontsize=fontsize, family="DejaVu Sans Mono", ha="left",
                va="center")
    bottom = top - body_h
    ax.plot([x0, x0 + width], [bottom, bottom], color="black", lw=1.1)
    for k, line in enumerate(note_lines):
        ax.text(x0, bottom - (k + 0.9) * 1.3 * em * 1.1, line, fontsize=fontsize * 1.1, fontstyle="italic",
                ha="left", va="center", color="#333333")
    return fig
