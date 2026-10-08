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

"""Plain-file persistence of tables passed between notebooks."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .exceptions import ArtifactError

__all__ = ["load_frame", "save_frame"]

_INDEX = "quarter"


def save_frame(frame: pd.DataFrame, path: str | os.PathLike[str]) -> Path:
    """Write a frame on a quarterly ``PeriodIndex`` to CSV and return the path.

    Raises:
        TypeError: If the index is not a ``PeriodIndex``.
    """
    if not isinstance(frame.index, pd.PeriodIndex):
        raise TypeError(f"expected a PeriodIndex, got {type(frame.index).__name__}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.index = out.index.astype(str)
    out.to_csv(target, index_label=_INDEX, float_format="%.10g")
    return target


def load_frame(path: str | os.PathLike[str], *, producer: str = "the notebook that writes it") -> pd.DataFrame:
    """Read a frame written by :func:`save_frame`.

    Args:
        path: CSV file.
        producer: Name of the notebook that creates the file, used in the error message.

    Raises:
        ArtifactError: If the file is missing or has no quarterly index column.
    """
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"{source} not found; run {producer} first")
    try:
        frame = pd.read_csv(source, index_col=_INDEX)
        frame.index = pd.PeriodIndex(frame.index, freq="Q", name=_INDEX)
    except (KeyError, ValueError) as exc:
        raise ArtifactError(f"{source} is not a quarterly table: {exc}") from exc
    return frame
