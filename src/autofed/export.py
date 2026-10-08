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

"""Conversion of an executed notebook to PDF through ``nbconvert``."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final, Literal

from .exceptions import ConfigurationError, ExportError

__all__ = ["Engine", "disk_execution_count", "notebook_to_pdf"]

Engine = Literal["auto", "webpdf", "latex"]
_NBCONVERT_FORMAT: Final[dict[str, str]] = {"webpdf": "webpdf", "latex": "pdf"}


def disk_execution_count(notebook: Path) -> int:
    """Largest ``execution_count`` among the code cells of the notebook file as saved on disk.

    Args:
        notebook: Path of the ``.ipynb`` file.

    Returns:
        The largest execution count, or ``0`` if no code cell has been executed.

    Raises:
        ExportError: If the file is missing or is not a notebook document.
    """
    if not notebook.is_file():
        raise ExportError(f"notebook {notebook} does not exist")
    try:
        document = json.loads(notebook.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"could not read {notebook} as a notebook: {exc}") from exc
    if not isinstance(document, dict) or "cells" not in document:
        raise ExportError(f"{notebook} has no 'cells' entry; not a notebook document")
    counts = [cell.get("execution_count") or 0 for cell in document["cells"] if cell.get("cell_type") == "code"]
    return max(counts, default=0)


def _available_engines() -> list[str]:
    engines = []
    if importlib.util.find_spec("playwright") is not None:
        engines.append("webpdf")
    if shutil.which("xelatex") is not None and shutil.which("pandoc") is not None:
        engines.append("latex")
    return engines


def notebook_to_pdf(notebook: Path, out_dir: Path, *, engine: Engine = "auto", timeout: float = 900.0) -> Path:
    """Convert the notebook file on disk to ``<out_dir>/<stem>.pdf``.

    ``webpdf`` prints the HTML rendering with headless Chromium (needs ``playwright`` and its
    Chromium build); ``latex`` goes through pandoc and XeLaTeX. ``auto`` tries every engine that is
    installed, ``webpdf`` first.

    Args:
        notebook: Path of the ``.ipynb`` file. The file is converted as saved; it is not executed.
        out_dir: Directory of the PDF; created if absent.
        engine: Conversion engine.
        timeout: Seconds allowed per engine.

    Returns:
        Path of the PDF.

    Raises:
        ConfigurationError: If ``nbconvert`` or every usable engine is missing.
        ExportError: If the notebook is missing or every attempted engine fails.
    """
    if engine not in ("auto", "webpdf", "latex"):
        raise ValueError(f"engine must be 'auto', 'webpdf' or 'latex', got {engine!r}")
    if not notebook.is_file():
        raise ExportError(f"notebook {notebook} does not exist")
    if importlib.util.find_spec("nbconvert") is None:
        raise ConfigurationError("nbconvert is not installed; install the extra with `pip install autofed[notebooks]`")
    engines = _available_engines() if engine == "auto" else [engine]
    if not engines:
        raise ConfigurationError(
            "no PDF engine found: install playwright (`playwright install chromium`) for webpdf, "
            "or pandoc and XeLaTeX for latex"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{notebook.stem}.pdf"
    failures: list[str] = []
    for name in engines:
        command = [sys.executable, "-m", "nbconvert", "--to", _NBCONVERT_FORMAT[name], str(notebook),
                   "--output-dir", str(out_dir), "--output", notebook.stem]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            failures.append(f"{name}: timed out after {timeout:.0f} s")
            continue
        if done.returncode == 0 and target.is_file():
            return target
        failures.append(f"{name}: exit {done.returncode}: {done.stderr.strip()[-600:]}")
    raise ExportError(f"could not convert {notebook.name} to PDF. " + " | ".join(failures))
