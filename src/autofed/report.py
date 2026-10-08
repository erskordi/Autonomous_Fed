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

"""Numbered figures, tables and PDF export for one notebook.

A notebook creates one :class:`NotebookSession` in its first code cell and routes every exhibit
through it. The session numbers exhibits in order of first appearance (``Figure 3.2`` is the second
figure of notebook 3), writes each to ``figures/<stem>/`` under a stable file name, keeps a manifest,
and converts the notebook to PDF in the last cell.
"""

from __future__ import annotations

import contextlib
import json
import re
import textwrap
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from . import _tables, style
from .exceptions import ConfigurationError
from .export import Engine, disk_execution_count, notebook_to_pdf
from .project import Workspace

__all__ = ["Exhibit", "NotebookSession"]

_SLUG: Final = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_PREFIX: Final[dict[str, str]] = {"figure": "fig", "table": "tab"}
Kind = Literal["figure", "table"]
_ACTIVE_HOOK: Any = None


@dataclass(frozen=True)
class Exhibit:
    """One numbered figure or table.

    Attributes:
        kind: ``"figure"`` or ``"table"``.
        number: Position among the exhibits of the same kind in this notebook, from 1.
        label: Display label, e.g. ``"Figure 3.2"``.
        slug: Short identifier used in the file name.
        title: Caption text without the label.
        note: Optional note printed under the exhibit.
        file: File name inside the notebook's figure directory.
    """

    kind: str
    number: int
    label: str
    slug: str
    title: str
    note: str | None
    file: str

    def _ipython_display_(self) -> None:
        """Show nothing when an exhibit is the last expression of a cell; the image is the output."""


class NotebookSession:
    """Exhibit numbering, saving and document export for one notebook.

    Args:
        stem: File name of the notebook without ``.ipynb``, e.g. ``"01_linear_svar"``.
        number: Notebook number used in exhibit labels.
        title: Human-readable notebook title, stored in the manifest.
        workspace: Output layout; discovered from the working directory when omitted.
        apply_style: Install the house Matplotlib style.
        clean: Remove this notebook's previously saved exhibits so a fresh run leaves no stale files.
        dpi: Resolution of the saved images.

    Raises:
        ValueError: If ``stem`` is empty or ``number`` is negative.

    Example:
        >>> session = NotebookSession("99_demo", number=99, workspace=Workspace(Path("/tmp/afed-doctest")))
        >>> session.label("figure", 1)
        'Figure 99.1'
    """

    def __init__(self, stem: str, *, number: int, title: str = "", workspace: Workspace | None = None,
                 apply_style: bool = True, clean: bool = True, dpi: int = 300) -> None:
        if not stem or stem.endswith(".ipynb"):
            raise ValueError(f"stem must be the notebook file name without extension, got {stem!r}")
        if number < 0:
            raise ValueError(f"number must be non-negative, got {number}")
        self.stem = stem
        self.number = int(number)
        self.title = title
        self.workspace = workspace if workspace is not None else Workspace.discover()
        self.dpi = int(dpi)
        self.figure_dir = self.workspace.figure_dir(stem)
        self._numbers: dict[str, dict[str, int]] = {"figure": {}, "table": {}}
        self._exhibits: dict[tuple[str, str], Exhibit] = {}
        self._last_cell_end = time.time()
        self._watch_cells()
        if apply_style:
            style.apply()
        if clean:
            for pattern in ("fig_*", "tab_*", "manifest.json"):
                for stale in self.figure_dir.glob(pattern):
                    stale.unlink()

    # ------------------------------------------------------------------ numbering
    def label(self, kind: Kind, number: int) -> str:
        """Display label of an exhibit, e.g. ``"Table 2.3"``."""
        return f"{kind.capitalize()} {self.number}.{number}"

    @property
    def exhibits(self) -> list[Exhibit]:
        """Registered exhibits in order of first appearance."""
        return list(self._exhibits.values())

    def _register(self, kind: Kind, slug: str, title: str, note: str | None, suffix: str) -> Exhibit:
        if not _SLUG.match(slug):
            raise ValueError(f"slug must be lower-case words joined by underscores, got {slug!r}")
        if not title.strip():
            raise ValueError("every exhibit needs a title")
        numbers = self._numbers[kind]
        number = numbers.setdefault(slug, len(numbers) + 1)      # re-running a cell keeps its number
        file = f"{_PREFIX[kind]}_{self.number:02d}_{number:02d}_{slug}{suffix}"
        exhibit = Exhibit(kind, number, self.label(kind, number), slug, title.strip(), note, file)
        self._exhibits[(kind, slug)] = exhibit
        manifest = {"notebook": self.stem, "title": self.title, "exhibits": [asdict(e) for e in self.exhibits]}
        (self.figure_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return exhibit

    def _finish(self, fig: Figure, exhibit: Exhibit, show: bool, *, fixed_layout: bool = False) -> None:
        # Table images place every element by hand. Matplotlib re-installs the rcParams layout engine
        # after a tight-bbox save, so the style's constrained layout is switched off around them.
        context = (mpl.rc_context({"figure.constrained_layout.use": False}) if fixed_layout
                   else contextlib.nullcontext())
        with context:
            fig.savefig(self.figure_dir / exhibit.file, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            if show:
                plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------ exhibits
    def figure(self, fig: Figure, slug: str, title: str, *, note: str | None = None, show: bool = True) -> Exhibit:
        """Number, caption, save and display a figure.

        The caption ``"Figure N.k: title"`` becomes the figure's super-title; ``note`` is printed
        under the axes. The image is written to ``figures/<stem>/fig_NN_kk_<slug>.png``.

        Args:
            fig: The Matplotlib figure. Axis labels, units and legends are the caller's job.
            slug: Short identifier, lower-case words joined by underscores.
            title: Caption text without the label.
            note: Source or reading note.
            show: Display the figure.

        Returns:
            The registered exhibit.

        Raises:
            TypeError: If ``fig`` is not a Matplotlib figure.
            ValueError: If ``slug`` or ``title`` is malformed.
        """
        if not isinstance(fig, Figure):
            raise TypeError(f"expected matplotlib Figure, got {type(fig).__name__}")
        exhibit = self._register("figure", slug, title, note, ".png")
        width_chars = max(int(fig.get_figwidth() * 11.5), 40)
        fig.suptitle("\n".join(textwrap.wrap(f"{exhibit.label}: {exhibit.title}", width=width_chars)),
                     fontsize=10.5, fontweight="bold", x=0.01, ha="left")
        if note:
            fig.text(0.01, -0.015, "\n".join(textwrap.wrap(note, width=int(width_chars * 1.25))),
                     ha="left", va="top", fontsize=8, fontstyle="italic", color="#333333")
        self._finish(fig, exhibit, show)
        return exhibit

    def table(self, frame: pd.DataFrame, slug: str, title: str, *, note: str | None = None,
              formats: Mapping[str, _tables.Formatter] | None = None,
              float_format: _tables.Formatter = "{:.3f}", index: bool = True, fontsize: float = 9.0,
              show: bool = True) -> Exhibit:
        """Number, render, save and display a DataFrame as a Matplotlib table.

        Writes ``figures/<stem>/tab_NN_kk_<slug>.png`` and the underlying numbers to a ``.csv`` of
        the same name. A two-level row index is drawn with the outer level as group headings.

        Args:
            frame: The table.
            slug: Short identifier, lower-case words joined by underscores.
            title: Caption text without the label.
            note: Note printed under the bottom rule.
            formats: Per-column format string or callable, keyed by column name.
            float_format: Format of floating-point cells without an entry in ``formats``.
            index: Draw the row index as the first column.
            fontsize: Body font size in points.
            show: Display the table.

        Returns:
            The registered exhibit.
        """
        header, rows, align, groups = _tables.frame_to_cells(frame, index=index, formats=formats,
                                                            float_format=float_format)
        exhibit = self._register("table", slug, title, note, ".png")
        frame.to_csv(self.figure_dir / exhibit.file.replace(".png", ".csv"), index=index)
        fig = _tables.render_table(header, rows, align, caption=f"{exhibit.label}: {exhibit.title}", note=note,
                                   group_rows=groups, fontsize=fontsize)
        self._finish(fig, exhibit, show, fixed_layout=True)
        return exhibit

    def listing(self, text: str, slug: str, title: str, *, note: str | None = None, fontsize: float = 7.5,
                show: bool = True) -> Exhibit:
        """Number, render, save and display a verbatim text block as a table exhibit.

        For estimation summaries that a library returns as formatted text. The text is also written
        to a ``.txt`` file of the same name.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("listing text must be a non-empty string")
        exhibit = self._register("table", slug, title, note, ".png")
        (self.figure_dir / exhibit.file.replace(".png", ".txt")).write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        fig = _tables.render_listing(text, caption=f"{exhibit.label}: {exhibit.title}", note=note, fontsize=fontsize)
        self._finish(fig, exhibit, show, fixed_layout=True)
        return exhibit

    # ------------------------------------------------------------------ artifacts and export
    def artifacts(self, *parts: str) -> Path:
        """Create and return a sub-directory of the shared artifacts directory."""
        return self.workspace.artifact_dir(*parts)

    def export_pdf(self, *, engine: Engine = "auto", wait: float = 20.0, allow_stale: bool = False) -> Path | None:
        """Convert this notebook to ``documents/<stem>.pdf``. Call it in the last cell.

        A kernel can only convert the copy of the notebook that is saved on disk. The method waits
        up to ``wait`` seconds for the file to contain the outputs of the cells executed so far
        (editors with auto-save write them within moments). If the file is still behind, nothing is
        exported and a warning says to save the notebook and run the cell again; a stale PDF is
        never written silently.

        Args:
            engine: ``"auto"``, ``"webpdf"`` or ``"latex"``; see :func:`autofed.export.notebook_to_pdf`.
            wait: Seconds to wait for the saved file to catch up with the kernel.
            allow_stale: Export whatever is on disk without the freshness check.

        Returns:
            Path of the PDF, or ``None`` if the export was skipped.

        Raises:
            ConfigurationError: If the notebook file is not in the workspace root.
            ExportError: If the conversion itself fails.
        """
        notebook = self.workspace.root / f"{self.stem}.ipynb"
        if not notebook.is_file():
            raise ConfigurationError(f"{notebook} not found; run the notebook from its own directory "
                                     f"or set AUTOFED_WORKSPACE")
        if not allow_stale and not self._wait_until_saved(notebook, wait):
            warnings.warn(
                f"{notebook.name} on disk does not yet contain this run's outputs, so no PDF was written. "
                "Save the notebook and run this cell again.",
                stacklevel=2,
            )
            return None
        target = notebook_to_pdf(notebook, self.workspace.documents, engine=engine)
        print(f"Saved {target.relative_to(self.workspace.root)}")
        return target

    @staticmethod
    def _shell() -> Any:
        try:
            from IPython import get_ipython
        except ImportError:                                        # plain Python
            return None
        return get_ipython()

    def _watch_cells(self) -> None:
        """Record when each cell finishes, so the export can tell a fresh save from an old one."""
        global _ACTIVE_HOOK  # noqa: PLW0603 - one hook per kernel, replaced when the session is re-created
        shell = self._shell()
        if shell is None:
            return
        if _ACTIVE_HOOK is not None:
            with contextlib.suppress(ValueError):
                shell.events.unregister("post_run_cell", _ACTIVE_HOOK)

        def hook(*_: Any) -> None:
            self._last_cell_end = time.time()

        shell.events.register("post_run_cell", hook)
        _ACTIVE_HOOK = hook

    def _wait_until_saved(self, notebook: Path, wait: float) -> bool:
        """True once the file on disk was written after the previous cell finished and holds its outputs."""
        shell = self._shell()
        if shell is None:
            return True
        needed = int(shell.execution_count) - 1                    # every cell before the export cell
        deadline = time.monotonic() + max(wait, 0.0)
        while True:
            saved_after_run = notebook.stat().st_mtime >= self._last_cell_end - 1.0
            if saved_after_run and disk_execution_count(notebook) >= needed:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)
