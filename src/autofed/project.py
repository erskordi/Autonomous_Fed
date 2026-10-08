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

"""Directory layout shared by the research notebooks.

Every notebook writes into three sibling directories of the notebook folder::

    <workspace>/figures/<notebook-stem>/    figures and table images of one notebook
    <workspace>/documents/                  one PDF per notebook
    <workspace>/artifacts/                  data and fitted models passed between notebooks

The workspace is the kernel's working directory unless ``AUTOFED_WORKSPACE`` is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError

__all__ = ["WORKSPACE_ENV", "Workspace"]

WORKSPACE_ENV = "AUTOFED_WORKSPACE"


@dataclass(frozen=True)
class Workspace:
    """Resolved output directories of the notebook collection.

    Attributes:
        root: Directory that holds the notebooks.

    Example:
        >>> ws = Workspace(Path("/tmp/afed-doctest"))
        >>> ws.figures.name, ws.documents.name, ws.artifacts.name
        ('figures', 'documents', 'artifacts')
    """

    root: Path

    @classmethod
    def discover(cls, root: str | os.PathLike[str] | None = None) -> Workspace:
        """Resolve the workspace from an explicit path, ``AUTOFED_WORKSPACE``, or the working directory.

        Args:
            root: Explicit workspace directory; takes precedence over the environment variable.

        Returns:
            The workspace.

        Raises:
            ConfigurationError: If the resolved path exists and is not a directory.
        """
        chosen = root if root is not None else os.environ.get(WORKSPACE_ENV)
        path = Path(chosen).expanduser() if chosen else Path.cwd()
        path = path.resolve()
        if path.exists() and not path.is_dir():
            raise ConfigurationError(f"workspace {path} exists and is not a directory")
        return cls(path)

    @property
    def figures(self) -> Path:
        """Shared figures directory."""
        return self.root / "figures"

    @property
    def documents(self) -> Path:
        """Shared documents (PDF) directory."""
        return self.root / "documents"

    @property
    def artifacts(self) -> Path:
        """Shared artifacts directory."""
        return self.root / "artifacts"

    def figure_dir(self, stem: str) -> Path:
        """Create and return the figure directory of one notebook."""
        path = self.figures / stem
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_dir(self, *parts: str) -> Path:
        """Create and return a sub-directory of the artifacts directory."""
        path = self.artifacts.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
