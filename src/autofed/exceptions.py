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

"""Exception hierarchy of :mod:`autofed`."""

from __future__ import annotations

__all__ = ["ArtifactError", "AutofedError", "ConfigurationError", "DataError", "ExportError", "SpecificationError"]


class AutofedError(Exception):
    """Base class of every error raised by :mod:`autofed`."""


class ConfigurationError(AutofedError):
    """A required setting (API key, directory, optional dependency) is missing or invalid."""


class DataError(AutofedError):
    """Downloaded or loaded data fail an integrity check."""


class ArtifactError(AutofedError):
    """A saved artifact is missing, malformed, or incompatible with the request."""


class SpecificationError(AutofedError):
    """A model specification is inconsistent with the data or with another specification."""


class ExportError(AutofedError):
    """A notebook could not be converted to a document."""
