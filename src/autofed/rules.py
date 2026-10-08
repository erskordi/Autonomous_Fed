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

"""Benchmark interest-rate rules in the level form of Hinterlang & Tänzer (2021, Table 4)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
import pandas as pd

__all__ = ["BA", "NPP", "REFERENCE_RULES", "TR93", "PolicyRule", "prescriptions"]

_Num = TypeVar("_Num", float, np.ndarray, pd.Series)


@dataclass(frozen=True)
class PolicyRule:
    """Linear rule ``i = alpha0 + beta_pi * pi + beta_y * y`` with ``alpha0 = r* - (beta_pi - 1) pi*``.

    Attributes:
        name: Label used in legends, tables and file names.
        beta_pi: Total response to inflation (not to the inflation gap).
        beta_y: Response to the output gap.
        r_star: Long-run real rate, percent.
        pi_star: Inflation target, percent.

    Example:
        >>> PolicyRule("TR93", 1.5, 0.5).alpha0
        1.0
        >>> PolicyRule("TR93", 1.5, 0.5).prescribe(2.0, 0.0)
        4.0
    """

    name: str
    beta_pi: float
    beta_y: float
    r_star: float = 2.0
    pi_star: float = 2.0

    @property
    def alpha0(self) -> float:
        """Intercept implied by ``r_star``, ``pi_star`` and ``beta_pi``."""
        return self.r_star - (self.beta_pi - 1.0) * self.pi_star

    def prescribe(self, pi: _Num, y: _Num, *, zlb: bool = False) -> _Num:
        """Prescribed policy rate for scalars, arrays or Series; optionally floored at zero."""
        rate = self.alpha0 + self.beta_pi * pi + self.beta_y * y
        if not zlb:
            return rate
        if isinstance(rate, pd.Series):
            return rate.clip(lower=0.0)
        return np.maximum(rate, 0.0) if isinstance(rate, np.ndarray) else max(rate, 0.0)


TR93: Final = PolicyRule("TR93", beta_pi=1.5, beta_y=0.5)
NPP: Final = PolicyRule("NPP", beta_pi=2.0, beta_y=0.5)
BA: Final = PolicyRule("BA", beta_pi=1.5, beta_y=1.0)
REFERENCE_RULES: Final[dict[str, PolicyRule]] = {rule.name: rule for rule in (TR93, NPP, BA)}


def prescriptions(frame: pd.DataFrame, rules: Iterable[PolicyRule] | None = None, *, zlb: bool = False) -> pd.DataFrame:
    """Static prescriptions of each rule on realised data.

    Args:
        frame: Data with columns ``pi``, ``y`` and ``i``.
        rules: Rules to evaluate; the three reference rules when omitted.
        zlb: Floor the prescriptions at zero.

    Returns:
        A frame with column ``Actual`` (the realised rate) and one column per rule.

    Raises:
        KeyError: If a required column is missing.
    """
    missing = [c for c in ("pi", "y", "i") if c not in frame.columns]
    if missing:
        raise KeyError(f"frame lacks columns {missing}")
    out = pd.DataFrame({"Actual": frame["i"]})
    for rule in (REFERENCE_RULES.values() if rules is None else rules):
        out[rule.name] = rule.prescribe(frame["pi"], frame["y"], zlb=zlb)
    return out
