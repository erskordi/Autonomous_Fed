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

"""The quarterly macro panel: download from FRED, sample windows, persistence.

The panel is fetched once (notebook ``00_data``) and saved; every other notebook loads the saved
copy, so all results rest on one data vintage.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from .exceptions import ArtifactError, ConfigurationError, DataError

__all__ = ["FRED_KEY_ENV", "NAMES", "SERIES", "MacroPanel", "SampleConfig", "fetch_macro_panel", "load_macro_panel"]

#: Model variables in recursive order: output gap, inflation, policy rate.
NAMES: Final[tuple[str, str, str]] = ("y", "pi", "i")
FRED_KEY_ENV: Final = "FRED_API_KEY"

#: FRED series behind each stored column, with the request options used.
SERIES: Final[dict[str, tuple[str, dict[str, str]]]] = {
    "pi": ("GDPDEF", {"units": "pc1"}),
    "gdp_real": ("GDPC1", {"units": "lin"}),
    "gdp_potential": ("GDPPOT", {"units": "lin"}),
    "i": ("FEDFUNDS", {"aggregation_method": "avg"}),
}
_CSV: Final = "macro_panel.csv"
_META: Final = "macro_panel.json"


@dataclass(frozen=True)
class SampleConfig:
    """Sample windows of the project.

    Attributes:
        train_start: First quarter of the estimation (training) window.
        train_end: Last quarter of the estimation window.
        oos_start: First out-of-sample quarter.
        tvp_training_quarters: Quarters before ``train_start`` that calibrate the TVP priors.
        order: VAR lag order; that many further pre-sample quarters supply initial lags.
        fetch_start: First observation requested from FRED.
    """

    train_start: str = "1987Q3"
    train_end: str = "2007Q2"
    oos_start: str = "2007Q3"
    tvp_training_quarters: int = 40
    order: int = 2
    fetch_start: str = "1976-07-01"

    def __post_init__(self) -> None:
        start, end, oos = (pd.Period(p, "Q") for p in (self.train_start, self.train_end, self.oos_start))
        if not start < end < oos:
            raise ValueError("windows must satisfy train_start < train_end < oos_start")
        if self.tvp_training_quarters < 1 or self.order < 1:
            raise ValueError("tvp_training_quarters and order must be positive")


@dataclass(frozen=True)
class MacroPanel:
    """Quarterly panel of the output gap, inflation and the policy rate.

    Attributes:
        data: Columns ``y``, ``pi``, ``i`` (percent) plus the GDP levels ``gdp_real`` and
            ``gdp_potential``, on a gap-free quarterly ``PeriodIndex``.
        config: Sample windows.
        retrieved: ISO date on which the data were downloaded.
        source: FRED series identifier behind each column.

    Raises:
        DataError: If the frame fails an integrity check.
    """

    data: pd.DataFrame
    config: SampleConfig = field(default_factory=SampleConfig)
    retrieved: str = ""
    source: Mapping[str, str] = field(default_factory=lambda: {k: v[0] for k, v in SERIES.items()})

    def __post_init__(self) -> None:
        frame = self.data
        if not isinstance(frame.index, pd.PeriodIndex) or not str(frame.index.freqstr).upper().startswith("Q"):
            raise DataError("panel index must be a quarterly PeriodIndex")
        missing = [c for c in NAMES if c not in frame.columns]
        if missing:
            raise DataError(f"panel lacks columns {missing}")
        if not frame.index.equals(pd.period_range(frame.index[0], frame.index[-1], freq="Q")):
            raise DataError("panel has missing or unsorted quarters")
        if not np.isfinite(frame.loc[:, list(NAMES)].to_numpy(dtype=float)).all():
            raise DataError("panel contains missing or non-finite values in y, pi or i")
        cfg = self.config
        start, end = pd.Period(cfg.train_start, "Q"), pd.Period(cfg.train_end, "Q")
        if start not in frame.index or end not in frame.index:
            raise DataError(f"panel ({frame.index[0]} to {frame.index[-1]}) does not cover the training window")
        if pd.Period(cfg.oos_start, "Q") not in frame.index:
            raise DataError(f"panel ends at {frame.index[-1]}, before the out-of-sample start {cfg.oos_start}")
        presample = int(frame.index.get_loc(start))
        needed = cfg.tvp_training_quarters + cfg.order
        if presample < needed:
            raise DataError(f"{presample} quarters precede {cfg.train_start}; the TVP training sample needs {needed}")

    def _slice(self, first: str | None, last: str | None) -> pd.DataFrame:
        return self.data.loc[first:last, list(NAMES)].copy()

    @property
    def train(self) -> pd.DataFrame:
        """Estimation window, columns ``y, pi, i``."""
        return self._slice(self.config.train_start, self.config.train_end)

    @property
    def oos(self) -> pd.DataFrame:
        """Out-of-sample window, columns ``y, pi, i``."""
        return self._slice(self.config.oos_start, None)

    @property
    def full(self) -> pd.DataFrame:
        """Estimation plus out-of-sample window, columns ``y, pi, i``."""
        return self._slice(self.config.train_start, None)

    @property
    def tvp(self) -> pd.DataFrame:
        """``full`` preceded by the TVP training sample and the initial lags."""
        start = int(self.data.index.get_loc(pd.Period(self.config.train_start, "Q")))
        return self.data.iloc[start - (self.config.tvp_training_quarters + self.config.order):][list(NAMES)].copy()

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Write the panel (``macro_panel.csv``) and its metadata (``macro_panel.json``).

        Returns:
            The directory written to.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        out = self.data.copy()
        out.index = out.index.astype(str)
        out.to_csv(target / _CSV, index_label="quarter", float_format="%.10g")
        meta = {"config": asdict(self.config), "retrieved": self.retrieved, "source": dict(self.source)}
        (target / _META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return target


def load_macro_panel(directory: str | os.PathLike[str]) -> MacroPanel:
    """Load a panel written by :meth:`MacroPanel.save`.

    Raises:
        ArtifactError: If the files are missing or malformed; the message names the notebook to run.
    """
    source = Path(directory)
    csv, meta_path = source / _CSV, source / _META
    if not csv.is_file() or not meta_path.is_file():
        raise ArtifactError(f"no macro panel in {source}; run notebook 00_data first")
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv, index_col="quarter")
        frame.index = pd.PeriodIndex(frame.index, freq="Q", name="quarter")
        config = SampleConfig(**meta["config"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"macro panel in {source} is malformed: {exc}") from exc
    return MacroPanel(frame, config, str(meta.get("retrieved", "")), dict(meta.get("source", {})))


def fetch_macro_panel(api_key: str | None = None, *, config: SampleConfig | None = None) -> MacroPanel:
    """Download the four FRED series and build the panel.

    Inflation is the year-over-year percent change of the GDP deflator, the output gap is
    ``100 * (GDPC1 / GDPPOT - 1)``, and the policy rate is the quarterly average of the effective
    federal funds rate.

    Args:
        api_key: FRED API key; read from the ``FRED_API_KEY`` environment variable when omitted.
        config: Sample windows; the project defaults when omitted.

    Returns:
        The validated panel, from the first complete quarter on.

    Raises:
        ConfigurationError: If no API key is available or ``fedfred`` is not installed.
        DataError: If a response is malformed or the panel fails validation.
    """
    cfg = config if config is not None else SampleConfig()
    key = api_key or os.environ.get(FRED_KEY_ENV)
    if not key:
        raise ConfigurationError(f"no FRED API key: pass api_key or set the {FRED_KEY_ENV} environment variable")
    try:
        import fedfred as fd
    except ImportError as exc:                                     # pragma: no cover - declared dependency
        raise ConfigurationError("fedfred is required to download data") from exc

    client = fd.FredAPI(api_key=key, cache_mode=True, cache_size=256)
    columns: dict[str, pd.Series] = {}
    for name, (series_id, options) in SERIES.items():
        observations = client.get_series_observations(
            series_id=series_id, observation_start=cfg.fetch_start, frequency="q", **options
        )
        if not isinstance(observations, pd.DataFrame) or "value" not in observations.columns:
            raise DataError(f"unexpected FRED response for {series_id}: {type(observations).__name__}")
        columns[name] = fd.FredHelpers.to_pd_series(observations, name)

    frame = pd.concat(columns, axis=1, sort=True)
    frame["y"] = 100.0 * (frame["gdp_real"] / frame["gdp_potential"] - 1.0)
    frame = frame.loc[:, [*NAMES, "gdp_real", "gdp_potential"]].dropna()
    frame.index = pd.PeriodIndex(frame.index, freq="Q", name="quarter")
    return MacroPanel(frame, cfg, dt.date.today().isoformat())
