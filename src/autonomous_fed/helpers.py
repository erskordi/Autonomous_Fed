"""
Helper functions for the linear environment.
"""
from datetime import datetime, timedelta
from typing import Union, Tuple
from dateutil.relativedelta import relativedelta
import pandas as pd
import statsmodels.api as sm #pragma: no cover
import numpy as np
import fedfred as fd

class LinearEnvironmentHelpers:
    """
    Helper functions for the linear environment.
    """
    @staticmethod
    def to_series(x: Union[pd.Series, pd.DataFrame], name: str) -> pd.Series:
        """
        Accepts a Series or a DataFrame with 'date'/'value' columns (fedfred style) and returns a float Series with DatetimeIndex and the given `name`.

        Args:
            x (pd.Series | pd.DataFrame): Input data to be converted.
            name (str): Name to assign to the resulting Series.

        Returns:
            pd.Series: A float Series with DatetimeIndex and the given `name`.

        Raises:
            TypeError: If the input is neither a pd.Series nor a pd.DataFrame.
        """
        if isinstance(x, pd.Series):
            s = x.copy()
            s.index = pd.to_datetime(s.index)
            s = s.astype(float)
            s.name = name
            return s

        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pd.Series or pd.DataFrame")

        df: pd.DataFrame = x.copy()

        # 1) Identify the datetime index/column
        assert isinstance(df.index.name, str)
        if df.index.name and df.index.name.lower() == "date":
            idx: Union[pd.DatetimeIndex, pd.Series] = pd.to_datetime(df.index)
        elif "date" in df.columns:
            idx = pd.to_datetime(df["date"])
        else:
            # try any column with 'date' in the name
            cand = next((c for c in df.columns if "date" in c.lower()), None)
            if cand is not None:
                idx = pd.to_datetime(df[cand])
            else:
                # last resort: try to_datetime on the current index
                idx = pd.to_datetime(df.index)

        # 2) Identify the value column (prefer 'value', else first numeric)
        if "value" in df.columns:
            vals = pd.to_numeric(df["value"], errors="coerce")
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                # take the last column and coerce
                vals = pd.to_numeric(df.iloc[:, -1], errors="coerce")
            else:
                vals = pd.to_numeric(df[num_cols[0]], errors="coerce")

        s = pd.Series(vals.values, index=idx, name=name).astype(float)
        # Clean up: sort, dedupe, and ensure DatetimeIndex
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = pd.to_datetime(s.index)
        return s

    @staticmethod
    def drop_insignificant_lag_two(
        model: sm.regression.linear_model.RegressionResultsWrapper,
        x: pd.DataFrame,
        y: pd.Series,
        alpha: float = 0.10,
    ) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
        """
        Drop any lag-2 regressors with p-values > alpha, then refit OLS.

        Args:
            model (sm.regression.linear_model.RegressionResultsWrapper): Fitted OLS model.
            x (pd.DataFrame): Design matrix used in the regression.
            y (pd.Series): Target variable used in the regression.
            alpha (float, optional): Significance level for dropping lag-2 terms. Defaults to 0.10.

        Returns:
            Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]: Fitted OLS model and design matrix after dropping insignificant lag-2 terms.

        Raises:
            None
        """
        drop_cols = [
            c for c in x.columns if c.endswith("_lag2") and c in model.pvalues.index and model.pvalues[c] > alpha
        ]
        if drop_cols:
            x2 = x.drop(columns=drop_cols)
            # Align x2 and y to the same index
            x2, y_aligned = x2.align(y, join="inner", axis=0)
            model2 = sm.OLS(y_aligned, x2).fit()
            return model2, x2
        return model, x

    @staticmethod
    def bump_date_by_one_period(date: Union[str, datetime], frequency: str = "Q") -> str:
        """
        Bump a date by one period.

        Args:
            date (Union[str, datetime]): Input date.
            frequency (str): Frequency code ('D', 'W', 'BW', 'M', 'Q', 'SA', 'A', etc.).

        Returns:
            datetime: Date bumped by one period.

        Raises:
            ValueError: If frequency is not recognized.
        """
        freq = frequency.upper()

        if isinstance(date, str):
            fd.FredHelpers.datestring_validation(date)
            dt_date = datetime.strptime(date, "%Y-%m-%d")
        elif isinstance(date, datetime):
            dt_date = date
        else:
            raise TypeError("date must be a str or datetime")

        if freq == 'D':
            result = dt_date + timedelta(days=1)
        elif freq in {'W', 'WEF', 'WETH', 'WEW', 'WETU', 'WEM', 'WESU', 'WESA'}:
            result = dt_date + timedelta(weeks=1)
        elif freq in {'BW', 'BWEW', 'BWEM'}:
            result = dt_date + timedelta(weeks=2)
        elif freq == 'M':
            result = dt_date + relativedelta(months=1)
        elif freq == 'Q':
            result = dt_date + relativedelta(months=3)
        elif freq == 'SA':
            result = dt_date + relativedelta(months=6)
        elif freq == 'A':
            result = dt_date + relativedelta(years=1)
        else:
            # logically unreachable but prevents mypy warning
            raise AssertionError(f"Unhandled frequency case: {freq}")

        return fd.FredHelpers.datetime_conversion(result)

    @staticmethod
    def get_recursive_lag(df: pd.DataFrame, t: Union[pd.Period, pd.Timestamp, int], col: str, k: int) -> float:
        """
        Use forecast if available, else actual, at lag k.

        This helper method returns the value of variable `col` at lag `k`
        relative to time period `t`. If a forecasted value (`<col>_hat`) exists
        and is not NaN, it is used; otherwise, the corresponding actual value
        (`<col>`) is returned. This logic enables recursive or dynamic forecasts
        where previous forecasted values should be used once available.

        Args:
            df (pd.DataFrame): DataFrame containing both actual and forecasted values.
            t (Hashable): Current time period index.
            col (str): Column name of the variable.
            k (int): Lag period.

        Returns:
            float: The value of `col` at lag `k`, using forecast if available.

        Raises:
            KeyError: If the required column or lagged index does not exist in ``df``.
        """
        # Resolve label 't' to a single integer position
        pos = df.index.get_loc(t)
        if isinstance(pos, (slice, np.ndarray)):
            raise ValueError("Index key 't' must map to exactly one row (unique index required).")

        row = df.iloc[pos - k]

        # Prefer forecasted value if present and non-NaN
        hat_col = f"{col}_hat"
        if hat_col not in row or col not in row:
            raise KeyError(f"Expected columns '{col}' and '{hat_col}' not found.")

        hat_val = row[hat_col]
        if pd.notna(hat_val):
            # Cast robustly to float (handles numpy scalars, decimals, etc.)
            return float(pd.to_numeric(hat_val))

        return float(pd.to_numeric(row[col]))

    @staticmethod
    def coerce_frequency_string(freq: str) -> str:
        """
        Coerce pandas incompatible frequency strings to compatible ones.

        Args:
            freq (str): Input frequency string.

        Returns:
            str: Coerced frequency string compatible with pandas.

        Raises:
            None
        """
        freq = freq.upper()
        # Passthrough compatible frequencies
        if freq in {'D', 'M', 'Q', 'W'}:
            return freq
        # Annual -> Yearly
        elif freq == 'A':
            return 'Y'
        # Weekly End variants
        elif freq == 'WEF':
            return 'W-FRI'
        elif freq == 'WETH':
            return 'W-THU'
        elif freq == 'WEW':
            return 'W-WED'
        elif freq == 'WETU':
            return 'W-TUE'
        elif freq == 'WEM':
            return 'W-MON'
        elif freq == 'WESU':
            return 'W-SUN'
        elif freq == 'WESA':
            return 'W-SAT'
        # Biweekly -> 2 x Weekly
        elif freq == 'BW':
            return '2W'
        # Biweekly End variants
        elif freq == 'BWEW':
            return '2W-WED'
        elif freq == 'BWEM':
            return '2W-MON'
        # Semiannual -> 2 x Quarterly
        elif freq == 'SA':
            return '2Q'
        # Logically unreachable but prevents mypy warning
        return freq
