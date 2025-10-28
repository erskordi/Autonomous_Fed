"""
Module for fitting and preparing data for the linear SVAR environment.
"""
from typing import Union, cast
from datetime import datetime
import pandas as pd
import statsmodels.api as sm #pragma: no cover
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import fedfred as fd
from linear_environment.helpers import LinearEnvironmentHelpers
from .objects import SVARResults

class LinearEnvironmentSolver:
    """
    Client for running models and simulations.

    Attributes:
        fred_key (str): API key for the FRED Database.
        start_date (str | datetime): Start date for the historical data.
        end_date (str | datetime): End date for the historical data.
        frequency (str): Frequency of the data ('Q' for quarterly).
        inflation_id (str): FRED series ID for inflation metric.
        real_output_id (str): FRED series ID for real output metric.
        potential_output_id (str): FRED series ID for potential output metric.
        interest_rate_id (str): FRED series ID for interest rate metric.
        historical_data (pd.DataFrame): Historical econometric dataset.
        forward_data (pd.DataFrame): Forward econometric dataset for forecasting.
        linear_svar (SVARResults): Fitted linear SVAR results.
        figure_five (Figure): Forecast figure five for the model.
        figure_six (Figure): Forecast figure six for the model.
        
    """
    def __init__(self,
                 fred_key: str,
                 start_date: Union[str, datetime] = "1987-07-01",
                 end_date: Union[str, datetime] = "2007-06-30",
                 frequency: str = "Q",
                 inflation_id: str = "GDPDEF",
                 real_output_id: str = "GDPC1",
                 potential_output_id: str = "GDPPOT",
                 interest_rate_id: str = "FEDFUNDS") -> None:
        """
        Initialize the LinearEnvironmentSolver.

        Args:
            fred_key (str): API key for the FRED Database.
            start_date (str | datetime): Start date for the historical data.
            end_date (str | datetime): End date for the historical data.
            frequency (str): Frequency of the data ('Q' for quarterly).
            inflation_id (str): FRED series ID for inflation metric. Default is "GDPDEF" ~ GDP Implicit Price Deflator.
            real_output_id (str): FRED series ID for real output metric. Default is "GDPC1" ~ Real GDP.
            potential_output_id (str): FRED series ID for potential output metric. Default is "GDPPOT" ~ Potential GDP.
            interest_rate_id (str): FRED series ID for interest rate metric. Default is "FEDFUNDS" ~ Effective Federal Funds Rate.
            fred_key (str): API key for the FRED Database.
        
        Returns:
            None

        Raises:
            ValueError: If any of the FRED series cannot be fetched.
        """
        self.__fred_key: str = fred_key
        self.fred: fd.FredAPI = fd.FredAPI(api_key=self.__fred_key, cache_mode=True, cache_size=256)
        self.start_date: Union[str, datetime] = start_date
        self.end_date: Union[str, datetime] = end_date
        self.frequency: str = frequency
        self.inflation_id: str = inflation_id
        self.real_output_id: str = real_output_id
        self.potential_output_id: str = potential_output_id
        self.interest_rate_id: str = interest_rate_id
        self.historical_data: pd.DataFrame = self.__create_econometric_dataset(self.start_date, self.end_date)
        self.forward_data: pd.DataFrame = self.__create_econometric_dataset(
            LinearEnvironmentHelpers.bump_date_by_one_period(self.end_date, self.frequency),
            datetime.today()
        )
        self.linear_svar: SVARResults = self.__fit_linear_svar(self.historical_data)
        self.forecast_data: pd.DataFrame = self.__forecast_output()
        self.figure_five: Figure = self.__create_forecast_figure_five()
        self.figure_six: Figure = self.__create_forecast_figure_six()
        self.squared_errors_data: pd.DataFrame = self.__create_squared_errors_data()
        self.figure_seven: Figure = self.__create_se_figure_seven()
        self.figure_eight: Figure = self.__create_se_figure_eight()
    # Dunder Methods

    # Private Methods
    def __fit_linear_svar(self, df: pd.DataFrame, alpha_drop: float = 0.10) -> SVARResults:
        """
        Fit the paper's restricted linear SVAR environment via recursive OLS.

        Args:
            df (pd.DataFrame): Quarterly data with a (PeriodIndex or DatetimeIndex) sorted in time and columns ['pi', 'y', 'i'] for inflation (π), output gap (y), and the federal funds rate (i).
            alpha_drop (float): Significance level for pruning lag-2 terms.

        Returns:
            SVARResults: Dataclass containing fitted models, design matrices, and residuals.

        Raises:
            AssertionError: If the required columns are missing.
        """
        # Check input
        assert {"pi", "y", "i"}.issubset(df.columns), "df must have columns: 'pi', 'y', 'i'"

        # Work copy with lags up to 2
        work = df.copy().sort_index()
        for c in ("pi", "y", "i"):
            work[f"{c}_lag1"] = work[c].shift(1)
            work[f"{c}_lag2"] = work[c].shift(2)

        # Output gap equation: y_t ~ y_{t-1}, pi_{t-1}, i_{t-1}, i_{t-2}
        y_target = work["y"]
        x_y = work[["y_lag1", "pi_lag1", "i_lag1", "i_lag2"]]
        x_y = sm.add_constant(x_y, has_constant="add")
        ymask = x_y.notna().all(axis=1) & y_target.notna()
        mod_y = sm.OLS(y_target[ymask], x_y[ymask]).fit()
        mod_y, x_y = LinearEnvironmentHelpers.drop_insignificant_lag_two(mod_y, x_y[ymask], y_target, alpha_drop)

        # Inflation equation: pi_t ~ y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1}
        pi_target = work["pi"]
        x_pi = work[["y", "y_lag1", "y_lag2", "pi_lag1", "pi_lag2", "i_lag1"]]
        x_pi = sm.add_constant(x_pi, has_constant="add")
        pimask = x_pi.notna().all(axis=1) & pi_target.notna()
        mod_pi = sm.OLS(pi_target[pimask], x_pi[pimask]).fit()
        mod_pi, x_pi = LinearEnvironmentHelpers.drop_insignificant_lag_two(mod_pi, x_pi[pimask], pi_target, alpha_drop)

        # Residuals (aligned to design matrices)
        resid_y = y_target.loc[x_y.index] - mod_y.predict(x_y)
        resid_y.name = "eps_y"
        resid_pi = pi_target.loc[x_pi.index] - mod_pi.predict(x_pi)
        resid_pi.name = "eps_pi"

        return SVARResults(
            mod_y=mod_y,
            mod_pi=mod_pi,
            x_y=x_y,
            x_pi=x_pi,
            resid_y=resid_y,
            resid_pi=resid_pi,
        )

    def __create_econometric_dataset(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Create a synthetic dataset for testing the linear SVAR fitting function.

        Args:
            start_date (str | datetime): Start date for the dataset.
            end_date (str | datetime): End date for the dataset.
        Returns:
            pd.DataFrame: Synthetic dataset with columns ['pi', 'y', 'i'] corresponding to inflation, ouput gap, and interest rate.
        """
        fred = fd.FredAPI(api_key=self.__fred_key, cache_mode=True)

        # Fetch data from FRED
        inflation = fred.get_series_observations(
            series_id=self.inflation_id,
            observation_start=start_date,
            observation_end=end_date,
            frequency=self.frequency.lower(),
            units="pc1" # YoY percent change
        )

        real_output = fred.get_series_observations(
            series_id=self.real_output_id,
            observation_start=start_date,
            observation_end=end_date,
            frequency=self.frequency.lower(),
            units="lin" # Linear units (level)
        )

        potential_output = fred.get_series_observations(
            series_id=self.potential_output_id,
            observation_start=start_date,
            observation_end=end_date,
            frequency=self.frequency.lower(),
            units="lin" # Linear units (level)
        )

        interest_rate = fred.get_series_observations(
            series_id=self.interest_rate_id,
            observation_start=start_date,
            observation_end=end_date,
            frequency=self.frequency.lower(),
            aggregation_method="avg" # Aggregate daily data to quarterly by averaging (specificly important for interest rates)
        )

        # Convert to Series with proper names
        pi = LinearEnvironmentHelpers.to_series(inflation, "pi")
        y_real = LinearEnvironmentHelpers.to_series(real_output, "y_real")
        y_pot = LinearEnvironmentHelpers.to_series(potential_output, "y_pot")
        i = LinearEnvironmentHelpers.to_series(interest_rate, "i")

        # Compute output gap
        y_gap = 100.0 * (y_real / y_pot - 1.0)
        y_gap.name = "y"

        # Merge
        df = pd.concat([pi, y_gap, i], axis=1).dropna()

        # Coerce pandas Incompatible Frequency Strings
        pandas_frequency = LinearEnvironmentHelpers.coerce_frequency_string(self.frequency)

        df.index = pd.PeriodIndex(df.index, freq=pandas_frequency) # Use coerced frequency for PeriodIndex

        return df

    def __forecast_output(self) -> pd.DataFrame:
        """
        Forecast future values using the fitted models.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame containing both historical and forecasted values for 'y_hat' and 'pi_hat'.
        
        Raises:
            None
        """
        # Pull coefficients directly from your fitted models
        coef_y  = self.linear_svar.mod_y.params.to_dict()
        coef_pi = self.linear_svar.mod_pi.params.to_dict()
        df_all = pd.concat([self.historical_data, self.forward_data]).sort_index()
        df_all["y_hat"]  = np.nan
        df_all["pi_hat"] = np.nan

        # start at a point where t-2 exists
        start = max(df_all.index.min() + 2, pd.Period(self.forward_data.index.min(), self.frequency.upper()))

        for t in df_all.loc[start:].index:
            # --- Output gap: y_t ---
            y_lag1  = LinearEnvironmentHelpers.get_recursive_lag(df_all, t, "y", 1)
            pi_lag1 = LinearEnvironmentHelpers.get_recursive_lag(df_all, t, "pi", 1)
            i_lag1  = df_all.at[t - 1, "i"]
            i_lag2  = df_all.at[t - 2, "i"]

            df_all.at[t, "y_hat"] = (
                coef_y["const"]
                + coef_y["y_lag1"]  * y_lag1
                + coef_y["pi_lag1"] * pi_lag1
                + coef_y["i_lag1"]  * i_lag1
                + coef_y["i_lag2"]  * i_lag2
            )

            # --- Inflation: pi_t (uses contemporaneous y_hat) ---
            y_t     = df_all.at[t, "y_hat"]
            y_lag1  = LinearEnvironmentHelpers.get_recursive_lag(df_all, t, "y", 1)
            y_lag2  = LinearEnvironmentHelpers.get_recursive_lag(df_all, t, "y", 2)
            pi_lag1 = LinearEnvironmentHelpers.get_recursive_lag(df_all, t, "pi", 1)
            pi_lag2 = LinearEnvironmentHelpers.get_recursive_lag(df_all, t, "pi", 2)
            i_lag1  = df_all.at[t - 1, "i"]

            df_all.at[t, "pi_hat"] = (
                coef_pi["const"]
                + coef_pi["y"]       * y_t
                + coef_pi["y_lag1"]  * y_lag1
                + coef_pi["y_lag2"]  * y_lag2
                + coef_pi["pi_lag1"] * pi_lag1
                + coef_pi["pi_lag2"] * pi_lag2
                + coef_pi["i_lag1"]  * i_lag1
            )
        
        return df_all

    def __create_forecast_figure_five(self) -> Figure:
        """
        Create forecast figure five for the model.

        Args:
            None
        
        Returns:
            fig: Matplotlib figure object for the output gap forecast.

        Raises:
            None
        """
        df_all = self.forecast_data
        sub = df_all.loc[self.forward_data.index.min():]

        index = cast(pd.PeriodIndex, sub.index)

        # Output gap
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(index.to_timestamp(), sub["y"], label="Output gap (actual)")
        ax.plot(index.to_timestamp(), sub["y_hat"], "--", label="Output gap (forecast)")
        ax.set_title("Output Gap — Actual vs Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percent")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def __create_forecast_figure_six(self) -> Figure:
        """
        Create forecast figures six for the model.

        Args:
            None
        
        Returns:
            fig: Matplotlib figure object for the output gap forecast.

        Raises:
            None
        """
        df_all = self.forecast_data
        sub = df_all.loc[self.forward_data.index.min():]

        index = cast(pd.PeriodIndex, sub.index)

        # Inflation
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(index.to_timestamp(), sub["pi"], label="Inflation YoY (actual)")
        ax.plot(index.to_timestamp(), sub["pi_hat"], "--", label="Inflation YoY (forecast)")
        ax.set_title("Inflation — Actual vs Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percent")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def __create_squared_errors_data(self):
        """
        Create a DataFrame to hold squared errors for the model.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame containing squared errors for output gap and inflation.

        Raises:
            None
        """
        # Training window (paper window)
        df_train = self.historical_data.loc[self.historical_data.index.min():self.historical_data.index.max()].copy()

        # Predict on the exact in-sample design matrices
        y_fit  = self.linear_svar.mod_y.predict(self.linear_svar.x_y)
        pi_fit = self.linear_svar.mod_pi.predict(self.linear_svar.x_pi)

        # Sanity: indexes should match design matrices you trained on
        assert y_fit.index.equals(self.linear_svar.x_y.index)
        assert pi_fit.index.equals(self.linear_svar.x_pi.index)

        # Build aligned frame on the training index
        fit_df = pd.DataFrame(index=df_train.index)
        fit_df["y_actual"]  = df_train["y"]
        fit_df["pi_actual"] = df_train["pi"]
        fit_df["y_fit"]     = y_fit.reindex(fit_df.index)
        fit_df["pi_fit"]    = pi_fit.reindex(fit_df.index)

        # Residuals (in-sample)
        resid_y  = (fit_df["y_actual"]  - fit_df["y_fit"])
        resid_pi = (fit_df["pi_actual"] - fit_df["pi_fit"])

        # Squared errors
        fit_df["se_y_svar"]  = resid_y**2
        fit_df["se_pi_svar"] = resid_pi**2

        # Drop rows without a fitted value (initial lags)
        fit_df = fit_df.dropna(subset=["y_fit","pi_fit"])

        return fit_df

    def __create_se_figure_seven(self) -> Figure:
        """
        Create squared error figure seven for the model.

        Args:
            None
        
        Returns:
            Figure: Matplotlib figure object for the in-sample fit.

        Raises:
            None
        """

        fit_df = self.__create_squared_errors_data()

        # Figure 4: Output gap fit — squared errors
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(fit_df.index.to_timestamp(), fit_df["se_y_svar"], color="red", label="SVAR", linewidth=1.5)
        ax.set_title(f"Figure 4 Replica: Output Gap Fit — Squared Errors ({fit_df.index.min()}-{fit_df.index.max()})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Squared error")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    def __create_se_figure_eight(self) -> Figure:
        """
        Create squared error figure eight for the model.

        Args:
            None

        Returns:
            Figure: Matplotlib figure object for the in-sample fit.

        Raises:
            None
        """

        fit_df = self.__create_squared_errors_data()

        # Figure 5: Inflation fit — squared errors
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(fit_df.index.to_timestamp(), fit_df["se_pi_svar"], color="red", label="SVAR", linewidth=1.5)
        ax.set_title(f"Figure 5 Replica: Inflation Fit — Squared Errors ({fit_df.index.min()}-{fit_df.index.max()})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Squared error")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig
