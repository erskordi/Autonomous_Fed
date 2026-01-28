"""
Module for fitting and preparing data for the environment.
"""

from typing import Union, Optional
from datetime import datetime
from IPython.display import Markdown, display
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import fedfred as fd
import torch
from .helpers import EnvironmentHelpers
from .objects import SVARResults

class EnvironmentSolver:
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
        figure_seven (Figure): Squared error figure seven for the model.
        figure_eight (Figure): Squared error figure eight for the model.

    Methods:
        create_econometric_dataset(start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
            Create econometric dataset from FRED data.
        fit_linear_svar(df: pd.DataFrame, alpha_drop: float = 0.10) -> SVARResults:
            Fit a linear SVAR model to the provided econometric dataset.
        train_one_run_lm(model: nn.Module, train_ds: TensorDataset, val_ds: TensorDataset, max_epochs: int=200, patience: int=6, mu_init: float=1e-3) -> Tuple[Optional[float], Optional[nn.Module]]:
            Train a neural network model using the Levenberg-Marquardt optimization algorithm with early stopping based on validation loss.
    """
    # Dunder Methods
    def __init__(self,
                 fred_key: str,
                 override_device: Optional[str] = None,
                 override_dtype: Optional[torch.dtype] = None,
                 start_date: Union[str, datetime] = "1987-07-01",
                 end_date: Union[str, datetime] = "2007-06-30",
                 frequency: str = "Q",
                 inflation_id: str = "GDPDEF",
                 real_output_id: str = "GDPC1",
                 potential_output_id: str = "GDPPOT",
                 interest_rate_id: str = "FEDFUNDS") -> None:
        """
        Initialize the EnvironmentSolver.

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
        torch.use_deterministic_algorithms(True)

        self.__fred_key: str = fred_key
        self.fred: fd.FredAPI = fd.FredAPI(api_key=self.__fred_key, cache_mode=True, cache_size=256)
        self.start_date: Union[str, datetime] = start_date
        self.end_date: Union[str, datetime] = end_date
        self.frequency: str = frequency
        self.inflation_id: str = inflation_id
        self.real_output_id: str = real_output_id
        self.potential_output_id: str = potential_output_id
        self.interest_rate_id: str = interest_rate_id
        self.historical_data: pd.DataFrame = self.create_econometric_dataset(self.start_date, self.end_date)
        self.forward_data: pd.DataFrame = self.create_econometric_dataset(
            EnvironmentHelpers.bump_date_by_one_period(self.end_date, self.frequency),
            datetime.today()
        )
        self.linear_svar: SVARResults = self.fit_linear_svar(self.historical_data)
        self.override_device: Optional[str] = override_device
        self.override_dtype: Optional[torch.dtype] = override_dtype
        self.device: torch.device = self.__device_settr()
        self.dtype: torch.dtype = self.__dtype_settr()

    # Private Methods
    def __device_settr(self) -> torch.device:
        """
        Determine the appropriate torch device to use (CUDA, MPS, or CPU).

        Args:
            None

        Returns:
            torch.device: The selected device for torch operations.

        Raises:
            None
        """
        device = None

        if self.override_device is not None:
            if self.override_device not in ["cpu", "cuda", "mps", None]:
                raise ValueError("override device if specified, must be one of 'cpu', 'cuda', or 'mps'")
            device = torch.device(self.override_device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        return device

    def __dtype_settr(self) -> torch.dtype:
        """
        Determine the appropriate torch dtype to use based on the device.

        Args:
            None

        Returns:
            torch.dtype: The selected dtype for torch operations.

        Raises:
            None

        Note:
            The dtype can be is based on device capabilities and is coerced through the override paramet
        """
        dtype = None

        if self.override_dtype is not None:
            dtype = self.override_dtype
        elif self.device.type == "cuda":
            dtype = torch.float64
        elif self.device.type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float64

        return dtype

    # Public Methods
    def create_econometric_dataset(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
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
        pi = EnvironmentHelpers.to_series(inflation, "pi")
        y_real = EnvironmentHelpers.to_series(real_output, "y_real")
        y_pot = EnvironmentHelpers.to_series(potential_output, "y_pot")
        i = EnvironmentHelpers.to_series(interest_rate, "i")

        # Compute output gap
        y_gap = 100.0 * (y_real / y_pot - 1.0)
        y_gap.name = "y"

        # Merge
        df = pd.concat([pi, y_gap, i], axis=1).dropna()

        # Coerce pandas Incompatible Frequency Strings
        pandas_frequency = EnvironmentHelpers.coerce_frequency_string(self.frequency)

        df.index = pd.PeriodIndex(df.index, freq=pandas_frequency) # Use coerced frequency for PeriodIndex

        return df

    def fit_linear_svar(self, df: pd.DataFrame, alpha_drop: float = 0.10) -> SVARResults:
        """
        Fit a linear SVAR model to the provided econometric dataset.

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
        mod_y, x_y = EnvironmentHelpers.drop_insignificant_lag_two(mod_y, x_y[ymask], y_target, alpha_drop)

        # Inflation equation: pi_t ~ y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1}
        pi_target = work["pi"]
        x_pi = work[["y", "y_lag1", "y_lag2", "pi_lag1", "pi_lag2", "i_lag1"]]
        x_pi = sm.add_constant(x_pi, has_constant="add")
        pimask = x_pi.notna().all(axis=1) & pi_target.notna()
        mod_pi = sm.OLS(pi_target[pimask], x_pi[pimask]).fit()
        mod_pi, x_pi = EnvironmentHelpers.drop_insignificant_lag_two(mod_pi, x_pi[pimask], pi_target, alpha_drop)

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

    def forecast_data(self) -> pd.DataFrame:
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
            y_lag1  = EnvironmentHelpers.get_recursive_lag(df_all, t, "y", 1)
            pi_lag1 = EnvironmentHelpers.get_recursive_lag(df_all, t, "pi", 1)
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
            y_lag1  = EnvironmentHelpers.get_recursive_lag(df_all, t, "y", 1)
            y_lag2  = EnvironmentHelpers.get_recursive_lag(df_all, t, "y", 2)
            pi_lag1 = EnvironmentHelpers.get_recursive_lag(df_all, t, "pi", 1)
            pi_lag2 = EnvironmentHelpers.get_recursive_lag(df_all, t, "pi", 2)
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

    # Properties
    @property
    def squared_errors_data(self) -> pd.DataFrame:
        """
        Create a DataFrame to hold squared errors for the model.
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
        resid_y  = fit_df["y_actual"]  - fit_df["y_fit"]
        resid_pi = fit_df["pi_actual"] - fit_df["pi_fit"]

        # Squared errors
        fit_df["se_y_svar"]  = resid_y**2
        fit_df["se_pi_svar"] = resid_pi**2

        # Drop rows without a fitted value (initial lags)
        fit_df = fit_df.dropna(subset=["y_fit","pi_fit"])

        return fit_df

    @property
    def figure_four(self) -> Figure:
        """Create squared error figure four for the model."""

        with plt.ioff():
            fit_df = self.squared_errors_data

            assert isinstance(fit_df.index, pd.PeriodIndex)
            # Figure 4: Output gap fit — squared errors
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(fit_df.index.to_timestamp(), fit_df["se_y_svar"], color="red", label="SVAR", linewidth=1.5)
            ax.set_title(f"Figure 4 Replica: Output Gap Fit — Squared Errors ({fit_df.index.min()}-{fit_df.index.max()})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Squared error")
            ax.grid(True, alpha=0.3)
            ax.legend()

            return fig

    @property
    def figure_five(self) -> Figure:
        """Create squared error figure five for the model."""

        with plt.ioff():
            fit_df = self.squared_errors_data

            assert isinstance(fit_df.index, pd.PeriodIndex)

            # Figure 5: Inflation fit — squared errors
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(fit_df.index.to_timestamp(), fit_df["se_pi_svar"], color="red", label="SVAR", linewidth=1.5)
            ax.set_title(f"Figure 5 Replica: Inflation Fit — Squared Errors ({fit_df.index.min()}-{fit_df.index.max()})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Squared error")
            ax.grid(True, alpha=0.3)
            ax.legend()

            return fig

    @property
    def table_nine(self) -> str:
        """Create comparison table nine for the model."""

        table_nine_str = fr"""
        | Parameters                       | Estimate                                                       | p-Value                                                    |
        |----------------------------------|----------------------------------------------------------------|------------------------------------------------------------|
        | **Output gap equation**          |                                                                |                                                            |
        | $C^y$                            | {self.linear_svar.mod_y.params['const']:.4f}                   | {self.linear_svar.mod_y.pvalues['const']:.4f}              |
        | $a^y_{{y,1}}$                    | {self.linear_svar.mod_y.params['y_lag1']:.4f}                  | {self.linear_svar.mod_y.pvalues['y_lag1']:.4f}             |
        | $a^y_{{\pi,1}}$                  | {self.linear_svar.mod_y.params['pi_lag1']:.4f}                 | {self.linear_svar.mod_y.pvalues['pi_lag1']:.4f}            |
        | $a^y_{{i,1}}$                    | {self.linear_svar.mod_y.params['i_lag1']:.4f}                  | {self.linear_svar.mod_y.pvalues['i_lag1']:.4f}             |
        | $a^y_{{i,2}}$                    | {self.linear_svar.mod_y.params['i_lag2']:.4f}                  | {self.linear_svar.mod_y.pvalues['i_lag2']:.4f}             |
        | $\bar{{R}}^2$                    | {self.linear_svar.mod_y.rsquared:.4f}                          |                                                            |
        | MSE                              | {self.linear_svar.mod_y.ssr/self.linear_svar.mod_y.nobs:.4f}   |                                                            |
        | $\hat\sigma_{{\varepsilon_1}}^2$ | {self.linear_svar.mod_y.mse_resid:.4f}                         |                                                            |
        | DW                               | {durbin_watson(self.linear_svar.mod_y.resid):.4f}              |                                                            |
        | LM(1)                            | {acorr_breusch_godfrey(self.linear_svar.mod_y, 1)[0]:.4f}      | {acorr_breusch_godfrey(self.linear_svar.mod_y, 1)[1]:.4f}  |
        | **Inflation equation**           |                                                                |                                                            |
        | $C^\pi$                          | {self.linear_svar.mod_pi.params['const']:.4f}                  | {self.linear_svar.mod_pi.pvalues['const']:.4f}             |
        | $a^\pi_{{y,0}}$                  | {self.linear_svar.mod_pi.params['y']:.4f}                      | {self.linear_svar.mod_pi.pvalues['y']:.4f}                 |
        | $a^\pi_{{y,1}}$                  | {self.linear_svar.mod_pi.params['y_lag1']:.4f}                 | {self.linear_svar.mod_pi.pvalues['y_lag1']:.4f}            |
        | $a^\pi_{{y,2}}$                  | {self.linear_svar.mod_pi.params['y_lag2']:.4f}                 | {self.linear_svar.mod_pi.pvalues['y_lag2']:.4f}            |
        | $a^\pi_{{\pi,1}}$                | {self.linear_svar.mod_pi.params['pi_lag1']:.4f}                | {self.linear_svar.mod_pi.pvalues['pi_lag1']:.4f}           |
        | $a^\pi_{{\pi,2}}$                | {self.linear_svar.mod_pi.params['pi_lag2']:.4f}                | {self.linear_svar.mod_pi.pvalues['pi_lag2']:.4f}           |
        | $a^\pi_{{i,1}}$                  | {self.linear_svar.mod_pi.params['i_lag1']:.4f}                 | {self.linear_svar.mod_pi.pvalues['i_lag1']:.4f}            |
        | $\bar{{R}}^2$                    | {self.linear_svar.mod_pi.rsquared:.4f}                         |                                                            |
        | MSE                              | {self.linear_svar.mod_pi.ssr/self.linear_svar.mod_pi.nobs:.4f} |                                                            |
        | $\hat\sigma_{{\varepsilon_2}}^2$ | {self.linear_svar.mod_pi.mse_resid:.4f}                        |                                                            |
        | DW                               | {durbin_watson(self.linear_svar.mod_pi.resid):.4f}             |                                                            |
        | LM(1)                            | {acorr_breusch_godfrey(self.linear_svar.mod_pi, 1)[0]:.4f}     | {acorr_breusch_godfrey(self.linear_svar.mod_pi, 1)[1]:.4f} |

        \* Residual std dev from OLS output."""

        return display(Markdown(table_nine_str))

    @property
    def table_nine_comparison(self) -> str:
        """Create comparison table nine for the model."""

        table_nine_str = fr"""
        | Parameters                       | Bundesbank Estimate | p-Value | Local OLS Estimate                                             | p-Value                                                    |
        |----------------------------------|---------------------|---------|----------------------------------------------------------------|------------------------------------------------------------|
        | **Output gap equation**          |                     |         |                                                                |                                                            |
        | $C^y$                            | 0.3834              | 0.0351  | {self.linear_svar.mod_y.params['const']:.4f}                   | {self.linear_svar.mod_y.pvalues['const']:.4f}              |
        | $a^y_{{y,1}}$                    | 0.9084              | 0.0000  | {self.linear_svar.mod_y.params['y_lag1']:.4f}                  | {self.linear_svar.mod_y.pvalues['y_lag1']:.4f}             |
        | $a^y_{{\pi,1}}$                  | -0.1437             | 0.1409  | {self.linear_svar.mod_y.params['pi_lag1']:.4f}                 | {self.linear_svar.mod_y.pvalues['pi_lag1']:.4f}            |
        | $a^y_{{i,1}}$                    | 0.2726              | 0.0661  | {self.linear_svar.mod_y.params['i_lag1']:.4f}                  | {self.linear_svar.mod_y.pvalues['i_lag1']:.4f}             |
        | $a^y_{{i,2}}$                    | -0.2896             | 0.0313  | {self.linear_svar.mod_y.params['i_lag2']:.4f}                  | {self.linear_svar.mod_y.pvalues['i_lag2']:.4f}             |
        | $\bar{{R}}^2$                    | 0.9100              |         | {self.linear_svar.mod_y.rsquared:.4f}                          |                                                            |
        | MSE                              | 0.2108              |         | {self.linear_svar.mod_y.ssr/self.linear_svar.mod_y.nobs:.4f}   |                                                            |
        | $\hat\sigma_{{\varepsilon_1}}^2$ | 0.2136              |         | {self.linear_svar.mod_y.mse_resid:.4f}                         |                                                            |
        | DW                               | 1.8206              |         | {durbin_watson(self.linear_svar.mod_y.resid):.4f}              |                                                            |
        | LM(1)                            | 3.1037              | 0.1644  | {acorr_breusch_godfrey(self.linear_svar.mod_y, 1)[0]:.4f}      | {acorr_breusch_godfrey(self.linear_svar.mod_y, 1)[1]:.4f}  |
        | **Inflation equation**           |                     |         |                                                                |                                                            |
        | $C^\pi$                          | 0.1035              | 0.1659  | {self.linear_svar.mod_pi.params['const']:.4f}                  | {self.linear_svar.mod_pi.pvalues['const']:.4f}             |
        | $a^\pi_{{y,0}}$                  | -0.0655             | 0.1578  | {self.linear_svar.mod_pi.params['y']:.4f}                      | {self.linear_svar.mod_pi.pvalues['y']:.4f}                 |
        | $a^\pi_{{y,1}}$                  | 0.1970              | 0.0048  | {self.linear_svar.mod_pi.params['y_lag1']:.4f}                 | {self.linear_svar.mod_pi.pvalues['y_lag1']:.4f}            |
        | $a^\pi_{{y,2}}$                  | -0.1121             | 0.0163  | {self.linear_svar.mod_pi.params['y_lag2']:.4f}                 | {self.linear_svar.mod_pi.pvalues['y_lag2']:.4f}            |
        | $a^\pi_{{\pi,1}}$                | 1.2970              | 0.0000  | {self.linear_svar.mod_pi.params['pi_lag1']:.4f}                | {self.linear_svar.mod_pi.pvalues['pi_lag1']:.4f}           |
        | $a^\pi_{{\pi,2}}$                | -0.3116             | 0.0076  | {self.linear_svar.mod_pi.params['pi_lag2']:.4f}                | {self.linear_svar.mod_pi.pvalues['pi_lag2']:.4f}           |
        | $a^\pi_{{i,1}}$                  | -0.0122             | 0.4174  | {self.linear_svar.mod_pi.params['i_lag1']:.4f}                 | {self.linear_svar.mod_pi.pvalues['i_lag1']:.4f}            |
        | $\bar{{R}}^2$                    | 0.9450              |         | {self.linear_svar.mod_pi.rsquared:.4f}                         |                                                            |
        | MSE                              | 0.0326              |         | {self.linear_svar.mod_pi.ssr/self.linear_svar.mod_pi.nobs:.4f} |                                                            |
        | $\hat\sigma_{{\varepsilon_2}}^2$ | 0.0330              |         | {self.linear_svar.mod_pi.mse_resid:.4f}                        |                                                            |
        | DW                               | 2.1095              |         | {durbin_watson(self.linear_svar.mod_pi.resid):.4f}             |                                                            |
        | LM(1)                            | 2.5542              | 0.0696  | {acorr_breusch_godfrey(self.linear_svar.mod_pi, 1)[0]:.4f}     | {acorr_breusch_godfrey(self.linear_svar.mod_pi, 1)[1]:.4f} |

        \* Residual std dev from OLS output."""

        return display(Markdown(table_nine_str))
