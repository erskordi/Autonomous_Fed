"""
Module for fitting and preparing data for the environment.
"""

from typing import Union, cast, Optional, Tuple, Dict, Any
from datetime import datetime
import random
import pandas as pd
import statsmodels.api as sm #pragma: no cover
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import fedfred as fd
import torch
from torch import nn
from torch.utils.data import TensorDataset
from .helpers import EnvironmentHelpers
from .objects import SVARResults
from .optimizers import LevenbergMarquardt
from .objects import MapMinMax
from .networks import SingleHiddenLayerNet

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
        self.device: torch.device = self.__device_settr()
        self.dtype: torch.dtype = self.__dtype_settr()
        self.override_device: Optional[str] = override_device
        self.override_dtype: Optional[torch.dtype] = override_dtype

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
            if self.override_device not in ["cpu", "cuda", "mps"]:
                raise ValueError("override must be one of 'cpu', 'cuda', or 'mps'")
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

    def make_lag_matrix(self,
                        df: pd.DataFrame,
                        target_col: str,
                        holdout_frac: float=0.15,
                        feature_range: Tuple[float, float]=(-1.0, 1.0)) -> Tuple[TensorDataset, TensorDataset, Dict[str, Any]]:
        """
        Constructs lagged input-output data matrices for time series modeling.

        Args:
            df (pd.DataFrame): DataFrame containing the time series data.
            target_col (str): Name of the target column to predict. Must be either 'y' or 'pi'.
            dtype (torch.dtype): Desired data type for the tensors.
            holdout_frac (float, optional): Fraction of data to hold out for validation. Default is 0.15.
            feature_range (Tuple[float, float], optional): Desired range of transformed features. Default is (-1.0, 1.0).

        Returns:
            Tuple[TensorDataset, TensorDataset, Dict[str, Any]]: Training and validation datasets as TensorDataset objects.

        Raises:
            ValueError: If target_col is not 'y' or 'pi', or if required columns are missing.
        """
        required = {"y", "pi", "i"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        data = df.copy().sort_index()

        # Create needed lags
        data["y_lag1"]  = data["y"].shift(1)
        data["y_lag2"]  = data["y"].shift(2)
        data["pi_lag1"] = data["pi"].shift(1)
        data["pi_lag2"] = data["pi"].shift(2)
        data["i_lag1"]  = data["i"].shift(1)
        data["i_lag2"]  = data["i"].shift(2)

        if target_col == "y":
            feature_cols = [
                "y_lag1",
                "pi_lag1",
                "i_lag1",
                "i_lag2"
            ]
            y_series = data["y"]
        elif target_col == "pi":
            feature_cols = [
                "y",
                "y_lag1",
                "y_lag2",
                "pi_lag1",
                "pi_lag2",
                "i_lag1"
            ]
            y_series = data["pi"]
        else:
            raise ValueError("target_col must be 'y' or 'pi' for enforced paper design.")

        x_df = data[feature_cols]
        mask = x_df.notna().all(axis=1) & y_series.notna()
        x = x_df[mask].values.astype(np.float64)
        y = y_series[mask].values.astype(np.float64).reshape(-1, 1)

        # deterministic “last 15%” validation split
        n_total = len(x)
        n_hold  = int(np.floor(holdout_frac * n_total))
        n_train = n_total - n_hold
        xtr_raw, xva_raw = x[:n_train], x[n_train:]
        ytr_raw, yva_raw = y[:n_train], y[n_train:]

        # fit scalers on TRAIN ONLY
        x_scaler = MapMinMax(out_lo=feature_range[0], out_hi=feature_range[1]).fit(xtr_raw)
        y_scaler = MapMinMax(out_lo=feature_range[0], out_hi=feature_range[1]).fit(ytr_raw)

        xtr = x_scaler.transform(xtr_raw).astype(np.float64)
        xva = x_scaler.transform(xva_raw).astype(np.float64)
        ytr = y_scaler.transform(ytr_raw).astype(np.float64)
        yva = y_scaler.transform(yva_raw).astype(np.float64)

        train_ds = TensorDataset(torch.from_numpy(xtr).to(self.dtype), torch.from_numpy(ytr).to(self.dtype))
        val_ds   = TensorDataset(torch.from_numpy(xva).to(self.dtype), torch.from_numpy(yva).to(self.dtype))

        meta = {
            "feature_cols": feature_cols,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "xtr_raw": xtr_raw,
            "xva_raw": xva_raw,
            "ytr_raw": ytr_raw,
            "yva_raw": yva_raw
        }
        return train_ds, val_ds, meta

    def train_one_run_lm(self,
                         model: nn.Module,
                         train_ds: TensorDataset,
                         val_ds: TensorDataset,
                         max_epochs: int=200,
                         patience: int=6,
                         mu_init: float=1e-3) -> Tuple[Optional[float], Optional[nn.Module]]:
        """
        Trains a neural network model using the Levenberg-Marquardt optimization algorithm with early stopping based on validation loss.

        Args:
            model (nn.Module): Neural network model to train.
            train_ds (TensorDataset): Training dataset.
            val_ds (TensorDataset): Validation dataset.
            max_epochs (int, optional): Maximum number of training epochs. Default is 200.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is 25.
            mu_init (float, optional): Initial damping factor for LM optimizer. Default is 1e-3.

        Returns:
            Tuple[float, nn.Module]: Best validation MSE and the trained model.

        Raises:
            None
        """
        xtr, ytr = train_ds.tensors
        xva, yva = val_ds.tensors
        model = model.to(self.device, dtype=self.dtype)

        # Levenberg–Marquardt optimizer instance
        opt = LevenbergMarquardt(model.parameters(), mu=mu_init)
        loss_fn = torch.nn.MSELoss()

        best = {"val": float("inf"), "state": None, "epoch": -1}
        bad_epochs = 0

        # LM closure returning residual vector (NOT scalar loss)
        def closure_train():
            model.zero_grad(set_to_none=True)
            pred = model(xtr)
            r = (pred - ytr).reshape(-1)
            return r

        for ep in range(max_epochs):
            # One LM parameter update
            info = opt.step(closure_train)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_pred = model(xva)
                val_mse = loss_fn(val_pred, yva).item()

            # Early stopping logic
            if val_mse < best["val"] - 1e-8: # type: ignore[operator]
                best.update(state={k: v.clone() for k, v in model.state_dict().items()}, val=val_mse, epoch=ep) # type: ignore[call-overload]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

            # Optional monitoring
            if (ep+1) % 10 == 0:
                print(f"Epoch {ep+1:03d}: val={val_mse:.6f}, μ={info['mu']:.2e}, accepted={info['accepted']}")

        # Restore best parameters
        if best["state"] is not None:
            model.load_state_dict(best["state"]) # type: ignore[arg-type]

        return best["val"], model

    def search_hidden_units_lm(self,
                               df: pd.DataFrame,
                               target_col: str,
                               hidden_grid: range = range(1, 11),
                               seeds: int=30,
                               holdout_frac: float=0.15) -> Dict[str, Union[float, int, nn.Module]]:
        """
        Searches for the optimal number of hidden units in a single hidden layer neural network using the Levenberg-Marquardt optimization algorithm.

        Args:
            df (pd.DataFrame): DataFrame containing the time series data.
            target_col (str): Name of the target column to predict. Must be either 'y' or 'pi'.
            hidden_grid (range, optional): Range of hidden units to search over. Default is range(1, 11).
            seeds (int, optional): Number of random initializations per hidden unit setting. Default is 30.
            holdout_frac (float, optional): Fraction of data to hold out for validation. Default is 0.15.

        Returns:
            Dict[str, Union[float, int, nn.Module]]: Dictionary containing the best overall
                - "metric": Best validation MSE (float).
                - "h": Number of hidden units for the best model (int).
                - "model": The best trained model (nn.Module).
                - "meta": Metadata dictionary from the best run (Dict[str, Any]).

        Raises:
            ValueError: If target_col is not 'y' or 'pi', or if required columns are missing.
        """
        overall = {"metric": float("inf")}
        per_h_stats = []
        for h in hidden_grid:
            mses, runs = [], []

            for s in range(seeds):
                seed = 10_000 + 97*h + s
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                train_ds, val_ds, meta = self.make_lag_matrix(
                    df, target_col=target_col, holdout_frac=holdout_frac
                )
                n_in = train_ds.tensors[0].shape[1]

                model = SingleHiddenLayerNet(n_in=n_in, n_hidden=h, n_out=1).to(dtype=self.dtype)

                val_mse, fitted = self.train_one_run_lm(model, train_ds, val_ds)
                mses.append(val_mse)
                runs.append((val_mse, fitted, meta))
            mean_mse = float(np.mean(mses)) # type: ignore[arg-type]
            per_h_stats.append((h, mean_mse, mses))
            if mean_mse < overall["metric"]:
                # pick the single best run *within this h* to keep
                best_run = min(runs, key=lambda t: t[0]) # type: ignore[arg-type, return-value]
                overall = {
                    "metric": mean_mse,
                    "h": h,
                    "model": best_run[1], # type: ignore[dict-item]
                    "meta": best_run[2] # type: ignore[dict-item]
                }
        return overall # type: ignore[return-value]

    @torch.no_grad()
    def predict_inverse(self, model: nn.Module, x_raw: np.ndarray, x_scaler: MapMinMax, y_scaler: MapMinMax,) -> np.ndarray:
        """
        Predict using the model and inverse transform the output to the original scale.

        Args:
            model (nn.Module): Trained PyTorch model.
            x_raw (np.ndarray): Raw input features.
            x_scaler (MapMinMax): Scaler for input features.
            y_scaler (MapMinMax): Scaler for output.

        Returns:
            np.ndarray: Predicted values in the original scale.

        Raises:
            None
        """
        xs = x_scaler.transform(x_raw).astype(np.float64)
        x_tensor = torch.from_numpy(xs).to(dtype=self.dtype, device=self.device)
        yhat_s = model.to(dtype=self.dtype, device=self.device)(x_tensor)
        yhat_s_cpu = yhat_s.detach().to("cpu").numpy()
        yhat = y_scaler.inverse_transform(yhat_s_cpu)
        return yhat
    # Properties
    @property
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

    @property
    def figure_five(self) -> Figure:
        """
        Create forecast figure five for the model.

        Args:
            None

        Returns:
            fig: Matplotlib figure object for the output gap forecast.

        Raises:
            None
        """
        with plt.ioff():
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

    @property
    def figure_six(self) -> Figure:
        """
        Create forecast figures six for the model.

        Args:
            None

        Returns:
            fig: Matplotlib figure object for the output gap forecast.

        Raises:
            None
        """
        with plt.ioff():
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

    @property
    def squared_errors_data(self) -> pd.DataFrame:
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
        resid_y  = fit_df["y_actual"]  - fit_df["y_fit"]
        resid_pi = fit_df["pi_actual"] - fit_df["pi_fit"]

        # Squared errors
        fit_df["se_y_svar"]  = resid_y**2
        fit_df["se_pi_svar"] = resid_pi**2

        # Drop rows without a fitted value (initial lags)
        fit_df = fit_df.dropna(subset=["y_fit","pi_fit"])

        return fit_df

    @property
    def figure_seven(self) -> Figure:
        """
        Create squared error figure seven for the model.

        Args:
            None

        Returns:
            Figure: Matplotlib figure object for the in-sample fit.

        Raises:
            None
        """
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
    def figure_eight(self) -> Figure:
        """
        Create squared error figure eight for the model.

        Args:
            None

        Returns:
            Figure: Matplotlib figure object for the in-sample fit.

        Raises:
            None
        """
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
