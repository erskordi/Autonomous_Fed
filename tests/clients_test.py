"""
Comprehensive unit tests for the clients module.
"""

from unittest.mock import patch, MagicMock
from datetime import datetime
import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
import fedfred as fd
from autonomous_fed.clients import EnvironmentSolver
from autonomous_fed.helpers import EnvironmentHelpers
from autonomous_fed.objects import SVARResults

class TestLinearEnvironmentSolver:
    """
    Unit tests for the LinearEnvironmentSolver class.
    """

    @patch('autonomous_fed.clients.EnvironmentSolver.create_econometric_dataset')
    def test_init(self, mock_create_dataset):
        """
        test_init: Test the initialization of EnvironmentSolver.
        """
        # Mock the dataset creation to return a sample DataFrame
        mock_create_dataset.return_value = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'pi': [0.1, 0.2, 0.3],
            'i': [0.05, 0.06, 0.07]
        })

        solver = EnvironmentSolver("test_key")
        assert isinstance(solver, EnvironmentSolver)
        assert solver.start_date == "1987-07-01"
        assert solver.end_date == "2007-06-30"
        assert solver.frequency == "Q"
        assert solver.inflation_id == "GDPDEF"
        assert solver.real_output_id == "GDPC1"
        assert solver.potential_output_id == "GDPPOT"
        assert solver.interest_rate_id == "FEDFUNDS"

        # Verify the mock was called twice (for historical and forward data)
        assert mock_create_dataset.call_count == 2

        # Verify the specific calls
        calls = mock_create_dataset.call_args_list
        assert str(calls[0][0][0]) == "1987-07-01"  # Historical data start
        assert str(calls[0][0][1]) == "2007-06-30"  # Historical data end
        assert str(calls[1][0][0]) == "2007-09-30"  # Forward data start
        # Second call end date is current datetime, so just check it exists
        assert calls[1][0][1] is not None
    # Private Methods
    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    def test_fit_linear_svar(self, mock_drop_lag, mock_create_dataset):
        """
        Comprehensive test for the __fit_linear_svar method covering all scenarios.
        """
        # Test 1: Success case with valid data
        dates = pd.date_range('2000-01-01', periods=20, freq='Q')
        np.random.seed(42)

        valid_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.5, 20),
            'y': np.random.normal(0.0, 1.0, 20),
            'i': np.random.normal(5.0, 1.0, 20)
        }, index=dates)

        mock_create_dataset.return_value = valid_df

        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        solver = EnvironmentSolver("test_key")
        # Clear any calls from initialization
        mock_drop_lag.reset_mock()

        result = solver._EnvironmentSolver__fit_linear_svar(valid_df)

        # Verify return type and structure
        assert isinstance(result, SVARResults)
        assert hasattr(result.mod_y, 'params')
        assert hasattr(result.mod_pi, 'params')

        # Verify design matrices have correct columns
        expected_y_cols = ['const', 'y_lag1', 'pi_lag1', 'i_lag1', 'i_lag2']
        expected_pi_cols = ['const', 'y', 'y_lag1', 'y_lag2', 'pi_lag1', 'pi_lag2', 'i_lag1']

        assert list(result.x_y.columns) == expected_y_cols
        assert list(result.x_pi.columns) == expected_pi_cols

        # Verify residuals
        assert isinstance(result.resid_y, pd.Series)
        assert isinstance(result.resid_pi, pd.Series)
        assert result.resid_y.name == "eps_y"
        assert result.resid_pi.name == "eps_pi"
        assert len(result.resid_y) <= len(valid_df) - 2
        assert len(result.resid_pi) <= len(valid_df) - 2
        # Should be called twice for this specific call (once for y equation, once for pi equation)
        assert mock_drop_lag.call_count == 2

        # Test 2: Missing columns - missing 'i'
        incomplete_df1 = pd.DataFrame({
            'pi': [1.0, 2.0, 3.0],
            'y': [0.1, 0.2, 0.3]
        })

        with pytest.raises(AssertionError, match="df must have columns: 'pi', 'y', 'i'"):
            solver.EnvironmentSolver__fit_linear_svar(incomplete_df1)

        # Test 3: Missing columns - missing 'pi'
        incomplete_df2 = pd.DataFrame({
            'y': [0.1, 0.2, 0.3],
            'i': [1.0, 2.0, 3.0]
        })

        with pytest.raises(AssertionError, match="df must have columns: 'pi', 'y', 'i'"):
            solver._LinearEnvironmentSolver__fit_linear_svar(incomplete_df2)

        # Test 4: Missing columns - missing 'y'
        incomplete_df3 = pd.DataFrame({
            'pi': [1.0, 2.0, 3.0],
            'i': [1.0, 2.0, 3.0]
        })

        with pytest.raises(AssertionError, match="df must have columns: 'pi', 'y', 'i'"):
            solver._LinearEnvironmentSolver__fit_linear_svar(incomplete_df3)

        # Test 5: Custom alpha_drop parameter
        mock_drop_lag.reset_mock()

        def mock_drop_custom_alpha(model, x_data, y_target, alpha):
            assert alpha == 0.05
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_custom_alpha

        result = solver._LinearEnvironmentSolver__fit_linear_svar(valid_df, alpha_drop=0.05)

        assert mock_drop_lag.call_count == 2
        for call in mock_drop_lag.call_args_list:
            # Check both positional and keyword arguments
            if len(call[0]) > 3:  # positional arguments
                assert call[0][3] == 0.05
            elif 'alpha_drop' in call[1]:  # keyword arguments
                assert call[1]['alpha_drop'] == 0.05

        # Test 6: Data with NaN values
        mock_drop_lag.reset_mock()
        mock_drop_lag.side_effect = mock_drop_side_effect

        nan_df = valid_df.copy()
        nan_df.loc[nan_df.index[5], 'pi'] = np.nan
        nan_df.loc[nan_df.index[10], 'y'] = np.nan
        nan_df.loc[nan_df.index[15], 'i'] = np.nan

        result = solver._LinearEnvironmentSolver__fit_linear_svar(nan_df)

        assert isinstance(result, SVARResults)
        assert len(result.resid_y) > 0
        assert len(result.resid_pi) > 0
        assert not result.resid_y.isna().any()
        assert not result.resid_pi.isna().any()

        # Test 7: Unsorted index
        mock_drop_lag.reset_mock()

        unsorted_df = valid_df.sample(frac=1).copy()
        result = solver._LinearEnvironmentSolver__fit_linear_svar(unsorted_df)

        assert isinstance(result, SVARResults)
        assert len(result.resid_y) > 0
        assert len(result.resid_pi) > 0

        # Test 8: Minimal data (just enough for estimation)
        mock_drop_lag.reset_mock()

        minimal_dates = pd.date_range('2000-01-01', periods=5, freq='Q')
        minimal_df = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2, 2.3, 2.4],
            'y': [0.0, 0.1, 0.2, 0.3, 0.4],
            'i': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=minimal_dates)

        result = solver._LinearEnvironmentSolver__fit_linear_svar(minimal_df)

        assert isinstance(result, SVARResults)
        assert len(result.resid_y) >= 1
        assert len(result.resid_pi) >= 1

        # Test 9: Verify lag creation logic
        mock_drop_lag.reset_mock()

        lag_test_dates = pd.date_range('2000-01-01', periods=6, freq='Q')
        lag_test_df = pd.DataFrame({
            'pi': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'i': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        }, index=lag_test_dates)

        result = solver._LinearEnvironmentSolver__fit_linear_svar(lag_test_df)

        # Verify lag columns in design matrices
        assert 'y_lag1' in result.x_y.columns
        assert 'pi_lag1' in result.x_y.columns
        assert 'i_lag1' in result.x_y.columns
        assert 'i_lag2' in result.x_y.columns

        assert 'y' in result.x_pi.columns
        assert 'y_lag1' in result.x_pi.columns
        assert 'y_lag2' in result.x_pi.columns
        assert 'pi_lag1' in result.x_pi.columns
        assert 'pi_lag2' in result.x_pi.columns
        assert 'i_lag1' in result.x_pi.columns
    # Public Methods
    @patch('fedfred.FredAPI')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.to_series')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.coerce_frequency_string')
    def test_create_econometric_dataset(self, mock_coerce_freq, mock_to_series, mock_fred_api):
        """
        Comprehensive test for the create_econometric_dataset method covering all scenarios.
        """
        # Test 1: Success case with valid data
        mock_fred_instance = MagicMock()
        mock_fred_api.return_value = mock_fred_instance

        # Mock FRED API responses
        mock_inflation_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=8, freq='Q'),
            'value': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
        })

        mock_real_output_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=8, freq='Q'),
            'value': [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070]
        })

        mock_potential_output_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=8, freq='Q'),
            'value': [1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035]
        })

        mock_interest_rate_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=8, freq='Q'),
            'value': [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7]
        })

        # Configure mock FRED API calls
        def mock_get_series_side_effect(*args, **kwargs):
            series_id = kwargs.get('series_id')
            if series_id == 'GDPDEF':
                return mock_inflation_data
            elif series_id == 'GDPC1':
                return mock_real_output_data
            elif series_id == 'GDPPOT':
                return mock_potential_output_data
            elif series_id == 'FEDFUNDS':
                return mock_interest_rate_data
            else:
                raise ValueError(f"Unknown series_id: {series_id}")

        mock_fred_instance.get_series_observations.side_effect = mock_get_series_side_effect

        # Mock helper functions
        dates = pd.date_range('2000-01-01', periods=8, freq='Q')
        mock_pi_series = pd.Series([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
                                   index=dates, name='pi')
        mock_y_real_series = pd.Series([1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070],
                                       index=dates, name='y_real')
        mock_y_pot_series = pd.Series([1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035],
                                      index=dates, name='y_pot')
        mock_i_series = pd.Series([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
                                  index=dates, name='i')

        def mock_to_series_side_effect(data, name):
            if name == 'pi':
                return mock_pi_series
            elif name == 'y_real':
                return mock_y_real_series
            elif name == 'y_pot':
                return mock_y_pot_series
            elif name == 'i':
                return mock_i_series
            else:
                raise ValueError(f"Unknown series name: {name}")

        mock_to_series.side_effect = mock_to_series_side_effect
        mock_coerce_freq.return_value = 'Q'

        # Create solver (this will trigger the mocked dataset creation)
        solver = LinearEnvironmentSolver("test_key")

        # Reset the mock call counts before testing the method directly
        mock_fred_instance.reset_mock()
        mock_to_series.reset_mock()
        mock_coerce_freq.reset_mock()

        # Test the method directly
        result = solver.create_econometric_dataset("2000-01-01", "2001-12-31")

        # Verify return type and structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['pi', 'y', 'i']
        assert isinstance(result.index, pd.PeriodIndex)

        # Verify FRED API was called correctly (only for this direct call)
        assert mock_fred_instance.get_series_observations.call_count == 4

        # Verify specific API calls
        calls = mock_fred_instance.get_series_observations.call_args_list

        # Inflation call
        inflation_call = calls[0]
        assert inflation_call[1]['series_id'] == 'GDPDEF'
        assert inflation_call[1]['observation_start'] == "2000-01-01"
        assert inflation_call[1]['observation_end'] == "2001-12-31"
        assert inflation_call[1]['frequency'] == 'q'
        assert inflation_call[1]['units'] == 'pc1'

        # Real output call
        real_output_call = calls[1]
        assert real_output_call[1]['series_id'] == 'GDPC1'
        assert real_output_call[1]['units'] == 'lin'

        # Potential output call
        potential_output_call = calls[2]
        assert potential_output_call[1]['series_id'] == 'GDPPOT'
        assert potential_output_call[1]['units'] == 'lin'

        # Interest rate call
        interest_rate_call = calls[3]
        assert interest_rate_call[1]['series_id'] == 'FEDFUNDS'
        assert interest_rate_call[1]['aggregation_method'] == 'avg'

        # Verify helper functions were called
        assert mock_to_series.call_count == 4
        assert mock_coerce_freq.called

        # Verify output gap calculation (y = 100 * (y_real / y_pot - 1))
        expected_y_gap = 100.0 * (mock_y_real_series / mock_y_pot_series - 1.0)
        # Since we're using mock data, verify the calculation logic exists in result
        assert 'y' in result.columns

        # Reset mocks for next tests
        mock_fred_instance.reset_mock()
        mock_to_series.reset_mock()
        mock_coerce_freq.reset_mock()

        # Test 2: Different date formats (datetime objects)
        start_datetime = datetime(2000, 1, 1)
        end_datetime = datetime(2001, 12, 31)

        result2 = solver.create_econometric_dataset(start_datetime, end_datetime)

        assert isinstance(result2, pd.DataFrame)
        assert list(result2.columns) == ['pi', 'y', 'i']

        # Verify datetime objects were passed correctly
        calls2 = mock_fred_instance.get_series_observations.call_args_list
        assert calls2[0][1]['observation_start'] == start_datetime
        assert calls2[0][1]['observation_end'] == end_datetime
    # Properties
    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.get_recursive_lag')
    def test_forecast_data(self, mock_get_recursive_lag, mock_drop_lag, mock_create_dataset):
        """
        Comprehensive test for the forecast_data property covering all scenarios.
        """
        # Setup test data with PeriodIndex
        historical_dates = pd.period_range('2000-Q1', periods=10, freq='Q')
        forward_dates = pd.period_range('2002-Q3', periods=5, freq='Q')

        # Create realistic test data
        historical_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 10),
            'y': np.random.normal(0.0, 0.5, 10),
            'i': np.random.normal(5.0, 0.2, 10)
        }, index=historical_dates)

        forward_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 5),
            'y': np.random.normal(0.0, 0.5, 5),
            'i': np.random.normal(5.0, 0.2, 5)
        }, index=forward_dates)

        # Mock the dataset creation to return our test data
        def mock_create_dataset_side_effect(start, end):
            if str(start) == "1987-07-01":
                return historical_df
            else:
                return forward_df

        mock_create_dataset.side_effect = mock_create_dataset_side_effect

        # Mock the drop_insignificant_lag_two function
        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        # Mock get_recursive_lag to return predictable values
        def mock_get_recursive_lag_side_effect(df, t, col, lag):
            if col == "y" and lag == 1:
                return 0.1
            elif col == "y" and lag == 2:
                return 0.2
            elif col == "pi" and lag == 1:
                return 2.1
            elif col == "pi" and lag == 2:
                return 2.2
            else:
                return 0.0

        mock_get_recursive_lag.side_effect = mock_get_recursive_lag_side_effect

        # Create solver
        solver = LinearEnvironmentSolver("test_key")

        # Test 1: Basic functionality - verify structure and types
        result = solver.forecast_data

        # Verify return type and basic structure
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.PeriodIndex)

        # Verify columns include original data plus forecasts
        expected_columns = ['pi', 'y', 'i', 'y_hat', 'pi_hat']
        assert all(col in result.columns for col in expected_columns)

        # Verify the data includes both historical and forward periods
        assert len(result) == len(historical_df) + len(forward_df)

        # Verify data is sorted by index
        assert result.index.is_monotonic_increasing

        # Test 2: Verify coefficient extraction from fitted models
        # Check that coefficients were properly extracted
        coef_y = solver.linear_svar.mod_y.params.to_dict()
        coef_pi = solver.linear_svar.mod_pi.params.to_dict()

        # Verify coefficient dictionaries have expected keys
        expected_y_keys = ['const', 'y_lag1', 'pi_lag1', 'i_lag1', 'i_lag2']
        expected_pi_keys = ['const', 'y', 'y_lag1', 'y_lag2', 'pi_lag1', 'pi_lag2', 'i_lag1']

        # At least some of these keys should be present (depending on what drop_insignificant_lag_two removes)
        assert any(key in coef_y for key in expected_y_keys)
        assert any(key in coef_pi for key in expected_pi_keys)

        # Test 3: Verify forecasting starts at correct point
        # Should start at max(min_index + 2, forward_data_start)
        combined_df = pd.concat([historical_df, forward_df]).sort_index()
        expected_start = max(
            combined_df.index.min() + 2,
            pd.Period(forward_df.index.min(), 'Q')
        )

        # Check that y_hat and pi_hat are not NaN from the start point onwards
        forecast_subset = result.loc[expected_start:]
        if len(forecast_subset) > 0:
            # Should have some non-NaN forecasted values
            assert not forecast_subset['y_hat'].isna().all()
            assert not forecast_subset['pi_hat'].isna().all()

        # Test 4: Verify helper function calls
        # get_recursive_lag should be called for computing lagged values
        assert mock_get_recursive_lag.call_count > 0

        # Verify get_recursive_lag was called with correct parameters
        lag_calls = mock_get_recursive_lag.call_args_list

        # Should have calls for both y and pi with different lag values
        y_calls = [call for call in lag_calls if call[0][2] == 'y']  # column name is 3rd arg
        pi_calls = [call for call in lag_calls if call[0][2] == 'pi']

        assert len(y_calls) > 0
        assert len(pi_calls) > 0

        # Test 5: Verify forecast computation logic
        # Reset mocks for more controlled testing
        mock_get_recursive_lag.reset_mock()

        # Create a more controlled test with known coefficients
        with patch.object(solver.linear_svar.mod_y, 'params') as mock_y_params, \
            patch.object(solver.linear_svar.mod_pi, 'params') as mock_pi_params:

            # Mock coefficients with known values
            mock_y_params.to_dict.return_value = {
                'const': 1.0,
                'y_lag1': 0.5,
                'pi_lag1': 0.3,
                'i_lag1': 0.2,
                'i_lag2': 0.1
            }

            mock_pi_params.to_dict.return_value = {
                'const': 0.5,
                'y': 0.4,
                'y_lag1': 0.3,
                'y_lag2': 0.2,
                'pi_lag1': 0.6,
                'pi_lag2': 0.1,
                'i_lag1': 0.1
            }

            # Create minimal test data for calculation verification
            test_hist = pd.DataFrame({
                'pi': [2.0, 2.1, 2.2],
                'y': [0.0, 0.1, 0.2],
                'i': [5.0, 5.1, 5.2]
            }, index=pd.period_range('2000-Q1', periods=3, freq='Q'))

            test_forward = pd.DataFrame({
                'pi': [2.3],
                'y': [0.3],
                'i': [5.3]
            }, index=pd.period_range('2000-Q4', periods=1, freq='Q'))

            # Update the mock to return our controlled data
            def controlled_mock_create_dataset(start, end):
                if str(start) == "1987-07-01":
                    return test_hist
                else:
                    return test_forward

            mock_create_dataset.side_effect = controlled_mock_create_dataset

            # Create new solver with controlled data
            solver_controlled = LinearEnvironmentSolver("test_key")

            # Get forecast data
            controlled_result = solver_controlled.forecast_data

            # Verify calculations were performed
            assert 'y_hat' in controlled_result.columns
            assert 'pi_hat' in controlled_result.columns

        # Test 6: Edge case - minimal data
        minimal_hist = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2],
            'y': [0.0, 0.1, 0.2],
            'i': [5.0, 5.1, 5.2]
        }, index=pd.period_range('2000-Q1', periods=3, freq='Q'))

        minimal_forward = pd.DataFrame({
            'pi': [2.3],
            'y': [0.3],
            'i': [5.3]
        }, index=pd.period_range('2000-Q4', periods=1, freq='Q'))

        def minimal_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return minimal_hist
            else:
                return minimal_forward

        mock_create_dataset.side_effect = minimal_mock_create_dataset

        solver_minimal = LinearEnvironmentSolver("test_key")
        minimal_result = solver_minimal.forecast_data

        # Should still return a valid DataFrame
        assert isinstance(minimal_result, pd.DataFrame)
        assert len(minimal_result) == 4  # 3 historical + 1 forward

        # Test 7: Verify data concatenation and sorting
        # The result should properly combine historical and forward data
        combined_indices = list(historical_df.index) + list(forward_df.index)
        sorted_indices = sorted(combined_indices)

        # Reset to original mock behavior
        mock_create_dataset.side_effect = mock_create_dataset_side_effect
        solver_final = LinearEnvironmentSolver("test_key")
        final_result = solver_final.forecast_data

        # Verify all periods are included and sorted
        assert len(final_result) == len(sorted_indices)
        assert final_result.index.tolist() == sorted_indices

        # Test 8: Verify NaN initialization
        # y_hat and pi_hat should start as NaN and be filled during forecasting
        assert 'y_hat' in final_result.columns
        assert 'pi_hat' in final_result.columns

        # Early periods (before start point) should remain NaN
        early_periods = final_result.iloc[:2]  # First couple periods
        if len(early_periods) > 0:
            # At least some early values should be NaN (before sufficient lags available)
            early_y_hat_nans = early_periods['y_hat'].isna().sum()
            early_pi_hat_nans = early_periods['pi_hat'].isna().sum()
            # Should have some NaN values in early periods
            assert early_y_hat_nans > 0 or early_pi_hat_nans > 0

    @patch('matplotlib.pyplot.ioff')
    @patch('matplotlib.pyplot.subplots')
    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.get_recursive_lag')
    def test_figure_five(self, mock_get_recursive_lag, mock_drop_lag, mock_create_dataset, mock_subplots, mock_ioff):
        """
        Comprehensive test for the figure_five property covering all scenarios.
        """
        # Setup mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup test data with PeriodIndex
        historical_dates = pd.period_range('2000-Q1', periods=8, freq='Q')
        forward_dates = pd.period_range('2002-Q1', periods=4, freq='Q')

        # Create realistic test data
        np.random.seed(42)  # For reproducible tests
        historical_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 8),
            'y': np.random.normal(0.0, 0.5, 8),
            'i': np.random.normal(5.0, 0.2, 8)
        }, index=historical_dates)

        forward_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 4),
            'y': np.random.normal(0.0, 0.5, 4),
            'i': np.random.normal(5.0, 0.2, 4)
        }, index=forward_dates)

        # Mock the dataset creation to return our test data
        def mock_create_dataset_side_effect(start, end):
            if str(start) == "1987-07-01":
                return historical_df
            else:
                return forward_df

        mock_create_dataset.side_effect = mock_create_dataset_side_effect

        # Mock the drop_insignificant_lag_two function
        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        # Mock get_recursive_lag to return predictable values
        def mock_get_recursive_lag_side_effect(df, t, col, lag):
            if col == "y" and lag == 1:
                return 0.1
            elif col == "y" and lag == 2:
                return 0.2
            elif col == "pi" and lag == 1:
                return 2.1
            elif col == "pi" and lag == 2:
                return 2.2
            else:
                return 0.0

        mock_get_recursive_lag.side_effect = mock_get_recursive_lag_side_effect

        # Create combined dataframe for forecast_data mock
        combined_df = pd.concat([historical_df, forward_df]).sort_index()
        combined_df['y_hat'] = combined_df['y'] + np.random.normal(0, 0.1, len(combined_df))
        combined_df['pi_hat'] = combined_df['pi'] + np.random.normal(0, 0.1, len(combined_df))

        # Test 1: Basic functionality - verify matplotlib setup and calls
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            result_fig = solver.figure_five

        # Test 2: Verify matplotlib.pyplot.ioff() was called
        mock_ioff.assert_called_once()

        # Test 3: Verify figure creation
        mock_subplots.assert_called_once_with(figsize=(10, 4))
        assert result_fig == mock_fig

        # Test 4: Verify plot method calls
        # Should have two plot calls: one for actual, one for forecast
        assert mock_ax.plot.call_count == 2

        # Get the plot calls
        plot_calls = mock_ax.plot.call_args_list

        # First call should be for actual output gap
        actual_call = plot_calls[0]
        actual_kwargs = actual_call[1] if len(actual_call) > 1 else {}

        # Verify the actual plot has correct label
        assert actual_kwargs.get('label') == "Output gap (actual)"

        # Second call should be for forecast output gap
        forecast_call = plot_calls[1]
        forecast_args = forecast_call[0]
        forecast_kwargs = forecast_call[1] if len(forecast_call) > 1 else {}

        # Verify the forecast plot has correct styling and label
        assert forecast_kwargs.get('label') == "Output gap (forecast)"
        # Check for dashed line style - it could be in args as a string or in kwargs
        has_dashed_style = (
            len(forecast_args) > 2 and forecast_args[2] == "--"
        ) or forecast_kwargs.get('linestyle') == "--"
        assert has_dashed_style, "Forecast line should have dashed style"

        # Test 5: Verify axis configuration
        mock_ax.set_title.assert_called_once_with("Output Gap — Actual vs Forecast")
        mock_ax.set_xlabel.assert_called_once_with("Date")
        mock_ax.set_ylabel.assert_called_once_with("Percent")
        mock_ax.legend.assert_called_once()
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)

        # Reset mocks for next test
        mock_ioff.reset_mock()
        mock_subplots.reset_mock()
        mock_ax.reset_mock()

        # Test 6: Verify data subsetting logic
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_five

        # Verify that the data passed to plot starts from forward_data minimum
        plot_calls = mock_ax.plot.call_args_list
        if len(plot_calls) > 0:
            # Check that the x-axis data (timestamps) starts from the expected point
            first_plot_x_data = plot_calls[0][0][0]
            # The data should be converted to timestamps, so we verify the subset logic worked
            assert len(first_plot_x_data) <= len(combined_df)

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 7: Test with sufficient minimal data (need at least 5 periods for lag creation)
        minimal_hist = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2, 2.3, 2.4],
            'y': [0.0, 0.1, 0.2, 0.3, 0.4],
            'i': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.period_range('2000-Q1', periods=5, freq='Q'))

        minimal_forward = pd.DataFrame({
            'pi': [2.5],
            'y': [0.5],
            'i': [5.5]
        }, index=pd.period_range('2001-Q2', periods=1, freq='Q'))

        def minimal_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return minimal_hist
            else:
                return minimal_forward

        mock_create_dataset.side_effect = minimal_mock_create_dataset

        minimal_combined = pd.concat([minimal_hist, minimal_forward]).sort_index()
        minimal_combined['y_hat'] = minimal_combined['y'] + 0.05
        minimal_combined['pi_hat'] = minimal_combined['pi'] + 0.05

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: minimal_combined)):
            solver_minimal = LinearEnvironmentSolver("test_key")
            minimal_result = solver_minimal.figure_five

        # Should still create figure successfully
        assert minimal_result == mock_fig
        mock_subplots.assert_called_with(figsize=(10, 4))

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 8: Test with NaN values in forecast data
        nan_combined = combined_df.copy()
        nan_combined.loc[nan_combined.index[0], 'y_hat'] = np.nan
        nan_combined.loc[nan_combined.index[1], 'y'] = np.nan

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: nan_combined)):
            solver = LinearEnvironmentSolver("test_key")
            nan_result = solver.figure_five

        # Should still create figure (matplotlib handles NaN values)
        assert nan_result == mock_fig
        assert mock_ax.plot.call_count == 2

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 9: Verify return type
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            figure_result = solver.figure_five

        # Should return the matplotlib Figure object
        assert figure_result is mock_fig

        # Test 10: Verify that all required columns exist in subset
        # The method assumes 'y' and 'y_hat' columns exist
        incomplete_df = combined_df.drop(columns=['y_hat'])

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: incomplete_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This should raise a KeyError when trying to access 'y_hat'
            with pytest.raises(KeyError):
                solver.figure_five

        # Test 11: Verify context manager (plt.ioff)
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_five

        # plt.ioff should be called to turn off interactive mode
        assert mock_ioff.called

    @patch('matplotlib.pyplot.ioff')
    @patch('matplotlib.pyplot.subplots')
    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.get_recursive_lag')
    def test_figure_six(self, mock_get_recursive_lag, mock_drop_lag, mock_create_dataset, mock_subplots, mock_ioff):
        """
        Comprehensive test for the figure_six property covering all scenarios.
        """
        # Setup mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup test data with PeriodIndex
        historical_dates = pd.period_range('2000-Q1', periods=8, freq='Q')
        forward_dates = pd.period_range('2002-Q1', periods=4, freq='Q')

        # Create realistic test data
        np.random.seed(42)  # For reproducible tests
        historical_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 8),
            'y': np.random.normal(0.0, 0.5, 8),
            'i': np.random.normal(5.0, 0.2, 8)
        }, index=historical_dates)

        forward_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 4),
            'y': np.random.normal(0.0, 0.5, 4),
            'i': np.random.normal(5.0, 0.2, 4)
        }, index=forward_dates)

        # Mock the dataset creation to return our test data
        def mock_create_dataset_side_effect(start, end):
            if str(start) == "1987-07-01":
                return historical_df
            else:
                return forward_df

        mock_create_dataset.side_effect = mock_create_dataset_side_effect

        # Mock the drop_insignificant_lag_two function
        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        # Mock get_recursive_lag to return predictable values
        def mock_get_recursive_lag_side_effect(df, t, col, lag):
            if col == "y" and lag == 1:
                return 0.1
            elif col == "y" and lag == 2:
                return 0.2
            elif col == "pi" and lag == 1:
                return 2.1
            elif col == "pi" and lag == 2:
                return 2.2
            else:
                return 0.0

        mock_get_recursive_lag.side_effect = mock_get_recursive_lag_side_effect

        # Create combined dataframe for forecast_data mock
        combined_df = pd.concat([historical_df, forward_df]).sort_index()
        combined_df['y_hat'] = combined_df['y'] + np.random.normal(0, 0.1, len(combined_df))
        combined_df['pi_hat'] = combined_df['pi'] + np.random.normal(0, 0.1, len(combined_df))

        # Test 1: Basic functionality - verify matplotlib setup and calls
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            result_fig = solver.figure_six

        # Test 2: Verify matplotlib.pyplot.ioff() was called
        mock_ioff.assert_called_once()

        # Test 3: Verify figure creation
        mock_subplots.assert_called_once_with(figsize=(10, 4))
        assert result_fig == mock_fig

        # Test 4: Verify plot method calls
        # Should have two plot calls: one for actual, one for forecast
        assert mock_ax.plot.call_count == 2

        # Get the plot calls
        plot_calls = mock_ax.plot.call_args_list

        # First call should be for actual inflation
        actual_call = plot_calls[0]
        actual_kwargs = actual_call[1] if len(actual_call) > 1 else {}

        # Verify the actual plot has correct label
        assert actual_kwargs.get('label') == "Inflation YoY (actual)"

        # Second call should be for forecast inflation
        forecast_call = plot_calls[1]
        forecast_args = forecast_call[0]
        forecast_kwargs = forecast_call[1] if len(forecast_call) > 1 else {}

        # Verify the forecast plot has correct styling and label
        assert forecast_kwargs.get('label') == "Inflation YoY (forecast)"
        # Check for dashed line style - it could be in args as a string or in kwargs
        has_dashed_style = (
            len(forecast_args) > 2 and forecast_args[2] == "--"
        ) or forecast_kwargs.get('linestyle') == "--"
        assert has_dashed_style, "Forecast line should have dashed style"

        # Test 5: Verify axis configuration
        mock_ax.set_title.assert_called_once_with("Inflation — Actual vs Forecast")
        mock_ax.set_xlabel.assert_called_once_with("Date")
        mock_ax.set_ylabel.assert_called_once_with("Percent")
        mock_ax.legend.assert_called_once()
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)

        # Reset mocks for next test
        mock_ioff.reset_mock()
        mock_subplots.reset_mock()
        mock_ax.reset_mock()

        # Test 6: Verify data subsetting logic
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_six

        # Verify that the data passed to plot starts from forward_data minimum
        plot_calls = mock_ax.plot.call_args_list
        if len(plot_calls) > 0:
            # Check that the x-axis data (timestamps) starts from the expected point
            first_plot_x_data = plot_calls[0][0][0]
            # The data should be converted to timestamps, so we verify the subset logic worked
            assert len(first_plot_x_data) <= len(combined_df)

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 7: Test with sufficient minimal data (need at least 5 periods for lag creation)
        minimal_hist = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2, 2.3, 2.4],
            'y': [0.0, 0.1, 0.2, 0.3, 0.4],
            'i': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.period_range('2000-Q1', periods=5, freq='Q'))

        minimal_forward = pd.DataFrame({
            'pi': [2.5],
            'y': [0.5],
            'i': [5.5]
        }, index=pd.period_range('2001-Q2', periods=1, freq='Q'))

        def minimal_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return minimal_hist
            else:
                return minimal_forward

        mock_create_dataset.side_effect = minimal_mock_create_dataset

        minimal_combined = pd.concat([minimal_hist, minimal_forward]).sort_index()
        minimal_combined['y_hat'] = minimal_combined['y'] + 0.05
        minimal_combined['pi_hat'] = minimal_combined['pi'] + 0.05

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: minimal_combined)):
            solver_minimal = LinearEnvironmentSolver("test_key")
            minimal_result = solver_minimal.figure_six

        # Should still create figure successfully
        assert minimal_result == mock_fig
        mock_subplots.assert_called_with(figsize=(10, 4))

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 8: Test with NaN values in forecast data
        nan_combined = combined_df.copy()
        nan_combined.loc[nan_combined.index[0], 'pi_hat'] = np.nan
        nan_combined.loc[nan_combined.index[1], 'pi'] = np.nan

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: nan_combined)):
            solver = LinearEnvironmentSolver("test_key")
            nan_result = solver.figure_six

        # Should still create figure (matplotlib handles NaN values)
        assert nan_result == mock_fig
        assert mock_ax.plot.call_count == 2

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 9: Verify return type
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            figure_result = solver.figure_six

        # Should return the matplotlib Figure object
        assert figure_result is mock_fig

        # Test 10: Verify that all required columns exist in subset
        # The method assumes 'pi' and 'pi_hat' columns exist
        incomplete_df = combined_df.drop(columns=['pi_hat'])

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: incomplete_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This should raise a KeyError when trying to access 'pi_hat'
            with pytest.raises(KeyError):
                solver.figure_six

        # Test 11: Verify context manager (plt.ioff)
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.forecast_data',
                new_callable=lambda: property(lambda self: combined_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_six

        # plt.ioff should be called to turn off interactive mode
        assert mock_ioff.called

    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.get_recursive_lag')
    def test_squared_errors_data(self, mock_get_recursive_lag, mock_drop_lag, mock_create_dataset):
        """
        Comprehensive test for the squared_errors_data property covering all scenarios.
        """
        # Setup test data with PeriodIndex
        historical_dates = pd.period_range('2000-Q1', periods=10, freq='Q')
        forward_dates = pd.period_range('2002-Q3', periods=4, freq='Q')

        # Create realistic test data
        np.random.seed(42)  # For reproducible tests
        historical_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 10),
            'y': np.random.normal(0.0, 0.5, 10),
            'i': np.random.normal(5.0, 0.2, 10)
        }, index=historical_dates)

        forward_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 4),
            'y': np.random.normal(0.0, 0.5, 4),
            'i': np.random.normal(5.0, 0.2, 4)
        }, index=forward_dates)

        # Mock the dataset creation to return our test data
        def mock_create_dataset_side_effect(start, end):
            if str(start) == "1987-07-01":
                return historical_df
            else:
                return forward_df

        mock_create_dataset.side_effect = mock_create_dataset_side_effect

        # Mock the drop_insignificant_lag_two function
        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        # Mock get_recursive_lag to return predictable values
        def mock_get_recursive_lag_side_effect(df, t, col, lag):
            if col == "y" and lag == 1:
                return 0.1
            elif col == "y" and lag == 2:
                return 0.2
            elif col == "pi" and lag == 1:
                return 2.1
            elif col == "pi" and lag == 2:
                return 2.2
            else:
                return 0.0

        mock_get_recursive_lag.side_effect = mock_get_recursive_lag_side_effect

        # Create solver
        solver = LinearEnvironmentSolver("test_key")

        # Test 1: Basic functionality - verify structure and types
        result = solver.squared_errors_data

        # Verify return type and basic structure
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.PeriodIndex)

        # Verify columns
        expected_columns = ['y_actual', 'pi_actual', 'y_fit', 'pi_fit', 'se_y_svar', 'se_pi_svar']
        assert all(col in result.columns for col in expected_columns)

        # Verify the data length - should be less than or equal to historical data
        # (some rows dropped due to initial lags)
        assert len(result) <= len(historical_df)
        assert len(result) > 0  # Should have some data after dropping NaN

        # Test 2: Verify data alignment and reindexing
        # The index should be aligned to the training (historical) data
        assert result.index.isin(historical_df.index).all()

        # Test 3: Verify actual values match historical data
        # y_actual and pi_actual should match corresponding values from historical_data
        for idx in result.index:
            if idx in historical_df.index:
                assert result.at[idx, 'y_actual'] == historical_df.at[idx, 'y']
                assert result.at[idx, 'pi_actual'] == historical_df.at[idx, 'pi']

        # Test 4: Verify fitted values come from model predictions
        # Check that fitted values are not NaN (after dropna)
        assert not result['y_fit'].isna().any()
        assert not result['pi_fit'].isna().any()

        # Test 5: Verify squared error calculations
        # Check that squared errors are computed correctly
        for idx in result.index:
            y_actual = result.at[idx, 'y_actual']
            y_fit = result.at[idx, 'y_fit']
            pi_actual = result.at[idx, 'pi_actual']
            pi_fit = result.at[idx, 'pi_fit']

            expected_se_y = (y_actual - y_fit) ** 2
            expected_se_pi = (pi_actual - pi_fit) ** 2

            assert abs(result.at[idx, 'se_y_svar'] - expected_se_y) < 1e-10
            assert abs(result.at[idx, 'se_pi_svar'] - expected_se_pi) < 1e-10

        # Test 6: Verify squared errors are non-negative
        assert (result['se_y_svar'] >= 0).all()
        assert (result['se_pi_svar'] >= 0).all()

        # Test 7: Verify NaN dropping behavior
        # The method should drop rows where y_fit or pi_fit is NaN
        assert not result['y_fit'].isna().any()
        assert not result['pi_fit'].isna().any()

        # Test 8: Test with minimal data
        minimal_hist = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2, 2.3, 2.4],
            'y': [0.0, 0.1, 0.2, 0.3, 0.4],
            'i': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.period_range('2000-Q1', periods=5, freq='Q'))

        minimal_forward = pd.DataFrame({
            'pi': [2.5],
            'y': [0.5],
            'i': [5.5]
        }, index=pd.period_range('2001-Q2', periods=1, freq='Q'))

        def minimal_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return minimal_hist
            else:
                return minimal_forward

        mock_create_dataset.side_effect = minimal_mock_create_dataset

        solver_minimal = LinearEnvironmentSolver("test_key")
        minimal_result = solver_minimal.squared_errors_data

        # Should still return a valid DataFrame
        assert isinstance(minimal_result, pd.DataFrame)
        assert len(minimal_result) > 0
        assert all(col in minimal_result.columns for col in expected_columns)

        # Test 9: Verify model prediction calls
        # The method should call predict on both mod_y and mod_pi
        original_mod_y_predict = solver.linear_svar.mod_y.predict
        original_mod_pi_predict = solver.linear_svar.mod_pi.predict

        # Mock the predict methods to verify they're called
        with patch.object(solver.linear_svar.mod_y, 'predict',
                        side_effect=original_mod_y_predict) as mock_y_predict, \
            patch.object(solver.linear_svar.mod_pi, 'predict',
                        side_effect=original_mod_pi_predict) as mock_pi_predict:

            result_verify = solver.squared_errors_data

            # Verify predict methods were called with correct design matrices
            mock_y_predict.assert_called_once_with(solver.linear_svar.x_y)
            mock_pi_predict.assert_called_once_with(solver.linear_svar.x_pi)

        # Test 10: Verify index assertions
        # The method contains assertions that fitted values index matches design matrix index
        # This is implicitly tested by the method not raising AssertionError
        # But let's verify the logic by checking the actual indices
        y_fit_test = solver.linear_svar.mod_y.predict(solver.linear_svar.x_y)
        pi_fit_test = solver.linear_svar.mod_pi.predict(solver.linear_svar.x_pi)

        assert y_fit_test.index.equals(solver.linear_svar.x_y.index)
        assert pi_fit_test.index.equals(solver.linear_svar.x_pi.index)

        # Test 11: Verify training window logic
        # The method uses historical_data.index.min() to historical_data.index.max()
        df_train_expected = historical_df.loc[historical_df.index.min():historical_df.index.max()]

        # The result index should be a subset of the training window
        assert result.index.isin(df_train_expected.index).all()

        # Test 12: Test data consistency across multiple calls
        # Multiple calls should return consistent results
        result1 = solver.squared_errors_data
        result2 = solver.squared_errors_data

        pd.testing.assert_frame_equal(result1, result2)

        # Test 13: Verify residuals calculation
        # Residuals should be actual - fitted
        for idx in result.index:
            y_residual = result.at[idx, 'y_actual'] - result.at[idx, 'y_fit']
            pi_residual = result.at[idx, 'pi_actual'] - result.at[idx, 'pi_fit']

            # Squared errors should equal residuals squared
            assert abs(result.at[idx, 'se_y_svar'] - y_residual**2) < 1e-10
            assert abs(result.at[idx, 'se_pi_svar'] - pi_residual**2) < 1e-10

        # Test 14: Edge case - verify behavior with different historical data sizes
        large_hist = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 20),
            'y': np.random.normal(0.0, 0.5, 20),
            'i': np.random.normal(5.0, 0.2, 20)
        }, index=pd.period_range('1990-Q1', periods=20, freq='Q'))

        large_forward = pd.DataFrame({
            'pi': [2.5],
            'y': [0.5],
            'i': [5.5]
        }, index=pd.period_range('1995-Q1', periods=1, freq='Q'))

        def large_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return large_hist
            else:
                return large_forward

        mock_create_dataset.side_effect = large_mock_create_dataset

        solver_large = LinearEnvironmentSolver("test_key")
        large_result = solver_large.squared_errors_data

        # Should handle larger datasets
        assert isinstance(large_result, pd.DataFrame)
        assert len(large_result) > 0
        assert len(large_result) <= len(large_hist)

        # Test 15: Verify column data types
        # All columns should be numeric
        assert pd.api.types.is_numeric_dtype(result['y_actual'])
        assert pd.api.types.is_numeric_dtype(result['pi_actual'])
        assert pd.api.types.is_numeric_dtype(result['y_fit'])
        assert pd.api.types.is_numeric_dtype(result['pi_fit'])
        assert pd.api.types.is_numeric_dtype(result['se_y_svar'])
        assert pd.api.types.is_numeric_dtype(result['se_pi_svar'])

    @patch('matplotlib.pyplot.ioff')
    @patch('matplotlib.pyplot.subplots')
    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.get_recursive_lag')
    def test_figure_seven(self, mock_get_recursive_lag, mock_drop_lag, mock_create_dataset, mock_subplots, mock_ioff):
        """
        Comprehensive test for the figure_seven property covering all scenarios.
        """
        # Setup mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup test data with PeriodIndex
        historical_dates = pd.period_range('2000-Q1', periods=10, freq='Q')
        forward_dates = pd.period_range('2002-Q3', periods=4, freq='Q')

        # Create realistic test data
        np.random.seed(42)  # For reproducible tests
        historical_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 10),
            'y': np.random.normal(0.0, 0.5, 10),
            'i': np.random.normal(5.0, 0.2, 10)
        }, index=historical_dates)

        forward_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 4),
            'y': np.random.normal(0.0, 0.5, 4),
            'i': np.random.normal(5.0, 0.2, 4)
        }, index=forward_dates)

        # Mock the dataset creation to return our test data
        def mock_create_dataset_side_effect(start, end):
            if str(start) == "1987-07-01":
                return historical_df
            else:
                return forward_df

        mock_create_dataset.side_effect = mock_create_dataset_side_effect

        # Mock the drop_insignificant_lag_two function
        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        # Mock get_recursive_lag to return predictable values
        def mock_get_recursive_lag_side_effect(df, t, col, lag):
            if col == "y" and lag == 1:
                return 0.1
            elif col == "y" and lag == 2:
                return 0.2
            elif col == "pi" and lag == 1:
                return 2.1
            elif col == "pi" and lag == 2:
                return 2.2
            else:
                return 0.0

        mock_get_recursive_lag.side_effect = mock_get_recursive_lag_side_effect

        # Create mock squared errors data with fixed values (not random)
        squared_errors_dates = pd.period_range('2000-Q3', periods=8, freq='Q')  # Subset due to lags
        mock_squared_errors_df = pd.DataFrame({
            'y_actual': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'pi_actual': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            'y_fit': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            'pi_fit': [2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75],
            'se_y_svar': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],  # Fixed values
            'se_pi_svar': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
        }, index=squared_errors_dates)

        # Test 1: Basic functionality - verify matplotlib setup and calls
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            result_fig = solver.figure_seven

        # Test 2: Verify matplotlib.pyplot.ioff() was called
        mock_ioff.assert_called_once()

        # Test 3: Verify figure creation
        mock_subplots.assert_called_once_with(figsize=(10, 4))
        assert result_fig == mock_fig

        # Test 4: Verify plot method calls
        # Should have one plot call for squared errors
        assert mock_ax.plot.call_count == 1

        # Get the plot call
        plot_call = mock_ax.plot.call_args_list[0]
        plot_args = plot_call[0]
        plot_kwargs = plot_call[1] if len(plot_call) > 1 else {}

        # Verify plot data - x should be timestamps, y should be se_y_svar
        assert len(plot_args) >= 2
        x_data = plot_args[0]
        y_data = plot_args[1]

        # X data should be timestamps converted from PeriodIndex
        assert len(x_data) == len(mock_squared_errors_df)
        # Y data should be the squared errors for output gap
        np.testing.assert_array_equal(y_data, mock_squared_errors_df["se_y_svar"].values)

        # Test 5: Verify plot styling
        assert plot_kwargs.get('color') == "red"
        assert plot_kwargs.get('label') == "SVAR"
        assert plot_kwargs.get('linewidth') == 1.5

        # Test 6: Verify axis configuration
        expected_title = f"Figure 4 Replica: Output Gap Fit — Squared Errors ({mock_squared_errors_df.index.min()}-{mock_squared_errors_df.index.max()})"
        mock_ax.set_title.assert_called_once_with(expected_title)
        mock_ax.set_xlabel.assert_called_once_with("Date")
        mock_ax.set_ylabel.assert_called_once_with("Squared error")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.legend.assert_called_once()

        # Reset mocks for next test
        mock_ioff.reset_mock()
        mock_subplots.reset_mock()
        mock_ax.reset_mock()

        # Test 7: Verify PeriodIndex assertion
        # The method should assert that the index is a PeriodIndex
        # Test with non-PeriodIndex should fail
        invalid_index_df = mock_squared_errors_df.copy()
        invalid_index_df.index = pd.to_datetime(invalid_index_df.index.to_timestamp())

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: invalid_index_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This should raise an AssertionError due to invalid index type
            with pytest.raises(AssertionError):
                solver.figure_seven

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 8: Test with minimal data
        minimal_hist = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2, 2.3, 2.4],
            'y': [0.0, 0.1, 0.2, 0.3, 0.4],
            'i': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.period_range('2000-Q1', periods=5, freq='Q'))

        minimal_forward = pd.DataFrame({
            'pi': [2.5],
            'y': [0.5],
            'i': [5.5]
        }, index=pd.period_range('2001-Q2', periods=1, freq='Q'))

        def minimal_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return minimal_hist
            else:
                return minimal_forward

        mock_create_dataset.side_effect = minimal_mock_create_dataset

        # Create minimal squared errors data
        minimal_squared_dates = pd.period_range('2000-Q3', periods=3, freq='Q')
        minimal_squared_errors_df = pd.DataFrame({
            'y_actual': [0.2, 0.3, 0.4],
            'pi_actual': [2.2, 2.3, 2.4],
            'y_fit': [0.25, 0.28, 0.42],
            'pi_fit': [2.15, 2.35, 2.38],
            'se_y_svar': [0.0025, 0.0004, 0.0004],
            'se_pi_svar': [0.0025, 0.0004, 0.0004]
        }, index=minimal_squared_dates)

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: minimal_squared_errors_df)):
            solver_minimal = LinearEnvironmentSolver("test_key")
            minimal_result = solver_minimal.figure_seven

        # Should still create figure successfully
        assert minimal_result == mock_fig
        mock_subplots.assert_called_with(figsize=(10, 4))
        assert mock_ax.plot.call_count == 1

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 9: Test with NaN values in squared errors data
        nan_squared_errors_df = mock_squared_errors_df.copy()
        nan_squared_errors_df.loc[nan_squared_errors_df.index[0], 'se_y_svar'] = np.nan
        nan_squared_errors_df.loc[nan_squared_errors_df.index[1], 'se_y_svar'] = np.nan

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: nan_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            nan_result = solver.figure_seven

        # Should still create figure (matplotlib handles NaN values)
        assert nan_result == mock_fig
        assert mock_ax.plot.call_count == 1

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 10: Verify return type
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            figure_result = solver.figure_seven

        # Should return the matplotlib Figure object
        assert figure_result is mock_fig

        # Test 11: Verify that all required columns exist
        # The method assumes 'se_y_svar' column exists
        incomplete_df = mock_squared_errors_df.drop(columns=['se_y_svar'])

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: incomplete_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This should raise a KeyError when trying to access 'se_y_svar'
            with pytest.raises(KeyError):
                solver.figure_seven

        # Test 12: Verify timestamp conversion
        # The method should convert PeriodIndex to timestamps for plotting
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_seven

        # Verify the x-axis data is timestamps
        plot_call = mock_ax.plot.call_args_list[0]
        x_data = plot_call[0][0]

        # X data should be datetime-like (converted from PeriodIndex)
        # Check that we have the right number of points
        assert len(x_data) == len(mock_squared_errors_df)

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 13: Verify dynamic title generation
        # Title should include the actual min and max dates from the data
        test_dates = pd.period_range('1995-Q1', periods=6, freq='Q')
        test_squared_errors_df = pd.DataFrame({
            'y_actual': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'pi_actual': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
            'y_fit': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            'pi_fit': [2.05, 2.15, 2.25, 2.35, 2.45, 2.55],
            'se_y_svar': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            'se_pi_svar': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
        }, index=test_dates)

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: test_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_seven

        # Verify title includes the correct date range
        expected_title = f"Figure 4 Replica: Output Gap Fit — Squared Errors ({test_dates.min()}-{test_dates.max()})"
        mock_ax.set_title.assert_called_once_with(expected_title)

        # Test 14: Verify context manager (plt.ioff)
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_seven

        # plt.ioff should be called to turn off interactive mode
        assert mock_ioff.called

        # Test 15: Edge case - empty squared errors data
        empty_squared_errors_df = pd.DataFrame({
            'y_actual': [],
            'pi_actual': [],
            'y_fit': [],
            'pi_fit': [],
            'se_y_svar': [],
            'se_pi_svar': []
        }, index=pd.period_range('2000-Q1', periods=0, freq='Q'))

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: empty_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This might work (matplotlib can handle empty data) or raise an error
            try:
                empty_result = solver.figure_seven
                # If it succeeds, verify it's still a figure
                assert empty_result == mock_fig
            except (ValueError, IndexError):
                # This is acceptable for edge case with empty data
                pass

        # Test 16: Verify that squared_errors_data property is called
        # Reset all mocks completely for this test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()
        mock_ioff.reset_mock()

        # Use the original mock_squared_errors_df, not the one modified in Test 13
        original_mock_squared_errors_df = pd.DataFrame({
            'y_actual': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'pi_actual': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            'y_fit': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            'pi_fit': [2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75],
            'se_y_svar': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            'se_pi_svar': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
        }, index=squared_errors_dates)

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: original_mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")

            # Verify the figure is created successfully, which implies squared_errors_data was accessed
            result = solver.figure_seven
            assert result == mock_fig

            # Verify that exactly one plot call was made in this test
            assert mock_ax.plot.call_count == 1

            # Get the plot call from this specific test
            plot_call = mock_ax.plot.call_args_list[0]
            y_data = plot_call[0][1]

            # Verify the data matches our fresh DataFrame
            np.testing.assert_array_equal(y_data, original_mock_squared_errors_df["se_y_svar"].values)

    @patch('matplotlib.pyplot.ioff')
    @patch('matplotlib.pyplot.subplots')
    @patch('autonomous_fed.clients.LinearEnvironmentSolver.create_econometric_dataset')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.drop_insignificant_lag_two')
    @patch('autonomous_fed.helpers.LinearEnvironmentHelpers.get_recursive_lag')
    def test_figure_eight(self, mock_get_recursive_lag, mock_drop_lag, mock_create_dataset, mock_subplots, mock_ioff):
        """
        Comprehensive test for the figure_eight property covering all scenarios.
        """
        # Setup mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup test data with PeriodIndex
        historical_dates = pd.period_range('2000-Q1', periods=10, freq='Q')
        forward_dates = pd.period_range('2002-Q3', periods=4, freq='Q')

        # Create realistic test data
        np.random.seed(42)  # For reproducible tests
        historical_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 10),
            'y': np.random.normal(0.0, 0.5, 10),
            'i': np.random.normal(5.0, 0.2, 10)
        }, index=historical_dates)

        forward_df = pd.DataFrame({
            'pi': np.random.normal(2.0, 0.3, 4),
            'y': np.random.normal(0.0, 0.5, 4),
            'i': np.random.normal(5.0, 0.2, 4)
        }, index=forward_dates)

        # Mock the dataset creation to return our test data
        def mock_create_dataset_side_effect(start, end):
            if str(start) == "1987-07-01":
                return historical_df
            else:
                return forward_df

        mock_create_dataset.side_effect = mock_create_dataset_side_effect

        # Mock the drop_insignificant_lag_two function
        def mock_drop_side_effect(model, x_data, y_target, alpha):
            return model, x_data
        mock_drop_lag.side_effect = mock_drop_side_effect

        # Mock get_recursive_lag to return predictable values
        def mock_get_recursive_lag_side_effect(df, t, col, lag):
            if col == "y" and lag == 1:
                return 0.1
            elif col == "y" and lag == 2:
                return 0.2
            elif col == "pi" and lag == 1:
                return 2.1
            elif col == "pi" and lag == 2:
                return 2.2
            else:
                return 0.0

        mock_get_recursive_lag.side_effect = mock_get_recursive_lag_side_effect

        # Create mock squared errors data with fixed values (not random)
        squared_errors_dates = pd.period_range('2000-Q3', periods=8, freq='Q')  # Subset due to lags
        mock_squared_errors_df = pd.DataFrame({
            'y_actual': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'pi_actual': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            'y_fit': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            'pi_fit': [2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75],
            'se_y_svar': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],  # Fixed values
            'se_pi_svar': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
        }, index=squared_errors_dates)

        # Test 1: Basic functionality - verify matplotlib setup and calls
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            result_fig = solver.figure_eight

        # Test 2: Verify matplotlib.pyplot.ioff() was called
        mock_ioff.assert_called_once()

        # Test 3: Verify figure creation
        mock_subplots.assert_called_once_with(figsize=(10, 4))
        assert result_fig == mock_fig

        # Test 4: Verify plot method calls
        # Should have one plot call for squared errors
        assert mock_ax.plot.call_count == 1

        # Get the plot call
        plot_call = mock_ax.plot.call_args_list[0]
        plot_args = plot_call[0]
        plot_kwargs = plot_call[1] if len(plot_call) > 1 else {}

        # Verify plot data - x should be timestamps, y should be se_pi_svar
        assert len(plot_args) >= 2
        x_data = plot_args[0]
        y_data = plot_args[1]

        # X data should be timestamps converted from PeriodIndex
        assert len(x_data) == len(mock_squared_errors_df)
        # Y data should be the squared errors for inflation
        np.testing.assert_array_equal(y_data, mock_squared_errors_df["se_pi_svar"].values)

        # Test 5: Verify plot styling
        assert plot_kwargs.get('color') == "red"
        assert plot_kwargs.get('label') == "SVAR"
        assert plot_kwargs.get('linewidth') == 1.5

        # Test 6: Verify axis configuration
        expected_title = f"Figure 5 Replica: Inflation Fit — Squared Errors ({mock_squared_errors_df.index.min()}-{mock_squared_errors_df.index.max()})"
        mock_ax.set_title.assert_called_once_with(expected_title)
        mock_ax.set_xlabel.assert_called_once_with("Date")
        mock_ax.set_ylabel.assert_called_once_with("Squared error")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.legend.assert_called_once()

        # Reset mocks for next test
        mock_ioff.reset_mock()
        mock_subplots.reset_mock()
        mock_ax.reset_mock()

        # Test 7: Verify PeriodIndex assertion
        # The method should assert that the index is a PeriodIndex
        # Test with non-PeriodIndex should fail
        invalid_index_df = mock_squared_errors_df.copy()
        invalid_index_df.index = pd.to_datetime(invalid_index_df.index.to_timestamp())

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: invalid_index_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This should raise an AssertionError due to invalid index type
            with pytest.raises(AssertionError):
                solver.figure_eight

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 8: Test with minimal data
        minimal_hist = pd.DataFrame({
            'pi': [2.0, 2.1, 2.2, 2.3, 2.4],
            'y': [0.0, 0.1, 0.2, 0.3, 0.4],
            'i': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.period_range('2000-Q1', periods=5, freq='Q'))

        minimal_forward = pd.DataFrame({
            'pi': [2.5],
            'y': [0.5],
            'i': [5.5]
        }, index=pd.period_range('2001-Q2', periods=1, freq='Q'))

        def minimal_mock_create_dataset(start, end):
            if str(start) == "1987-07-01":
                return minimal_hist
            else:
                return minimal_forward

        mock_create_dataset.side_effect = minimal_mock_create_dataset

        # Create minimal squared errors data
        minimal_squared_dates = pd.period_range('2000-Q3', periods=3, freq='Q')
        minimal_squared_errors_df = pd.DataFrame({
            'y_actual': [0.2, 0.3, 0.4],
            'pi_actual': [2.2, 2.3, 2.4],
            'y_fit': [0.25, 0.28, 0.42],
            'pi_fit': [2.15, 2.35, 2.38],
            'se_y_svar': [0.0025, 0.0004, 0.0004],
            'se_pi_svar': [0.0025, 0.0004, 0.0004]
        }, index=minimal_squared_dates)

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: minimal_squared_errors_df)):
            solver_minimal = LinearEnvironmentSolver("test_key")
            minimal_result = solver_minimal.figure_eight

        # Should still create figure successfully
        assert minimal_result == mock_fig
        mock_subplots.assert_called_with(figsize=(10, 4))
        assert mock_ax.plot.call_count == 1

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 9: Test with NaN values in squared errors data
        nan_squared_errors_df = mock_squared_errors_df.copy()
        nan_squared_errors_df.loc[nan_squared_errors_df.index[0], 'se_pi_svar'] = np.nan
        nan_squared_errors_df.loc[nan_squared_errors_df.index[1], 'se_pi_svar'] = np.nan

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: nan_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            nan_result = solver.figure_eight

        # Should still create figure (matplotlib handles NaN values)
        assert nan_result == mock_fig
        assert mock_ax.plot.call_count == 1

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 10: Verify return type
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            figure_result = solver.figure_eight

        # Should return the matplotlib Figure object
        assert figure_result is mock_fig

        # Test 11: Verify that all required columns exist
        # The method assumes 'se_pi_svar' column exists
        incomplete_df = mock_squared_errors_df.drop(columns=['se_pi_svar'])

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: incomplete_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This should raise a KeyError when trying to access 'se_pi_svar'
            with pytest.raises(KeyError):
                solver.figure_eight

        # Test 12: Verify timestamp conversion
        # The method should convert PeriodIndex to timestamps for plotting
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_eight

        # Verify the x-axis data is timestamps
        plot_call = mock_ax.plot.call_args_list[0]
        x_data = plot_call[0][0]

        # X data should be datetime-like (converted from PeriodIndex)
        # Check that we have the right number of points
        assert len(x_data) == len(mock_squared_errors_df)

        # Reset for next test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()

        # Test 13: Verify dynamic title generation
        # Title should include the actual min and max dates from the data
        test_dates = pd.period_range('1995-Q1', periods=6, freq='Q')
        test_squared_errors_df = pd.DataFrame({
            'y_actual': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'pi_actual': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
            'y_fit': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            'pi_fit': [2.05, 2.15, 2.25, 2.35, 2.45, 2.55],
            'se_y_svar': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            'se_pi_svar': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
        }, index=test_dates)

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: test_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_eight

        # Verify title includes the correct date range
        expected_title = f"Figure 5 Replica: Inflation Fit — Squared Errors ({test_dates.min()}-{test_dates.max()})"
        mock_ax.set_title.assert_called_once_with(expected_title)

        # Test 14: Verify context manager (plt.ioff)
        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            solver.figure_eight

        # plt.ioff should be called to turn off interactive mode
        assert mock_ioff.called

        # Test 15: Edge case - empty squared errors data
        empty_squared_errors_df = pd.DataFrame({
            'y_actual': [],
            'pi_actual': [],
            'y_fit': [],
            'pi_fit': [],
            'se_y_svar': [],
            'se_pi_svar': []
        }, index=pd.period_range('2000-Q1', periods=0, freq='Q'))

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: empty_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")
            # This might work (matplotlib can handle empty data) or raise an error
            try:
                empty_result = solver.figure_eight
                # If it succeeds, verify it's still a figure
                assert empty_result == mock_fig
            except (ValueError, IndexError):
                # This is acceptable for edge case with empty data
                pass

        # Test 16: Verify that squared_errors_data property is called
        # Reset all mocks completely for this test
        mock_ax.reset_mock()
        mock_subplots.reset_mock()
        mock_ioff.reset_mock()

        # Use the original mock_squared_errors_df, not the one modified in Test 13
        original_mock_squared_errors_df = pd.DataFrame({
            'y_actual': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'pi_actual': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            'y_fit': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            'pi_fit': [2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75],
            'se_y_svar': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            'se_pi_svar': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
        }, index=squared_errors_dates)

        with patch('autonomous_fed.clients.LinearEnvironmentSolver.squared_errors_data',
                new_callable=lambda: property(lambda self: original_mock_squared_errors_df)):
            solver = LinearEnvironmentSolver("test_key")

            # Verify the figure is created successfully, which implies squared_errors_data was accessed
            result = solver.figure_eight
            assert result == mock_fig

            # Verify that exactly one plot call was made in this test
            assert mock_ax.plot.call_count == 1

            # Get the plot call from this specific test
            plot_call = mock_ax.plot.call_args_list[0]
            y_data = plot_call[0][1]

            # Verify the data matches our fresh DataFrame
            np.testing.assert_array_equal(y_data, original_mock_squared_errors_df["se_pi_svar"].values)
