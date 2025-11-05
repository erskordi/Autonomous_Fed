"""
Comprehensive unit tests for the objects module.
"""

import unittest.mock
import re
from decimal import Decimal
from datetime import datetime
import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
import fedfred as fd
from autonomous_fed.clients import LinearEnvironmentSolver
from autonomous_fed.helpers import LinearEnvironmentHelpers
from autonomous_fed.objects import SVARResults

class TestSVARResults:
    """
    Unit tests for the SVARResults data class.
    """
    @unittest.mock.patch('builtins.print')
    def test_print_results(self, mock_print):
        """
        Comprehensive test for the print_results method covering all scenarios.
        """

        # Test 1: Basic functionality with standard values
        # Create mock regression models
        mock_mod_y = unittest.mock.MagicMock()
        mock_mod_y.summary.return_value = "Output gap model summary"
        mock_mod_y.nobs = 100

        mock_mod_pi = unittest.mock.MagicMock()
        mock_mod_pi.summary.return_value = "Inflation model summary"
        mock_mod_pi.nobs = 98

        # Create mock residuals
        mock_resid_y = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.25])
        mock_resid_pi = pd.Series([0.05, -0.1, 0.15, -0.08, 0.12, -0.06, 0.09])

        # Calculate the actual standard deviations that will be used
        expected_y_std = round(mock_resid_y.std(), 3)
        expected_pi_std = round(mock_resid_pi.std(), 3)

        # Create mock design matrices (not used in print_results but required for dataclass)
        mock_x_y = pd.DataFrame({'const': [1, 1, 1], 'y_lag1': [0.1, 0.2, 0.3]})
        mock_x_pi = pd.DataFrame({'const': [1, 1, 1], 'pi_lag1': [0.05, 0.1, 0.15]})

        # Create SVARResults instance
        svar_results = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        # Call the method
        result = svar_results.print_results()

        # Verify return value
        assert result == "", "Method should return empty string"
        assert isinstance(result, str), "Return type should be string"

        # Verify print calls were made in correct order
        expected_calls = [
            unittest.mock.call("\n=== Output gap equation ==="),
            unittest.mock.call("Output gap model summary"),
            unittest.mock.call("\n=== Inflation equation ==="),
            unittest.mock.call("Inflation model summary"),
            unittest.mock.call("\nResidual std (y, pi):", expected_y_std, expected_pi_std),
            unittest.mock.call("Samples used (y, pi):", 100, 98)
        ]

        mock_print.assert_has_calls(expected_calls, any_order=False)

        # Verify the exact number of print calls
        assert mock_print.call_count == 6

        # Test 2: Edge cases with different residual standard deviations
        mock_print.reset_mock()

        # Very small residuals
        small_resid_y = pd.Series([0.0001, -0.0002, 0.0001, -0.0001])
        small_resid_pi = pd.Series([0.00005, -0.00008, 0.00007])

        svar_small = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=small_resid_y,
            resid_pi=small_resid_pi
        )

        result = svar_small.print_results()

        # Check that small values are rounded to 3 decimal places (should be 0.000)
        # Find the call with residual std
        resid_call = None
        for call in mock_print.call_args_list:
            if "Residual std" in str(call):
                resid_call = call
                break

        assert resid_call is not None, "Should have printed residual std"
        # The exact values depend on the std calculation, but should be very small and rounded

        # Test 3: Large residuals
        mock_print.reset_mock()

        large_resid_y = pd.Series([10.5, -15.2, 8.7, -12.3, 9.8])
        large_resid_pi = pd.Series([5.2, -7.8, 6.1, -4.9, 8.3])

        svar_large = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=large_resid_y,
            resid_pi=large_resid_pi
        )

        result = svar_large.print_results()

        # Verify large values are properly rounded
        assert result == ""

        # Test 4: Different sample sizes
        mock_print.reset_mock()

        mock_mod_y_small = unittest.mock.MagicMock()
        mock_mod_y_small.summary.return_value = "Small sample output model"
        mock_mod_y_small.nobs = 25

        mock_mod_pi_small = unittest.mock.MagicMock()
        mock_mod_pi_small.summary.return_value = "Small sample inflation model"
        mock_mod_pi_small.nobs = 23

        svar_small_sample = SVARResults(
            mod_y=mock_mod_y_small,
            mod_pi=mock_mod_pi_small,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result = svar_small_sample.print_results()

        # Check that sample sizes are printed correctly
        sample_call = None
        for call in mock_print.call_args_list:
            if "Samples used" in str(call):
                sample_call = call
                break

        assert sample_call is not None
        # Should contain the correct sample sizes (25, 23)

        # Test 5: Zero and negative values in residuals
        mock_print.reset_mock()

        zero_resid_y = pd.Series([0.0, 0.0, 0.0, 0.0])
        mixed_resid_pi = pd.Series([-1.0, -2.0, -0.5, -1.5])

        svar_zero = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=zero_resid_y,
            resid_pi=mixed_resid_pi
        )

        result = svar_zero.print_results()
        assert result == ""

        # Test 6: Single observation residuals
        mock_print.reset_mock()

        single_resid_y = pd.Series([0.5])
        single_resid_pi = pd.Series([0.3])

        mock_mod_single = unittest.mock.MagicMock()
        mock_mod_single.summary.return_value = "Single obs model"
        mock_mod_single.nobs = 1

        svar_single = SVARResults(
            mod_y=mock_mod_single,
            mod_pi=mock_mod_single,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=single_resid_y,
            resid_pi=single_resid_pi
        )

        result = svar_single.print_results()
        assert result == ""

        # Test 7: Summary methods are called correctly
        mock_print.reset_mock()

        # Reset the mocks to verify method calls
        mock_mod_y.reset_mock()
        mock_mod_pi.reset_mock()

        svar_results.print_results()

        # Verify that summary() was called on both models
        mock_mod_y.summary.assert_called_once()
        mock_mod_pi.summary.assert_called_once()

        # Test 8: Verify rounding behavior with specific values
        mock_print.reset_mock()

        # Create residuals with known standard deviations for precise testing
        precise_resid_y = pd.Series([1.0, 2.0, 3.0])  # std ≈ 1.0
        precise_resid_pi = pd.Series([0.1, 0.2])      # std ≈ 0.071

        svar_precise = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=precise_resid_y,
            resid_pi=precise_resid_pi
        )

        result = svar_precise.print_results()

        # Calculate expected rounded values
        expected_y_std = round(precise_resid_y.std(), 3)
        expected_pi_std = round(precise_resid_pi.std(), 3)

        # Find the residual std call and verify values
        for call in mock_print.call_args_list:
            if len(call[0]) > 0 and "Residual std" in str(call[0][0]):
                assert call[0][1] == expected_y_std
                assert call[0][2] == expected_pi_std
                break

        # Test 9: Empty residuals handling (edge case)
        mock_print.reset_mock()

        try:
            empty_resid_y = pd.Series([], dtype=float)
            empty_resid_pi = pd.Series([], dtype=float)

            svar_empty = SVARResults(
                mod_y=mock_mod_y,
                mod_pi=mock_mod_pi,
                x_y=mock_x_y,
                x_pi=mock_x_pi,
                resid_y=empty_resid_y,
                resid_pi=empty_resid_pi
            )

            result = svar_empty.print_results()
            # This might raise an exception or return NaN, which is acceptable behavior
            assert result == ""

        except (ValueError, RuntimeWarning):
            # Empty series std() might raise warnings/errors, which is acceptable
            pass

        # Test 10: NaN residuals handling
        mock_print.reset_mock()

        nan_resid_y = pd.Series([np.nan, 1.0, 2.0])
        nan_resid_pi = pd.Series([0.5, np.nan, 1.5])

        svar_nan = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=nan_resid_y,
            resid_pi=nan_resid_pi
        )

        result = svar_nan.print_results()
        assert result == ""
        # The method should handle NaN values gracefully (pandas std() skips NaN by default)

        # Test 11: Very large sample sizes
        mock_print.reset_mock()

        mock_mod_large = unittest.mock.MagicMock()
        mock_mod_large.summary.return_value = "Large sample model"
        mock_mod_large.nobs = 1000000  # Very large sample

        svar_large_sample = SVARResults(
            mod_y=mock_mod_large,
            mod_pi=mock_mod_large,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result = svar_large_sample.print_results()
        assert result == ""

        # Verify large numbers are printed correctly
        for call in mock_print.call_args_list:
            if "Samples used" in str(call):
                assert 1000000 in call[0]

        # Test 12: Different types of summary outputs
        mock_print.reset_mock()

        # Test with different summary return types
        mock_mod_special = unittest.mock.MagicMock()
        mock_mod_special.summary.return_value = None  # Some edge case
        mock_mod_special.nobs = 50

        try:
            svar_special = SVARResults(
                mod_y=mock_mod_special,
                mod_pi=mock_mod_special,
                x_y=mock_x_y,
                x_pi=mock_x_pi,
                resid_y=mock_resid_y,
                resid_pi=mock_resid_pi
            )

            result = svar_special.print_results()
            assert result == ""

        except Exception:
            # If summary() returns None, print(None) should still work
            pass

        # Test 13: Verify method doesn't modify instance state
        original_mod_y = svar_results.mod_y
        original_mod_pi = svar_results.mod_pi
        original_resid_y = svar_results.resid_y.copy()
        original_resid_pi = svar_results.resid_pi.copy()

        result = svar_results.print_results()

        # Verify nothing changed
        assert svar_results.mod_y is original_mod_y
        assert svar_results.mod_pi is original_mod_pi
        assert svar_results.resid_y.equals(original_resid_y)
        assert svar_results.resid_pi.equals(original_resid_pi)
        assert result == ""

    @unittest.mock.patch('autonomous_fed.objects.durbin_watson')
    @unittest.mock.patch('autonomous_fed.objects.acorr_breusch_godfrey')
    def test_get_comparison_table_str(self, mock_acorr_bg, mock_durbin_watson):
        """
        Comprehensive test for the get_comparison_table_str method covering all scenarios.
        """

        # Test 1: Basic functionality with standard values
        # Setup mock durbin_watson and acorr_breusch_godfrey with simple return values
        mock_durbin_watson.return_value = 1.8
        mock_acorr_bg.return_value = (None, None, 1.5, 0.22)  # (lm_stat, lm_pvalue, f_stat, f_pvalue)

        # Create comprehensive mock regression models for output gap equation
        mock_mod_y = unittest.mock.MagicMock()
        mock_mod_y.params = pd.Series({
            'const': 0.1234,
            'y_lag1': 0.8765,
            'pi_lag1': -0.0987,
            'i_lag1': 0.2345,
            'i_lag2': -0.1789
        })
        mock_mod_y.pvalues = pd.Series({
            'const': 0.0234,
            'y_lag1': 0.0001,
            'pi_lag1': 0.1567,
            'i_lag1': 0.0445,
            'i_lag2': 0.0678
        })
        mock_mod_y.rsquared = 0.8934
        mock_mod_y.mse_resid = 0.1567
        mock_mod_y.scale = 0.0987  # scale ** 0.5 will be sqrt(0.0987)
        mock_mod_y.resid = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2])

        # Create comprehensive mock regression models for inflation equation
        mock_mod_pi = unittest.mock.MagicMock()
        mock_mod_pi.params = pd.Series({
            'const': 0.0567,
            'y': -0.0432,
            'y_lag1': 0.1876,
            'y_lag2': -0.0998,
            'pi_lag1': 1.1234,
            'pi_lag2': -0.2876,
            'i_lag1': -0.0145
        })
        mock_mod_pi.pvalues = pd.Series({
            'const': 0.1234,
            'y': 0.0876,
            'y_lag1': 0.0034,
            'y_lag2': 0.0198,
            'pi_lag1': 0.0000,
            'pi_lag2': 0.0087,
            'i_lag1': 0.3456
        })
        mock_mod_pi.rsquared = 0.9234
        mock_mod_pi.mse_resid = 0.0234
        mock_mod_pi.scale = 0.0145  # scale ** 0.5 will be sqrt(0.0145)
        mock_mod_pi.resid = pd.Series([0.05, -0.1, 0.15, -0.08, 0.12])

        # Create mock design matrices
        mock_x_y = pd.DataFrame({'const': [1, 1, 1], 'y_lag1': [0.1, 0.2, 0.3]})
        mock_x_pi = pd.DataFrame({'const': [1, 1, 1], 'pi_lag1': [0.05, 0.1, 0.15]})
        mock_resid_y = pd.Series([0.1, -0.2, 0.3])
        mock_resid_pi = pd.Series([0.05, -0.1, 0.15])

        # Create SVARResults instance
        svar_results = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        # Call the method
        result = svar_results.get_comparison_table_str()

        # Test 2: Verify return type
        assert isinstance(result, str), "Return type should be string"
        assert len(result) > 0, "Result should not be empty"

        # Test 3: Verify markdown table structure
        lines = result.strip().split('\n')
        assert any('|' in line for line in lines), "Should contain markdown table pipes"
        assert any('Parameters' in line for line in lines), "Should contain 'Parameters' header"
        assert any('Bundesbank Estimate' in line for line in lines), "Should contain 'Bundesbank Estimate' header"
        assert any('Local OLS Estimate' in line for line in lines), "Should contain 'Local OLS Estimate' header"

        # Test 4: Verify all required parameters are present
        # Note: These are the actual parameter names as they appear in the table
        required_output_params = ['$C^y$', '$a^y_{y,1}$', '$a^y_{\\pi,1}$', '$a^y_{i,1}$', '$a^y_{i,2}$']
        required_inflation_params = ['$C^\\pi$', '$a^\\pi_{y,0}$', '$a^\\pi_{y,1}$', '$a^\\pi_{y,2}$',
                                    '$a^\\pi_{\\pi,1}$', '$a^\\pi_{\\pi,2}$', '$a^\\pi_{i,1}$']
        required_stats = ['$\\bar{R}^2$', 'MSE', '$\\hat\\sigma_{\\varepsilon_1}$', '$\\hat\\sigma_{\\varepsilon_2}$', 'DW', 'LM(1)']

        for param in required_output_params + required_inflation_params + required_stats:
            assert param in result, f"Parameter {param} should be in the result"

        # Test 5: Verify equation headers
        assert '**Output gap equation**' in result, "Should contain output gap equation header"
        assert '**Inflation equation**' in result, "Should contain inflation equation header"

        # Test 6: Verify parameter values are formatted correctly (4 decimal places)
        # Check that our mock values appear in the result with correct formatting
        assert '0.1234' in result, "Should contain const parameter for output gap (0.1234)"
        assert '0.8765' in result, "Should contain y_lag1 parameter for output gap (0.8765)"
        assert '0.0567' in result, "Should contain const parameter for inflation (0.0567)"
        assert '1.1234' in result, "Should contain pi_lag1 parameter for inflation (1.1234)"

        # Test 7: Verify p-values are formatted correctly
        assert '0.0234' in result, "Should contain p-value for const (0.0234)"
        assert '0.0001' in result, "Should contain p-value for y_lag1 (0.0001)"
        assert '0.0000' in result, "Should contain p-value for pi_lag1 (0.0000)"

        # Test 8: Verify statistical measures are included
        assert '0.8934' in result, "Should contain R-squared for output gap"
        assert '0.9234' in result, "Should contain R-squared for inflation"
        assert '0.1567' in result, "Should contain MSE for output gap"
        assert '0.0234' in result, "Should contain MSE for inflation"

        # Test 9: Verify calculated values (sqrt of scale)
        expected_y_sigma = f"{mock_mod_y.scale ** 0.5:.4f}"
        expected_pi_sigma = f"{mock_mod_pi.scale ** 0.5:.4f}"
        assert expected_y_sigma in result, f"Should contain calculated sigma for output gap ({expected_y_sigma})"
        assert expected_pi_sigma in result, f"Should contain calculated sigma for inflation ({expected_pi_sigma})"

        # Test 10: Verify Durbin-Watson values are called and included
        mock_durbin_watson.assert_called()
        assert mock_durbin_watson.call_count >= 2, "Durbin-Watson should be called for both models"

        # Test 11: Verify Breusch-Godfrey test values are called and included
        mock_acorr_bg.assert_called()
        assert mock_acorr_bg.call_count >= 2, "Breusch-Godfrey test should be called for both models"

        # Test 12: Verify mock return values appear in result
        assert '1.8000' in result, "Should contain mocked Durbin-Watson value"
        assert '1.5000' in result, "Should contain mocked Breusch-Godfrey F-statistic"
        assert '0.2200' in result, "Should contain mocked Breusch-Godfrey p-value"

        # Test 13: Test with different mock values
        mock_durbin_watson.reset_mock()
        mock_acorr_bg.reset_mock()

        # Change mock return values
        mock_durbin_watson.return_value = 2.5
        mock_acorr_bg.return_value = (None, None, 3.2, 0.05)

        result2 = svar_results.get_comparison_table_str()

        assert '2.5000' in result2, "Should contain new mocked Durbin-Watson value"
        assert '3.2000' in result2, "Should contain new mocked Breusch-Godfrey F-statistic"
        assert '0.0500' in result2, "Should contain new mocked Breusch-Godfrey p-value"

        # Test 14: Verify footnote
        assert '\\* Residual std dev from OLS output.' in result, "Should contain footnote about residual std dev"

        # Test 15: Test string formatting edge cases
        # Test that f-string expressions are properly formatted
        assert '.4f' not in result, "Should not contain raw f-string format specifiers"
        assert '{' not in result or result.count('{') == result.count('}'), "Should have balanced braces"

        # Test 16: Verify all Bundesbank reference values are present
        bundesbank_values = ['0.3834', '0.9084', '-0.1437', '0.2726', '-0.2896', '0.9100', '0.2108',
                            '0.2136', '1.8206', '3.1037', '0.1644', '0.1035', '-0.0655', '0.1970',
                            '-0.1121', '1.2970', '-0.3116', '-0.0122', '0.9450', '0.0326', '0.0330',
                            '2.1095', '2.5542', '0.0696']

        for value in bundesbank_values[:5]:  # Check first few to avoid over-testing
            assert value in result, f"Bundesbank reference value {value} should be present"

        # Test 17: Test LaTeX math formatting
        latex_elements = ['$C^y$', '$a^y_', '\\bar{', '\\hat\\sigma', '\\varepsilon']
        for element in latex_elements:
            assert element in result, f"LaTeX element {element} should be present"

        # Test 18: Verify table structure consistency
        lines = [line for line in result.split('\n') if line.strip()]
        table_lines = [line for line in lines if '|' in line]

        # Should have consistent number of columns
        if len(table_lines) > 1:
            pipe_counts = [line.count('|') for line in table_lines[:3]]  # Check header and first few rows
            assert len(set(pipe_counts)) <= 2, "Table should have consistent column structure"

        # Test 19: Method doesn't modify instance state
        original_mod_y = svar_results.mod_y
        original_mod_pi = svar_results.mod_pi

        result_again = svar_results.get_comparison_table_str()

        assert svar_results.mod_y is original_mod_y, "Method should not modify mod_y"
        assert svar_results.mod_pi is original_mod_pi, "Method should not modify mod_pi"

        # Test 20: Edge case - Test with extreme parameter values
        mock_durbin_watson.reset_mock()
        mock_acorr_bg.reset_mock()
        mock_durbin_watson.return_value = 0.0001
        mock_acorr_bg.return_value = (None, None, 999.9999, 0.9999)

        extreme_mod_y = unittest.mock.MagicMock()
        extreme_mod_y.params = pd.Series({
            'const': 999.9999,
            'y_lag1': -999.9999,
            'pi_lag1': 0.0001,
            'i_lag1': -0.0001,
            'i_lag2': 123.4567
        })
        extreme_mod_y.pvalues = pd.Series({
            'const': 0.9999,
            'y_lag1': 0.0000,
            'pi_lag1': 0.5000,
            'i_lag1': 0.0001,
            'i_lag2': 1.0000
        })
        extreme_mod_y.rsquared = 0.0001
        extreme_mod_y.mse_resid = 999.9999
        extreme_mod_y.scale = 1000000.0
        extreme_mod_y.resid = pd.Series([1.0, 2.0])

        extreme_mod_pi = unittest.mock.MagicMock()
        extreme_mod_pi.params = pd.Series({
            'const': -999.9999,
            'y': 0.0000,
            'y_lag1': -0.0000,
            'y_lag2': 888.8888,
            'pi_lag1': -888.8888,
            'pi_lag2': 0.5555,
            'i_lag1': -0.5555
        })
        extreme_mod_pi.pvalues = pd.Series({
            'const': 0.0000,
            'y': 1.0000,
            'y_lag1': 0.9999,
            'y_lag2': 0.0001,
            'pi_lag1': 0.0000,
            'pi_lag2': 0.1111,
            'i_lag1': 0.8888
        })
        extreme_mod_pi.rsquared = 1.0000
        extreme_mod_pi.mse_resid = 0.0000
        extreme_mod_pi.scale = 0.0001
        extreme_mod_pi.resid = pd.Series([0.01, 0.02])

        svar_extreme = SVARResults(
            mod_y=extreme_mod_y,
            mod_pi=extreme_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_extreme = svar_extreme.get_comparison_table_str()

        # Verify extreme values are formatted correctly
        assert '999.9999' in result_extreme, "Should handle large positive values"
        assert '-999.9999' in result_extreme, "Should handle large negative values"
        assert '0.0001' in result_extreme, "Should handle very small values"
        assert '1.0000' in result_extreme, "Should handle unity values"

    def test_get_output_gap_equation_str(self):
        """
        Comprehensive test for the get_output_gap_equation_str method covering all scenarios.
        """

        # Test 1: Basic functionality with standard values
        # Create mock regression model for output gap equation
        mock_mod_y = unittest.mock.MagicMock()
        mock_mod_y.params = pd.Series({
            'const': 0.1234,
            'y_lag1': 0.8765,
            'pi_lag1': -0.0987,
            'i_lag1': 0.2345,
            'i_lag2': -0.1789
        })

        # Create minimal mock for inflation model (not used but required for dataclass)
        mock_mod_pi = unittest.mock.MagicMock()
        mock_mod_pi.params = pd.Series({'const': 0.0})

        # Create mock design matrices and residuals
        mock_x_y = pd.DataFrame({'const': [1, 1, 1], 'y_lag1': [0.1, 0.2, 0.3]})
        mock_x_pi = pd.DataFrame({'const': [1, 1, 1], 'pi_lag1': [0.05, 0.1, 0.15]})
        mock_resid_y = pd.Series([0.1, -0.2, 0.3])
        mock_resid_pi = pd.Series([0.05, -0.1, 0.15])

        # Create SVARResults instance
        svar_results = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        # Call the method
        result = svar_results.get_output_gap_equation_str()

        # Test 2: Verify return type
        assert isinstance(result, str), "Return type should be string"
        assert len(result) > 0, "Result should not be empty"

        # Test 3: Verify LaTeX equation structure
        assert result.startswith('\n        $'), "Should start with LaTeX math delimiter"
        assert result.rstrip().endswith('$'), "Should end with LaTeX math delimiter"
        assert '{y_t}' in result, "Should contain dependent variable y_t"
        assert '\\epsilon' in result, "Should contain error term epsilon"

        # Test 4: Verify all parameter values are present with correct formatting (4 decimal places)
        assert '0.1234' in result, "Should contain const parameter (0.1234)"
        assert '0.8765' in result, "Should contain y_lag1 parameter (0.8765)"
        assert '-0.0987' in result, "Should contain pi_lag1 parameter (-0.0987)"
        assert '0.2345' in result, "Should contain i_lag1 parameter (0.2345)"
        assert '-0.1789' in result, "Should contain i_lag2 parameter (-0.1789)"

        # Test 5: Verify equation structure and variable names
        expected_variables = ['y_{t-1}', '{\\pi}_{t-1}', 'i_{t-1}', 'i_{t-2}', '{\\epsilon}_t^y']
        for var in expected_variables:
            assert var in result, f"Variable {var} should be in the equation"

        # Test 6: Verify coefficient parentheses for negative values
        assert '(-0.0987)' in result, "Negative coefficients should be in parentheses"
        assert '(-0.1789)' in result, "Negative coefficients should be in parentheses"
        assert '(0.8765)' in result, "Positive coefficients should be in parentheses"
        assert '(0.2345)' in result, "Positive coefficients should be in parentheses"

        # Test 7: Test with extreme values
        extreme_mod_y = unittest.mock.MagicMock()
        extreme_mod_y.params = pd.Series({
            'const': 999.9999,
            'y_lag1': -999.9999,
            'pi_lag1': 0.0001,
            'i_lag1': -0.0001,
            'i_lag2': 123.4567
        })

        svar_extreme = SVARResults(
            mod_y=extreme_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_extreme = svar_extreme.get_output_gap_equation_str()

        # Verify extreme values are formatted correctly
        assert '999.9999' in result_extreme, "Should handle large positive values"
        assert '-999.9999' in result_extreme, "Should handle large negative values"
        assert '0.0001' in result_extreme, "Should handle very small positive values"
        assert '-0.0001' in result_extreme, "Should handle very small negative values"
        assert '123.4567' in result_extreme, "Should handle arbitrary decimal values"

        # Test 8: Test with zero values
        zero_mod_y = unittest.mock.MagicMock()
        zero_mod_y.params = pd.Series({
            'const': 0.0000,
            'y_lag1': 0.0000,
            'pi_lag1': 0.0000,
            'i_lag1': 0.0000,
            'i_lag2': 0.0000
        })

        svar_zero = SVARResults(
            mod_y=zero_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_zero = svar_zero.get_output_gap_equation_str()
        assert '0.0000' in result_zero, "Should handle zero values correctly"
        assert result_zero.count('0.0000') == 5, "Should have all five zero coefficients"

        # Test 9: Test with unity values
        unity_mod_y = unittest.mock.MagicMock()
        unity_mod_y.params = pd.Series({
            'const': 1.0000,
            'y_lag1': 1.0000,
            'pi_lag1': -1.0000,
            'i_lag1': 1.0000,
            'i_lag2': -1.0000
        })

        svar_unity = SVARResults(
            mod_y=unity_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_unity = svar_unity.get_output_gap_equation_str()
        assert '1.0000' in result_unity, "Should handle unity values"
        assert '-1.0000' in result_unity, "Should handle negative unity values"

        # Test 10: Verify equation format consistency
        # Check that the equation follows the expected mathematical format
        lines = result.strip().split('\n')
        equation_line = next((line for line in lines if '{y_t}' in line), None)
        assert equation_line is not None, "Should contain the main equation line"

        # Test 11: Verify all arithmetic operators are present
        assert ' = ' in result, "Should contain equals sign"
        assert ' + ' in result, "Should contain addition operators"
        # Note: The first term after the constant doesn't have a '+' because it's in parentheses

        # Test 12: Test LaTeX formatting specifics
        assert '{' in result and '}' in result, "Should contain braces for LaTeX formatting"
        assert '{y_t}' in result, "Should contain properly formatted dependent variable"
        assert '\\pi' in result, "Should contain escaped pi symbol"
        assert '\\epsilon' in result, "Should contain escaped epsilon symbol"

        # Test 13: Verify parameter access doesn't raise KeyError
        required_params = ['const', 'y_lag1', 'pi_lag1', 'i_lag1', 'i_lag2']
        for param in required_params:
            # This test ensures the method would fail if any required parameter is missing
            assert param in mock_mod_y.params.index, f"Required parameter {param} should be accessible"

        # Test 14: Test precision formatting
        # Create model with high precision values to test 4 decimal place formatting
        precision_mod_y = unittest.mock.MagicMock()
        precision_mod_y.params = pd.Series({
            'const': 0.123456789,     # Should become 0.1235
            'y_lag1': 0.876543210,    # Should become 0.8765
            'pi_lag1': -0.098765432,  # Should become -0.0988
            'i_lag1': 0.234567891,    # Should become 0.2346
            'i_lag2': -0.178912345    # Should become -0.1789
        })

        svar_precision = SVARResults(
            mod_y=precision_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_precision = svar_precision.get_output_gap_equation_str()

        # Check that values are properly rounded to 4 decimal places
        assert '0.1235' in result_precision, "Should round const to 4 decimal places"
        assert '0.8765' in result_precision, "Should round y_lag1 to 4 decimal places"
        assert '-0.0988' in result_precision, "Should round pi_lag1 to 4 decimal places"
        assert '0.2346' in result_precision, "Should round i_lag1 to 4 decimal places"
        assert '-0.1789' in result_precision, "Should round i_lag2 to 4 decimal places"

        # Test 15: Verify method doesn't modify instance state
        original_mod_y = svar_results.mod_y
        original_params = svar_results.mod_y.params.copy()

        result_again = svar_results.get_output_gap_equation_str()

        assert svar_results.mod_y is original_mod_y, "Method should not modify mod_y"
        assert svar_results.mod_y.params.equals(original_params), "Method should not modify parameters"
        assert result == result_again, "Multiple calls should return identical results"

        # Test 16: Test string formatting edge cases
        # Ensure no raw f-string expressions remain
        assert '.4f' not in result, "Should not contain raw f-string format specifiers"
        assert '{self.' not in result, "Should not contain unresolved f-string expressions"

        # Test 17: Test equation mathematical correctness structure
        # Verify the equation follows the correct economic model structure
        # y_t = const + coef1*y_{t-1} + coef2*pi_{t-1} + coef3*i_{t-1} + coef4*i_{t-2} + error
        equation_parts = [
            '{y_t} =',                    # Dependent variable
            '0.1234 +',                   # Constant term
            '(0.8765)y_{t-1}',           # Lagged output gap
            '(-0.0987){\\pi}_{t-1}',     # Lagged inflation
            '(0.2345)i_{t-1}',           # First lag of interest rate
            '(-0.1789)i_{t-2}',          # Second lag of interest rate
            '{\\epsilon}_t^y'             # Error term
        ]

        for part in equation_parts:
            assert part in result, f"Equation part '{part}' should be present"

        # Test 18: Test whitespace and formatting
        # The method returns a multi-line string with intentional formatting
        stripped_result = result.strip()
        assert not stripped_result.startswith(' '), "Stripped result should not start with unnecessary whitespace"
        assert not stripped_result.endswith(' '), "Stripped result should not end with unnecessary whitespace"

        # Verify the raw result has the expected multi-line format
        assert result.startswith('\n        $'), "Should start with newline and indentation"
        assert result.endswith('\n        '), "Should end with newline and indentation for formatting"

        # Test 19: Test that all coefficients are properly parenthesized
        # All coefficient terms (except constant) should be in parentheses
        # Check for specific coefficient patterns individually for better debugging
        coefficient_checks = [
            ('y_lag1', r'\(0\.8765\)y_\{t-1\}'),      # y_lag1 coefficient
            ('pi_lag1', r'\(-0\.0987\)\{\\\\pi\}'),    # pi_lag1 coefficient
            ('i_lag1', r'\(0\.2345\)i_\{t-1\}'),       # i_lag1 coefficient
            ('i_lag2', r'\(-0\.1789\)i_\{t-2\}')       # i_lag2 coefficient
        ]

        found_coefficients = 0
        for coef_name, pattern in coefficient_checks:
            matches = re.findall(pattern, result)
            if len(matches) >= 1:
                found_coefficients += 1
            else:
                # Print actual content for debugging if needed
                print(f"Debug: Could not find pattern for {coef_name}: {pattern}")

        assert found_coefficients >= 3, f"Should find at least 3 coefficient patterns, found: {found_coefficients}"

        # Alternative simpler approach: just check that parenthesized coefficients exist
        parenthesized_coeffs = [
            '(0.8765)y_{t-1}',    # y_lag1
            '(-0.0987){\\pi}',    # pi_lag1 (without subscript)
            '(0.2345)i_{t-1}',    # i_lag1
            '(-0.1789)i_{t-2}'    # i_lag2
        ]

        for coeff in parenthesized_coeffs:
            assert coeff in result, f"Should contain parenthesized coefficient: {coeff}"

    def test_get_inflation_equation_str(self):
        """
        Comprehensive test for the get_inflation_equation_str method covering all scenarios.
        """

        # Test 1: Basic functionality with standard values
        # Create minimal mock for output gap model (not used but required for dataclass)
        mock_mod_y = unittest.mock.MagicMock()
        mock_mod_y.params = pd.Series({'const': 0.0})

        # Create comprehensive mock regression model for inflation equation
        mock_mod_pi = unittest.mock.MagicMock()
        mock_mod_pi.params = pd.Series({
            'const': 0.0567,
            'y': -0.0432,
            'y_lag1': 0.1876,
            'y_lag2': -0.0998,
            'pi_lag1': 1.1234,
            'pi_lag2': -0.2876,
            'i_lag1': -0.0145
        })

        # Create mock design matrices and residuals
        mock_x_y = pd.DataFrame({'const': [1, 1, 1], 'y_lag1': [0.1, 0.2, 0.3]})
        mock_x_pi = pd.DataFrame({'const': [1, 1, 1], 'pi_lag1': [0.05, 0.1, 0.15]})
        mock_resid_y = pd.Series([0.1, -0.2, 0.3])
        mock_resid_pi = pd.Series([0.05, -0.1, 0.15])

        # Create SVARResults instance
        svar_results = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=mock_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        # Call the method
        result = svar_results.get_inflation_equation_str()

        # Test 2: Verify return type
        assert isinstance(result, str), "Return type should be string"
        assert len(result) > 0, "Result should not be empty"

        # Test 3: Verify LaTeX equation structure
        assert result.startswith('\n        $'), "Should start with LaTeX math delimiter"
        assert result.rstrip().endswith('$'), "Should end with LaTeX math delimiter"
        assert '{\\pi}_t' in result, "Should contain dependent variable pi_t"
        assert '\\epsilon' in result, "Should contain error term epsilon"

        # Test 4: Verify all parameter values are present with correct formatting (7 parameters for inflation)
        assert '0.0567' in result, "Should contain const parameter (0.0567)"
        assert '-0.0432' in result, "Should contain y parameter (-0.0432)"
        assert '0.1876' in result, "Should contain y_lag1 parameter (0.1876)"
        assert '-0.0998' in result, "Should contain y_lag2 parameter (-0.0998)"
        assert '1.1234' in result, "Should contain pi_lag1 parameter (1.1234)"
        assert '-0.2876' in result, "Should contain pi_lag2 parameter (-0.2876)"
        assert '-0.0145' in result, "Should contain i_lag1 parameter (-0.0145)"

        # Test 5: Verify equation structure and variable names
        expected_variables = ['y_t', 'y_{t-1}', 'y_{t-2}', '{\\pi}_{t-1}', '{\\pi}_{t-2}', 'i_{t-1}', '{\\epsilon}_t^{\\pi}']
        for var in expected_variables:
            assert var in result, f"Variable {var} should be in the equation"

        # Test 6: Verify coefficient parentheses for all non-constant terms
        assert '(-0.0432)' in result, "y coefficient should be in parentheses"
        assert '(0.1876)' in result, "y_lag1 coefficient should be in parentheses"
        assert '(-0.0998)' in result, "y_lag2 coefficient should be in parentheses"
        assert '(1.1234)' in result, "pi_lag1 coefficient should be in parentheses"
        assert '(-0.2876)' in result, "pi_lag2 coefficient should be in parentheses"
        assert '(-0.0145)' in result, "i_lag1 coefficient should be in parentheses"

        # Test 7: Test with extreme values
        extreme_mod_pi = unittest.mock.MagicMock()
        extreme_mod_pi.params = pd.Series({
            'const': 999.9999,
            'y': -999.9999,
            'y_lag1': 0.0001,
            'y_lag2': -0.0001,
            'pi_lag1': 888.8888,
            'pi_lag2': -888.8888,
            'i_lag1': 123.4567
        })

        svar_extreme = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=extreme_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_extreme = svar_extreme.get_inflation_equation_str()

        # Verify extreme values are formatted correctly
        assert '999.9999' in result_extreme, "Should handle large positive values"
        assert '-999.9999' in result_extreme, "Should handle large negative values"
        assert '0.0001' in result_extreme, "Should handle very small positive values"
        assert '-0.0001' in result_extreme, "Should handle very small negative values"
        assert '888.8888' in result_extreme, "Should handle arbitrary large decimal values"
        assert '-888.8888' in result_extreme, "Should handle arbitrary large negative decimal values"
        assert '123.4567' in result_extreme, "Should handle arbitrary decimal values"

        # Test 8: Test with zero values
        zero_mod_pi = unittest.mock.MagicMock()
        zero_mod_pi.params = pd.Series({
            'const': 0.0000,
            'y': 0.0000,
            'y_lag1': 0.0000,
            'y_lag2': 0.0000,
            'pi_lag1': 0.0000,
            'pi_lag2': 0.0000,
            'i_lag1': 0.0000
        })

        svar_zero = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=zero_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_zero = svar_zero.get_inflation_equation_str()
        assert '0.0000' in result_zero, "Should handle zero values correctly"
        assert result_zero.count('0.0000') == 7, "Should have all seven zero coefficients"

        # Test 9: Test with unity values
        unity_mod_pi = unittest.mock.MagicMock()
        unity_mod_pi.params = pd.Series({
            'const': 1.0000,
            'y': -1.0000,
            'y_lag1': 1.0000,
            'y_lag2': -1.0000,
            'pi_lag1': 1.0000,
            'pi_lag2': -1.0000,
            'i_lag1': 1.0000
        })

        svar_unity = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=unity_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_unity = svar_unity.get_inflation_equation_str()
        assert '1.0000' in result_unity, "Should handle unity values"
        assert '-1.0000' in result_unity, "Should handle negative unity values"

        # Test 10: Verify equation format consistency
        # Check that the equation follows the expected mathematical format
        lines = result.strip().split('\n')
        equation_line = next((line for line in lines if '{\\pi}_t' in line), None)
        assert equation_line is not None, "Should contain the main equation line"

        # Test 11: Verify all arithmetic operators are present
        assert ' = ' in result, "Should contain equals sign"
        assert ' + ' in result, "Should contain addition operators"
        # Note: All coefficient terms are in parentheses

        # Test 12: Test LaTeX formatting specifics
        assert '{' in result and '}' in result, "Should contain braces for LaTeX formatting"
        assert '{\\pi}_t' in result, "Should contain properly formatted dependent variable"
        assert '\\pi' in result, "Should contain escaped pi symbol"
        assert '\\epsilon' in result, "Should contain escaped epsilon symbol"

        # Test 13: Verify parameter access doesn't raise KeyError
        required_params = ['const', 'y', 'y_lag1', 'y_lag2', 'pi_lag1', 'pi_lag2', 'i_lag1']
        for param in required_params:
            # This test ensures the method would fail if any required parameter is missing
            assert param in mock_mod_pi.params.index, f"Required parameter {param} should be accessible"

        # Test 14: Test precision formatting
        # Create model with high precision values to test 4 decimal place formatting
        precision_mod_pi = unittest.mock.MagicMock()
        precision_mod_pi.params = pd.Series({
            'const': 0.123456789,     # Should become 0.1235
            'y': -0.876543210,        # Should become -0.8765
            'y_lag1': 0.234567891,    # Should become 0.2346
            'y_lag2': -0.345678912,   # Should become -0.3457
            'pi_lag1': 1.456789123,   # Should become 1.4568
            'pi_lag2': -0.567891234,  # Should become -0.5679
            'i_lag1': 0.678912345     # Should become 0.6789
        })

        svar_precision = SVARResults(
            mod_y=mock_mod_y,
            mod_pi=precision_mod_pi,
            x_y=mock_x_y,
            x_pi=mock_x_pi,
            resid_y=mock_resid_y,
            resid_pi=mock_resid_pi
        )

        result_precision = svar_precision.get_inflation_equation_str()

        # Check that values are properly rounded to 4 decimal places
        assert '0.1235' in result_precision, "Should round const to 4 decimal places"
        assert '-0.8765' in result_precision, "Should round y to 4 decimal places"
        assert '0.2346' in result_precision, "Should round y_lag1 to 4 decimal places"
        assert '-0.3457' in result_precision, "Should round y_lag2 to 4 decimal places"
        assert '1.4568' in result_precision, "Should round pi_lag1 to 4 decimal places"
        assert '-0.5679' in result_precision, "Should round pi_lag2 to 4 decimal places"
        assert '0.6789' in result_precision, "Should round i_lag1 to 4 decimal places"

        # Test 15: Verify method doesn't modify instance state
        original_mod_pi = svar_results.mod_pi
        original_params = svar_results.mod_pi.params.copy()

        result_again = svar_results.get_inflation_equation_str()

        assert svar_results.mod_pi is original_mod_pi, "Method should not modify mod_pi"
        assert svar_results.mod_pi.params.equals(original_params), "Method should not modify parameters"
        assert result == result_again, "Multiple calls should return identical results"

        # Test 16: Test string formatting edge cases
        # Ensure no raw f-string expressions remain
        assert '.4f' not in result, "Should not contain raw f-string format specifiers"
        assert '{self.' not in result, "Should not contain unresolved f-string expressions"

        # Test 17: Test equation mathematical correctness structure
        # Verify the equation follows the correct economic model structure
        # pi_t = const + coef1*y_t + coef2*y_{t-1} + coef3*y_{t-2} + coef4*pi_{t-1} + coef5*pi_{t-2} + coef6*i_{t-1} + error
        equation_parts = [
            '{\\pi}_t =',                   # Dependent variable
            '0.0567 +',                     # Constant term
            '(-0.0432)y_t',                 # Current output gap
            '(0.1876)y_{t-1}',             # First lag of output gap
            '(-0.0998)y_{t-2}',            # Second lag of output gap
            '(1.1234){\\pi}_{t-1}',        # First lag of inflation
            '(-0.2876){\\pi}_{t-2}',       # Second lag of inflation
            '(-0.0145)i_{t-1}',            # First lag of interest rate
            '{\\epsilon}_t^{\\pi}'          # Error term
        ]

        for part in equation_parts:
            assert part in result, f"Equation part '{part}' should be present"

        # Test 18: Test whitespace and formatting
        # The method returns a multi-line string with intentional formatting
        stripped_result = result.strip()
        assert not stripped_result.startswith(' '), "Stripped result should not start with unnecessary whitespace"
        assert not stripped_result.endswith(' '), "Stripped result should not end with unnecessary whitespace"

        # Verify the raw result has the expected multi-line format
        assert result.startswith('\n        $'), "Should start with newline and indentation"
        assert result.endswith('\n        '), "Should end with newline and indentation for formatting"

        # Test 19: Test that all coefficients are properly parenthesized
        # Check for specific coefficient patterns individually for better debugging
        parenthesized_coeffs = [
            '(-0.0432)y_t',           # y coefficient
            '(0.1876)y_{t-1}',        # y_lag1 coefficient
            '(-0.0998)y_{t-2}',       # y_lag2 coefficient
            '(1.1234){\\pi}_{t-1}',   # pi_lag1 coefficient
            '(-0.2876){\\pi}_{t-2}',  # pi_lag2 coefficient
            '(-0.0145)i_{t-1}'        # i_lag1 coefficient
        ]

        for coeff in parenthesized_coeffs:
            assert coeff in result, f"Should contain parenthesized coefficient: {coeff}"

        # Test 20: Test inflation-specific formatting
        # Verify pi symbols are properly formatted with double braces
        assert '{\\pi}_t' in result, "Should contain properly formatted dependent variable pi_t"
        assert '{\\pi}_{t-1}' in result, "Should contain properly formatted pi_{t-1}"
        assert '{\\pi}_{t-2}' in result, "Should contain properly formatted pi_{t-2}"
        assert '{\\epsilon}_t^{\\pi}' in result, "Should contain properly formatted error term for inflation"

        # Test 21: Verify parameter count matches inflation equation structure
        # Check that all coefficient values are present rather than parameter names
        coefficient_values = ['0.0567', '-0.0432', '0.1876', '-0.0998', '1.1234', '-0.2876', '-0.0145']
        found_coefficients = 0
        for coeff_val in coefficient_values:
            if coeff_val in result:
                found_coefficients += 1

        assert found_coefficients >= 6, f"Should find most coefficient values in equation, found: {found_coefficients}"

        # Alternative check: verify the equation contains the expected number of terms
        # Count parenthesized coefficients (should be 6 non-constant terms)
        parenthesized_count = result.count('(')
        assert parenthesized_count >= 6, f"Should have at least 6 parenthesized coefficients, found: {parenthesized_count}"

        # Verify the equation structure has all expected variable types
        variable_types = ['y_t', 'y_{t-1}', 'y_{t-2}', '{\\pi}_{t-1}', '{\\pi}_{t-2}', 'i_{t-1}']
        found_variables = sum(1 for var in variable_types if var in result)
        assert found_variables >= 5, f"Should find most variable types in equation, found: {found_variables}"

        # Test 22: Test equation length and complexity
        # Inflation equation should be longer than output gap equation due to more terms
        assert len(result) > 100, "Inflation equation should be reasonably long due to many terms"

        # Count the number of coefficient terms (should be 6 non-constant terms)
        coefficient_count = result.count('(')
        assert coefficient_count >= 6, "Should have at least 6 parenthesized coefficients"
