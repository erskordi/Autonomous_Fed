"""
Module defining data structures for the linear SVAR environment.
"""

from dataclasses import dataclass
import statsmodels.api as sm
import pandas as pd

@dataclass(frozen=True)
class SVARResults:
    """
    Container for the linear SVAR (recursive OLS) results.

    Attributes:
        mod_y (statsmodels.regression.linear_model.RegressionResultsWrapper) Fitted OLS model for the output-gap equation.
        mod_pi (statsmodels.regression.linear_model.RegressionResultsWrapper) Fitted OLS model for the inflation equation.
        x_y (pd.DataFrame) Design matrix used in the output-gap regression.
        x_pi (pd.DataFrame) Design matrix used in the inflation regression.
        resid_y (pd.Series) Residuals (innovations) from the output-gap regression.
        resid_pi (pd.Series) Residuals (innovations) from the inflation regression.

    Methods:
        print_results() -> str:
            Print a summary of the SVAR results.
        get_comparison_table_str() -> str:
            Print a comparison table of coefficients from both equations.
        get_output_gap_equation_str() -> str:
            Get the output gap equation as a formatted string.
        get_inflation_equation_str() -> str:
            Get the inflation equation as a formatted string.
    """
    mod_y: sm.regression.linear_model.RegressionResultsWrapper
    mod_pi: sm.regression.linear_model.RegressionResultsWrapper
    x_y: pd.DataFrame
    x_pi: pd.DataFrame
    resid_y: pd.Series
    resid_pi: pd.Series

    # Public Methods
    def print_results(self) -> str:
        """
        Print a summary of the SVAR results.

        Args:
            None

        Returns:
            str: Summary string.

        Raises:
            None
        """
        print("\n=== Output gap equation ===")
        print(self.mod_y.summary())
        print("\n=== Inflation equation ===")
        print(self.mod_pi.summary())

        # Sanity Checks:
        print("\nResidual std (y, pi):", round(self.resid_y.std(),3), round(self.resid_pi.std(),3))
        print("Samples used (y, pi):", self.mod_y.nobs, self.mod_pi.nobs)
        return ""

    # Properties
    @property
    def output_gap_equation_str(self) -> str:
        """
        Get the output gap equation as a formatted string.

        Args:
            None

        Returns:
            str: Output gap equation string.

        Raises:
            None

        Note:
            The return string should be wrapped in a markdown environment for proper display.
        """

        y_t_eq_str = fr"""
        ${{y_t}} = {self.mod_y.params['const']:.4f} + ({self.mod_y.params['y_lag1']:.4f})y_{{t-1}} + ({self.mod_y.params['pi_lag1']:.4f}){{\pi}}_{{t-1}} + ({self.mod_y.params['i_lag1']:.4f})i_{{t-1}} + ({self.mod_y.params['i_lag2']:.4f})i_{{t-2}} + {{\epsilon}}_t^y$
        """ # pylint: disable=line-too-long

        return y_t_eq_str

    @property
    def inflation_equation_str(self) -> str:
        """
        Get the inflation equation as a formatted string.

        Args:
            None

        Returns:
            str: Inflation equation string.

        Raises:
            None

        Note:
            The return string should be wrapped in a markdown environment for proper display.
        """

        pi_t_eq_str = fr"""
        ${{\pi}}_t = {self.mod_pi.params['const']:.4f} + ({self.mod_pi.params['y']:.4f})y_t + ({self.mod_pi.params['y_lag1']:.4f})y_{{t-1}} + ({self.mod_pi.params['y_lag2']:.4f})y_{{t-2}} + ({self.mod_pi.params['pi_lag1']:.4f}){{\pi}}_{{t-1}} + ({self.mod_pi.params['pi_lag2']:.4f}){{\pi}}_{{t-2}} + ({self.mod_pi.params['i_lag1']:.4f})i_{{t-1}} + {{\epsilon}}_t^{{\pi}}$
        """ # pylint: disable=line-too-long

        return pi_t_eq_str
