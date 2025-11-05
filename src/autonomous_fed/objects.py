"""
Module defining data structures for the linear SVAR environment.
"""
from dataclasses import dataclass
import statsmodels.api as sm #pragma: no cover
from statsmodels.stats.stattools import durbin_watson #pragma: no cover
from statsmodels.stats.diagnostic import acorr_breusch_godfrey #pragma: no cover
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

    def get_comparison_table_str(self) -> str:
        """
        Print a comparison table of coefficients from both equations.

        Args:
            None

        Returns:
            str: Comparison table string.

        Raises:
            None

        Note:
            The return table should be wrapped in a markdown environment for proper display.
        """

        table_nine_str = fr"""
        | Parameters                     | Bundesbank Estimate | p-Value | Local OLS Estimate                             | p-Value                                        |
        |--------------------------------|---------------------|---------|------------------------------------------------|------------------------------------------------|
        | **Output gap equation**        |                     |         |                                                |                                                |
        | $C^y$                          | 0.3834              | 0.0351  | {self.mod_y.params['const']:.4f}               | {self.mod_y.pvalues['const']:.4f}              |
        | $a^y_{{y,1}}$                  | 0.9084              | 0.0000  | {self.mod_y.params['y_lag1']:.4f}              | {self.mod_y.pvalues['y_lag1']:.4f}             |
        | $a^y_{{\pi,1}}$                | -0.1437             | 0.1409  | {self.mod_y.params['pi_lag1']:.4f}             | {self.mod_y.pvalues['pi_lag1']:.4f}            |
        | $a^y_{{i,1}}$                  | 0.2726              | 0.0661  | {self.mod_y.params['i_lag1']:.4f}              | {self.mod_y.pvalues['i_lag1']:.4f}             |
        | $a^y_{{i,2}}$                  | -0.2896             | 0.0313  | {self.mod_y.params['i_lag2']:.4f}              | {self.mod_y.pvalues['i_lag2']:.4f}             |
        | $\bar{{R}}^2$                  | 0.9100              |         | {self.mod_y.rsquared:.4f}                      |                                                |
        | MSE                            | 0.2108              |         | {self.mod_y.mse_resid:.4f}                     |                                                |
        | $\hat\sigma_{{\varepsilon_1}}$ | 0.2136              |         | {self.mod_y.scale ** 0.5:.4f}                  |                                                |
        | DW                             | 1.8206              |         | {durbin_watson(self.mod_y.resid):.4f}          |                                                |
        | LM(1)                          | 3.1037              | 0.1644  | {acorr_breusch_godfrey(self.mod_y, 1)[2]:.4f}  | {acorr_breusch_godfrey(self.mod_y, 1)[3]:.4f}  |
        | **Inflation equation**         |                     |         |                                                |                                                |
        | $C^\pi$                        | 0.1035              | 0.1659  | {self.mod_pi.params['const']:.4f}              | {self.mod_pi.pvalues['const']:.4f}             |
        | $a^\pi_{{y,0}}$                | -0.0655             | 0.1578  | {self.mod_pi.params['y']:.4f}                  | {self.mod_pi.pvalues['y']:.4f}                 |
        | $a^\pi_{{y,1}}$                | 0.1970              | 0.0048  | {self.mod_pi.params['y_lag1']:.4f}             | {self.mod_pi.pvalues['y_lag1']:.4f}            |
        | $a^\pi_{{y,2}}$                | -0.1121             | 0.0163  | {self.mod_pi.params['y_lag2']:.4f}             | {self.mod_pi.pvalues['y_lag2']:.4f}            |
        | $a^\pi_{{\pi,1}}$              | 1.2970              | 0.0000  | {self.mod_pi.params['pi_lag1']:.4f}            | {self.mod_pi.pvalues['pi_lag1']:.4f}           |
        | $a^\pi_{{\pi,2}}$              | -0.3116             | 0.0076  | {self.mod_pi.params['pi_lag2']:.4f}            | {self.mod_pi.pvalues['pi_lag2']:.4f}           |
        | $a^\pi_{{i,1}}$                | -0.0122             | 0.4174  | {self.mod_pi.params['i_lag1']:.4f}             | {self.mod_pi.pvalues['i_lag1']:.4f}            |
        | $\bar{{R}}^2$                  | 0.9450              |         | {self.mod_pi.rsquared:.4f}                     |                                                |
        | MSE                            | 0.0326              |         | {self.mod_pi.mse_resid:.4f}                    |                                                |
        | $\hat\sigma_{{\varepsilon_2}}$ | 0.0330              |         | {self.mod_pi.scale ** 0.5:.4f}                 |                                                |
        | DW                             | 2.1095              |         | {durbin_watson(self.mod_pi.resid):.4f}         |                                                |
        | LM(1)                          | 2.5542              | 0.0696  | {acorr_breusch_godfrey(self.mod_pi, 1)[2]:.4f} | {acorr_breusch_godfrey(self.mod_pi, 1)[3]:.4f} |

        \* Residual std dev from OLS output."""

        return table_nine_str

    def get_output_gap_equation_str(self) -> str:
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

    def get_inflation_equation_str(self) -> str:
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
