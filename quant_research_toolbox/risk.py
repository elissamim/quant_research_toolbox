import pandas as pd
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class ValueAtRisk:
    """
    A class for computing VaR and CVaR for a series of returns.

    Args:
        confidence_level (float): The confidence level of the VaR.

    Attributes:
        alpha (float): The probability that losses are higher than VaR.

    Examples:
        >>> from quant_research_toolbox.risk import ValueAtRisk
        >>> data = pd.Series([100, -100, 10, -10, -20, -12, -10, -100, -1000, 5])
        >>> var = ValueAtRisk(.99)
        >>> var.historical_var(data)
        -919
    """

    confidence_level: float = .99
    alpha : float = field(init=False)

    def __post_init__(self):
        if not (0<=self.confidence_level<=1):
            raise ValueError("`confidence_level` should be between 0 and 1")
        self.alpha = 1 - self.confidence_level

    def historical_var(self,
                      returns:pd.Series) -> Tuple[float, float]:
        """
        Return the potential loss for a given confidence interval, given
        the distribution of historical returns. It involves sorting the historical returns
        and finding the percentile that corresponds to the desired confidence level.

        Args:
            returns (pd.Series): Returns series.

        Returns:
            float: Historical VaR and Historical CVaR.
        """

        var = np.percentile(returns,
                            100*self.alpha)

        cvar = returns[returns <= var].mean()
        
        return float(var), float(cvar)
        
    def parametric_var(self,
                      returns:pd.Series) -> Tuple[float, float]:
        """
        Return the potential loss for a given confidence interval, given
        that the distribution of the returns is assumed normal. Calculate VaR
        using the mean and standard deviation of the portfolio's returns.

        Args:
            returns (pd.Series): Returns series.

        Returns:
            float : Parametric VaR and Parametric CVaR.
        """

        z_score = norm.ppf(self.alpha)
        mean_return = np.mean(returns)
        std_dev_return = np.std(returns)
        var = mean_return + z_score*std_dev_return
        cvar = mean_return + std_dev_return*norm.pdf(z_score)/self.alpha

        return float(var), float(cvar)

    def monte_carlo_var(self,
                       returns:pd.Series,
                       num_simulations:Optional[int]=1000,
                       simulation_horizon:Optional[int]=252,
                       initial_investment:Optional[float]=1e6) -> Tuple[float, float]:
        """
        Return the potential loss for a given confidence interval, using
        random sampling to simulate a range of potential outcomes based on historical data.

        Args:
            returns (pd.Series): Returns series.

        Returns:
            float: Monte-Carlo VaR and Monte-Carlo CVaR.
        """

        simulated_returns = (
            np.random.normal(
                np.mean(returns),
                np.std(returns),
                (simulation_horizon, num_simulations)
            )
        )

        portfolio_values = (
            initial_investment*np.exp(np.cumsum(simulated_returns,
                                               axis=0))
        )

        portfolio_returns = (
            portfolio_values[-1]/portfolio_values[0]-1
        )

        var = np.percentile(portfolio_returns,
                            100*self.alpha)

        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return var, cvar

class Drawdown:
    """
    A class for computing Daily Drawdown and Maximum Drawdown of a strategy,
    given its cumulative returns.

    Examples:
        >>> from quant_research_toolbox.risk import Drawdown
        >>> data = pd.Series([100, 110, 120, 98, 97, 99, 102])
        >>> dd = Drawdown()
        >>> dd.max_drawdown(data)
        -0.192
    """

    @staticmethod
    def daily_drawdown(cumulative_returns:pd.Series)->pd.Series:
        """
        Return the daily drawdown for a series of cumulative returns.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            pd.Series: Series of daily drawdown.
        """
    
        running_max = cumulative_returns.cummax()
        return cumulative_returns/running_max -1
         
    @staticmethod
    def max_drawdown(cumulative_returns:pd.Series)->float:
        """
        Return the max drawdown of a series of cumulative returns.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            float: Maximum Drawdown of the strategy. 
        """
    
        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        return float(daily_drawdowns.min())