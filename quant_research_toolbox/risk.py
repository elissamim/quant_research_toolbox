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

    confidence_level: float = 0.99
    alpha: float = field(init=False)

    def __post_init__(self):
        if not (0 <= self.confidence_level <= 1):
            raise ValueError("`confidence_level` should be between 0 and 1")
        self.alpha = 1 - self.confidence_level

    def historical_var(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Return the potential loss for a given confidence interval, given
        the distribution of historical returns. It involves sorting the historical returns
        and finding the percentile that corresponds to the desired confidence level.

        Args:
            returns (pd.Series): Returns series.

        Returns:
            float: Historical VaR and Historical CVaR.
        """

        var = np.percentile(returns, 100 * self.alpha)

        cvar = returns[returns <= var].mean()

        return float(var), float(cvar)

    def parametric_var(self, returns: pd.Series) -> Tuple[float, float]:
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
        var = mean_return + z_score * std_dev_return
        cvar = mean_return + std_dev_return * norm.pdf(z_score) / self.alpha

        return float(var), float(cvar)

    def monte_carlo_var(
        self,
        returns: pd.Series,
        num_simulations: Optional[int] = 1000,
        simulation_horizon: Optional[int] = 252,
        initial_investment: Optional[float] = 1e6,
    ) -> Tuple[float, float]:
        """
        Return the potential loss for a given confidence interval, using
        random sampling to simulate a range of potential outcomes based on historical data.

        Args:
            returns (pd.Series): Returns series.

        Returns:
            float: Monte-Carlo VaR and Monte-Carlo CVaR.
        """

        simulated_returns = np.random.normal(
            np.mean(returns), np.std(returns), (simulation_horizon, num_simulations)
        )

        portfolio_values = initial_investment * np.exp(
            np.cumsum(simulated_returns, axis=0)
        )

        portfolio_returns = portfolio_values[-1] / portfolio_values[0] - 1

        var = np.percentile(portfolio_returns, 100 * self.alpha)

        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return var, cvar


class Drawdown:
    """
    A class for computing Daily Drawdown and Maximum Drawdown of a strategy,
    given its cumulative returns.

    Examples:
        >>> from quant_research_toolbox.risk import Drawdown
        >>> data = pd.Series([.01, .05, -0.2, .01, .1, .2, -0.05])
        >>> dd = Drawdown()
        >>> dd.max_drawdown(data)
        -0.23809523809523814
        >>> dd.average_drawdown(data)
        -0.22321428571428575
    """

    @staticmethod
    def daily_drawdown(cumulative_returns: pd.Series) -> pd.Series:
        """
        Return the daily drawdown for a series of cumulative returns.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            pd.Series: Series of daily drawdown.
        """

        nav = 1 + cumulative_returns
        highwatermark = nav.cummax()
        return nav / highwatermark - 1

    @staticmethod
    def max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Return the max drawdown of a series of cumulative returns.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            float: Maximum Drawdown of the strategy.
        """

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        return float(daily_drawdowns.min())

    @staticmethod
    def average_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Return the average drawdown of a series of cumulative returns,
        by computing the average of the max drawdowns on each period of drawdown.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            float: Average Drawdown of the strategy.
        """

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        in_drawdowns = daily_drawdowns < 0

        current_drawdowns = []
        max_drawdowns = []

        for drawdown, in_drawdown in zip(daily_drawdowns, in_drawdowns):
            if in_drawdown:
                current_drawdowns.append(drawdown)
            elif current_drawdowns:
                max_drawdowns.append(min(current_drawdowns))
                current_drawdowns = []

        if current_drawdowns:
            max_drawdowns.append(min(current_drawdowns))

        if max_drawdowns:
            return float(np.mean(max_drawdowns))
        return 0.0

    @staticmethod
    def drawdown_duration(
        cumulative_returns: pd.Series, duration_stat: Optional[str] = "max"
    ) -> float:
        """
        Returns the drawdown duration which is the duration from peak to peak of NAV.
        This duration includes all recovered drawdown periods, but does not include
        in-progress drawdown periods, at the end of the series for example.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.
            duration_stat (Optional[str]): Statistic to compute for drawdown duration.
                                           One of `max`, `mean`, `median`. Defaults to `max`.

        Returns:
            float: Given statistic of duration of drawdown periods.
        """

        DURATION_STATS = {"max": np.max, "mean": np.mean, "median": np.median}

        if duration_stat not in DURATION_STATS:
            raise ValueError(
                f"Invalid value for `duration_stat` : '{duration_stat}'. Must be one of {list(DURATION_STATS)}."
            )

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        in_drawdowns = daily_drawdowns < 0

        current_drawdown_duration = 0
        drawdown_durations = []

        for daily_drawdown, in_drawdown in zip(daily_drawdowns, in_drawdowns):
            if in_drawdown:
                current_drawdown_duration += 1
            elif current_drawdown_duration > 0:
                drawdown_durations.append(current_drawdown_duration)
                current_drawdown_duration = 0

        if drawdown_durations:
            return float(DURATION_STATS[duration_stat](drawdown_durations))
        return 0.0

    @staticmethod
    def count_drawdown_periods(cumulative_returns:pd.Series) -> int:
        """
        Count the number of drawdown periods in cumulative returns series.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            int: Number of drawdown periods.
        """

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        in_drawdowns = daily_drawdowns < 0

        count_drawdowns = 0
        previous_drawdown = False

        for in_drawdown in in_drawdowns:
            if in_drawdown:
                if not previous_drawdown:
                    count_drawdowns +=1
                    previous_drawdown = True
            elif previous_drawdown:
                previous_drawdown = False


        return count_drawdowns

    @staticmethod
    def time_under_water(cumulative_returns:pd.Series,
                         duration_stat: Optional[str] = "max") -> int:
        """
        Return the time under water with a given statistic : this duration corresponds
        to all drawdown durations recovered and in-progress.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.
            duration_stat (Optional[str]): Statistic to compute for drawdown duration.
                                           One of `max`, `mean`, `median`. Defaults to `max`.

        Returns:
            float: Given statistic of duration of time under water periods.
        """

        DURATION_STATS = {"max": np.max, "mean": np.mean, "median": np.median}

        if duration_stat not in DURATION_STATS:
            raise ValueError(
                f"Invalid value for `duration_stat` : '{duration_stat}'. Must be one of {list(DURATION_STATS)}."
            )

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        in_drawdowns = daily_drawdowns < 0

        current_drawdown_duration = 0
        drawdown_durations = []

        for daily_drawdown, in_drawdown in zip(daily_drawdowns, in_drawdowns):
            if in_drawdown:
                current_drawdown_duration += 1
            elif current_drawdown_duration > 0:
                drawdown_durations.append(current_drawdown_duration)
                current_drawdown_duration = 0

        # Here relies the difference with drawdown_duration : we also add in-progress drawdowns
        if current_drawdown_duration > 0:
            drawdown_durations.append(current_drawdown_duration)

        if drawdown_durations:
            return float(DURATION_STATS[duration_stat](drawdown_durations))
        return 0.0

    @staticmethod
    def ulcer_index(cumulative_returns:pd.Series) -> float:
        """
        Return the Ulcer-Index of a series of cumulative returns.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.

        Returns:
            float: Ulcer-Index of the series.
        """

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        is_negative = daily_drawdowns[daily_drawdowns < 0]

        if is_negative.empty:
            return 0.0

        ui = float(np.sqrt((is_negative ** 2).mean()))

        return ui

    @staticmethod
    def conditional_drawdown_at_risk(cumulative_returns:pd.Series,
                                    confidence_level:Optional[float]=.99
                                    ) -> float:
        """
        Compute the conditional drawdown at risk for a given confidence level.
        It is the CVaR for the drawdown series.

        Args:
            cumulative_returns (pd.Series): Series of cumulative returns.
            confidence_level (Optional[float]): Confidence level of the CDaR.

        Returns:
            float: CDaR, i.e. the CVaR of the drawdown series.
        """

        daily_drawdowns = Drawdown.daily_drawdown(cumulative_returns)
        daily_drawdowns = daily_drawdowns[daily_drawdowns < 0]

        if daily_drawdowns.empty:
            return 0.0

        if (confidence_level > 1) or (confidence_level < 0):
            raise ValueError(
                f"Invalid value for the `confidence_level`:{confidence_level}. This variable must be between 0 and 1."
            )
        
        value_at_risk = np.percentile(daily_drawdowns, 
                                      100*(1-confidence_level))
        conditional_value_at_risk = (
            daily_drawdowns[daily_drawdowns <= value_at_risk].mean()
        )

        return float(conditional_value_at_risk)