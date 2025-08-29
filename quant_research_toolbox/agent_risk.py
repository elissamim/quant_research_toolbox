# import os
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from typing import TypedDict, Any, Dict, Literal
import json

from utils import load_ticker_data
from strategies import Momentum, MeanReversion

from risk_adjusted import sharpe_ratio, sortino_ratio, omega_ratio

# ------------- State -------------
class StrategyState(TypedDict):
    query: str
    ticker: str
    start_date: str
    end_date: str
    strategy: Literal["sma_crossover", "naive_momentum", "sma_mean_reversion"]
    data: Any
    returns: Any
    cumulative_returns: Any
    performance: Dict[str, float]


# ------------- Nodes -------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def parse_query(state: StrategyState) -> StrategyState:
    """
    Use the LLM to parse the query and identify the ticker of interest,
    the start and end dates for the strategy backtest, the type of
    strategy to apply on the ticker.
    """

    prompt = f"""
    Extract the following field from the user query:
    - ticker (stock symbol, e.g. "AAPL")
    - start_date ("YYYY-MM-DD")
    - end_date ("YYYY-MM-DD")
    - strategy ("sma_crossover", "naive_momentum", "sma_mean_reversion")

    User query : {state["query"]}
    Return JSON only (structured generation), with keys ticker, start_date, end_date, strategy
    """

    structured = llm.invoke(prompt).content
    parsed = json.loads(structured)

    return {
        "ticker": parsed["ticker"],
        "start_date": parsed["start_date"],
        "end_date": parsed["end_date"],
        "strategy": parsed["strategy"],
    }


def load_data(state: StrategyState) -> StrategyState:
    """
    Load the data for the given ticker and given start and en dates.
    """

    df_ticker = load_ticker_data(
        state["ticker"], state["start_date"], state["end_date"]
    )

    return {"data": df_ticker}


def apply_strategy(state: StrategyState) -> StrategyState:
    """
    Apply the given strategy to compute the returns and cumulative returns of the
    chosen strategy for the given ticker, start and end dates.
    """

    df_ticker = state["data"]

    if state["strategy"] == "sma_crossover":
        df_strategy = Momentum.sma_crossover(df_ticker, "ohlc_price")
    elif state["strategy"] == "naive_momentum":
        df_strategy = Momentum.naive_momentum(df_ticker, "ohlc_price")
    elif state["strategy"] == "sma_mean_reversion":
        df_strategy = MeanReversion.sma_mean_reversion(df_ticker, "ohlc_price")
    else:
        raise ValueError(f"Unknown strategy {state['strategy']}")

    return {
        "returns": df_strategy["returns"],
        "cumulative_returns": df_strategy["cumulative_returns"],
    }


def compute_performance(state: StrategyState) -> StrategyState:
    """
    Compute the performance for the chosen strategy on the chosen ticker and dates.
    """

    df_returns = state["returns"]

    performance = {
        "sharpe_ratio": sharpe_ratio(df_returns),
        "sortino_ratio": sortino_ratio(df_returns),
        "omega_ratio": omega_ratio(df_returns),
    }

    return {"performance": performance}


# ------------- Graph -------------
graph = StateGraph(StrategyState)

graph.add_node("parse_query", parse_query)
graph.add_node("load_data", load_data)
graph.add_node("apply_strategy", apply_strategy)
graph.add_node("compute_performance", compute_performance)

graph.add_edge(START, "parse_query")
graph.add_edge("parse_query", "load_data")
graph.add_edge("load_data", "apply_strategy")
graph.add_edge("apply_strategy", "compute_performance")
graph.add_edge("compute_performance", END)

compiled_graph = graph.compile()

# ------------- Example run -------------
if __name__ == "__main__":
    query = "What is the performance of a SMA crossover strategy applied on AAPL from 1st december 2021 to 29th january 2024 ?"
    result = compiled_graph.invoke({"query": query})
    print(result["performance"])