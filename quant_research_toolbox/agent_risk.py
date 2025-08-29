import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import TypedDict, Any, Dict, Literal
import json

from utils import load_ticker_data
from strategies import Momentum, MeanReversion
from risk import ValueAtRisk, Drawdown, DownsideRisk, TailRisk
from risk_adjusted import sharpe_ratio, sortino_ratio, omega_ratio


# ------------- State -------------  
class StrategyState(TypedDict):
    query: str
    ticker: str
    start_date: str
    end_date: str
    strategy: Literal["sma_crossover", "naive_momentum", "sma_mean_reversion"]
    returns: Any
    performance: Dict[str, float]

# ------------- Nodes -------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def parse_query(state:StrategyState) -> StrategyState:
    """
    Use the LLM to parse the query and identify the ticker of interest,
    the start and end dates for the strategy backtest, the type of 
    strategy to apply on the ticker.
    """

    prompt = f"""
    Extract the following field from the user query:
    - ticker (stock symbol, e.g. AAPL)
    - start_date (YYYY-MM-DD)
    - end_date (YYYY-MM-DD)
    - strategy ()
    """

    structured = llm.invoke(prompt).content
    parsed = json.loads(structured)

    return {
        "ticker":parsed["ticker"],
        "start_date":parsed["start_date"],
        "end_date":parsed["end_date"],
        "strategy":parsed["strategy"]
    }

# ------------- Graph -------------   















