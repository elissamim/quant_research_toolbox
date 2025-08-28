import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import TypedDict, Any, Dict

from utils import load_ticker_data, compute_returns, compute_cumulative_returns
from strategies import Momentum, MeanReversion
from risk import ValueAtRisk, Drawdown, DownsideRisk, TailRisk
from risk_adjusted import sharpe_ratio, sortino_ratio, omega_ratio

class RiskState(TypedDict):
    risk: Dict[str, Any]

# Add llm here
llm = ChatOpenAI(temperature=0)

tools = [
    Momentum,
    MeanReversion,
    ValueAtRisk,
    Drawdown, 
    TailRisk,
    sharpe_ratio,
    sortino_ratio,
    omega_ratio
]

risk_graph = StateGraph(RiskState)
risk_graph.add_node("tools", tools)
risk_graph.add_edge(START, "tools")
risk_graph.add_edge("tools", END)
risk_compiled_graph = risk_graph.compile()