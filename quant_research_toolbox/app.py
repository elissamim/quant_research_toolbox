from fastapi import FastAPI
from pydantic import BaseModel
from agent_risk import compiled_graph

app = FastAPI()

class QueryRequest(BaseModel):
    query:str

@app.post("/run")
def run_strategy(req:QueryRequest):
    result = compiled_graph.invoke({"query":req.query})
    return {k:v for k,v in result.items() if k not in {"query", "data", "returns", "cumulative_returns"}}