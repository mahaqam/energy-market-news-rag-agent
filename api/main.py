from pathlib import Path
import sys
import pandas as pd
from fastapi import FastAPI
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from assistant import EnergyNewsAssistant
assistant=EnergyNewsAssistant(pd.read_csv(ROOT/"data/demo_corpus.csv"))
app=FastAPI(title="Energy Market News Research API")
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/search")
def search(q:str,k:int=5): return assistant.run(q,max(1,min(k,10)),llm=False)
