from pathlib import Path
import os, sys
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from assistant import EnergyNewsAssistant

st.set_page_config(page_title="Energy Market News Research Assistant",layout="wide")
st.title("Energy Market News RAG & Agentic Research Assistant")
st.caption("Hybrid retrieval + source-grounded extractive workflow. Optional API-backed summarisation if you configure your own secret.")

@st.cache_resource
def build():
    df=pd.read_csv(ROOT/"data/demo_corpus.csv")
    return EnergyNewsAssistant(df)

assistant=build()
query=st.text_input("Research question","natural gas supply disruption and price risks")
k=st.slider("Evidence documents",3,8,5)

use_llm=st.checkbox("Use API-backed grounded summarisation",False)
api_key=""
model=""
if use_llm:
    try:
        api_key=st.secrets.get("OPENAI_API_KEY","")
        model=st.secrets.get("OPENAI_MODEL","")
    except Exception:
        api_key=os.getenv("OPENAI_API_KEY","")
        model=os.getenv("OPENAI_MODEL","")
    if not api_key or not model:
        st.info("Add OPENAI_API_KEY and OPENAI_MODEL to Streamlit secrets to enable the optional API-backed path. Do not paste a key into the app.")

if st.button("Run research workflow",type="primary"):
    try:
        result=assistant.run(query,k,llm=use_llm and bool(api_key) and bool(model),api_key=api_key or None,model=model or None)
        if "llm_summary" in result:
            st.subheader("Grounded summary")
            st.write(result["llm_summary"]["text"])
            st.caption(f'Citation check passed: {result["llm_summary"]["citation_check_passed"]}')
        st.subheader("Retrieved evidence")
        for i,e in enumerate(result["evidence"],1):
            with st.expander(f'{i}. {e["source_id"]} — score {e["score"]:.3f}'):
                st.write("Categories:",", ".join(e["categories"]))
                st.write(e["evidence"])
    except Exception as exc:
        st.error(str(exc))

with st.expander("Verified offline benchmark"):
    st.write("Full-corpus evaluation: 10,788 Reuters documents; 20 controlled energy queries; 85% hit@5; MRR 0.794.")
    st.write("The public demo uses a smaller bundled corpus. Verified metrics do not include real LLM-generation quality.")
