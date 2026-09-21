from __future__ import annotations
import ast, json, os, re, urllib.request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ENERGY_CATEGORIES={"crude","nat-gas","gas","fuel","pet-chem"}

def parse_categories(value):
    try:
        parsed=ast.literal_eval(str(value))
        if isinstance(parsed,(list,tuple)): return [str(x) for x in parsed]
    except Exception: pass
    return [t.strip(" []'\"") for t in re.split(r"[,;]",str(value)) if t.strip(" []'\"")]

def top_sentence(text,query):
    sents=[s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if s.strip()]
    if not sents: return text[:500]
    qterms=set(re.findall(r"[A-Za-z]{3,}",query.lower()))
    return max(((len(qterms&set(re.findall(r"[A-Za-z]{3,}",s.lower()))),-len(s),s) for s in sents))[2][:500]

class EnergyNewsAssistant:
    def __init__(self,df:pd.DataFrame):
        self.df=df.copy()
        self.df["cat_list"]=self.df["categories"].map(parse_categories)
        self.df["text_clean"]=self.df["text"].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
        self.word=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=20000,sublinear_tf=True,stop_words="english")
        self.char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=1,max_features=30000,sublinear_tf=True)
        self.Xw=self.word.fit_transform(self.df["text_clean"])
        self.Xc=self.char.fit_transform(self.df["text_clean"])

    def retrieve(self,query,k=5):
        qw=self.word.transform([query]); qc=self.char.transform([query])
        score=.7*np.asarray(self.Xw.dot(qw.T).todense()).ravel()+.3*np.asarray(self.Xc.dot(qc.T).todense()).ravel()
        order=np.argsort(-score)[:k]
        out=[]
        for idx in order:
            row=self.df.iloc[idx]
            out.append({
                "source_id":str(row["ids"]),"categories":row["cat_list"],
                "score":float(score[idx]),"evidence":top_sentence(row["text_clean"],query)
            })
        return out

    def run(self,query,k=5,llm=False,api_key=None,model=None):
        evidence=self.retrieve(query,k)
        result={"query":query,"evidence":evidence,"mode":"extractive"}
        if llm:
            key=api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is required for API-backed summarisation.")
            result["llm_summary"]=openai_grounded_summary(query,evidence,key,model)
            result["mode"]="api-backed"
        return result

def openai_grounded_summary(query,evidence,api_key,model=None):
    model=model or os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("OPENAI_MODEL is required for API-backed summarisation.")
    context="\n".join(f'[{e["source_id"]}] {e["evidence"]}' for e in evidence)
    prompt=(
        "You are an energy-market research assistant. Use only the supplied evidence. "
        "Return a concise market summary, key drivers, and risks. Cite source IDs in square brackets. "
        "If evidence is insufficient, say so.\n\n"
        f"Question: {query}\n\nEvidence:\n{context}"
    )
    body=json.dumps({"model":model,"input":prompt}).encode("utf-8")
    req=urllib.request.Request(
        "https://api.openai.com/v1/responses",data=body,method="POST",
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=45) as resp:
        data=json.loads(resp.read().decode("utf-8"))
    texts=[]
    for item in data.get("output",[]):
        for content in item.get("content",[]):
            if content.get("type")=="output_text":
                texts.append(content.get("text",""))
    text="\n".join(texts).strip()
    if not text:
        raise RuntimeError("No output_text returned by Responses API.")
    valid_ids={e["source_id"] for e in evidence}
    cited=set(re.findall(r"\[([^\]]+)\]",text))
    invalid=sorted(cited-valid_ids)
    return {"text":text,"cited_source_ids":sorted(cited),"invalid_citations":invalid,
            "citation_check_passed":len(invalid)==0}

def load_demo(path):
    return pd.read_csv(path)
