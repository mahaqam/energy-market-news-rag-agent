from __future__ import annotations
import argparse, ast, io, json, re, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BENCHMARK_QUERIES = [
("crude","crude oil prices and OPEC production"),
("crude","oil supply cuts and crude market prices"),
("crude","petroleum output exports and crude prices"),
("crude","oil production disruptions and market supply"),
("nat-gas","natural gas prices supply and demand"),
("nat-gas","gas pipeline supply and natural gas market"),
("nat-gas","natural gas production reserves and prices"),
("nat-gas","gas demand storage and natural gas supply"),
("gas","gasoline market prices and refinery supply"),
("gas","gasoline demand prices and inventories"),
("gas","motor gasoline supply and fuel market"),
("gas","gasoline production shortages and prices"),
("fuel","fuel oil prices supply and refinery demand"),
("fuel","fuel shortages prices and energy supply"),
("fuel","heating fuel market demand and prices"),
("fuel","fuel production stocks and consumption"),
("pet-chem","petrochemical prices production and demand"),
("pet-chem","petrochemical industry output and feedstocks"),
("pet-chem","chemical market prices and petroleum feedstock"),
("pet-chem","petrochemical supply production and exports"),
]
ENERGY_CATEGORIES={"crude","nat-gas","gas","fuel","pet-chem"}

def read_excel_any(path: str) -> pd.DataFrame:
    p=Path(path)
    if p.suffix.lower()==".zip":
        with zipfile.ZipFile(p) as z:
            names=[n for n in z.namelist() if n.lower().endswith((".xlsx",".xls"))]
            if len(names)!=1:
                raise ValueError(f"Expected one spreadsheet in {p}, found {names}")
            return pd.read_excel(io.BytesIO(z.read(names[0])))
    return pd.read_excel(p)

def parse_categories(value):
    try:
        parsed=ast.literal_eval(str(value))
        if isinstance(parsed,(list,tuple)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [t.strip(" []'\"") for t in re.split(r"[,;]",str(value)) if t.strip(" []'\"")]

def top_sentence(text: str, query: str) -> str:
    sents=[s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if s.strip()]
    if not sents:
        return text[:500]
    qterms=set(re.findall(r"[A-Za-z]{3,}",query.lower()))
    ranked=[]
    for s in sents:
        st=set(re.findall(r"[A-Za-z]{3,}",s.lower()))
        ranked.append((len(qterms&st),-len(s),s))
    return max(ranked)[2][:500]

def fit_retriever(df):
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,stop_words="english")
    char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True)
    Xw=word.fit_transform(df["text_clean"])
    Xc=char.fit_transform(df["text_clean"])
    return word,char,Xw,Xc

def retrieve(df,word,char,Xw,Xc,query,k=5):
    qw=word.transform([query]); qc=char.transform([query])
    score=0.7*np.asarray(Xw.dot(qw.T).todense()).ravel()+0.3*np.asarray(Xc.dot(qc.T).todense()).ravel()
    order=np.argsort(-score)[:k]
    return [(int(i),float(score[i])) for i in order]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--output-dir",default="results")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    df=read_excel_any(args.data)
    df["cat_list"]=df["categories"].map(parse_categories)
    df["text_clean"]=df["text"].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    t=time.perf_counter()
    word,char,Xw,Xc=fit_retriever(df)
    fit_seconds=time.perf_counter()-t

    rows=[]; evidence_items=[]
    for target,query in BENCHMARK_QUERIES:
        qw=word.transform([query]); qc=char.transform([query])
        score=0.7*np.asarray(Xw.dot(qw.T).todense()).ravel()+0.3*np.asarray(Xc.dot(qc.T).todense()).ravel()
        order=np.argsort(-score)
        first=next(rank for rank,idx in enumerate(order,1) if target in df.iloc[idx]["cat_list"])
        top5=order[:5]
        rows.append({
            "target_category":target,"query":query,"rank_first_relevant":int(first),
            "hit1":int(first<=1),"hit3":int(first<=3),"hit5":int(first<=5),"hit10":int(first<=10),
            "rr":1/first,
            "top5_relevance_rate":float(np.mean([target in df.iloc[i]["cat_list"] for i in top5]))
        })
        for idx in top5:
            row=df.iloc[idx]
            snippet=top_sentence(row["text_clean"],query)
            evidence_items.append({
                "query":query,"target_category":target,"source_id":str(row["ids"]),
                "categories":row["cat_list"],"score":float(score[idx]),"extractive_evidence":snippet,
            })

    eval_df=pd.DataFrame(rows)
    evidence_df=pd.DataFrame(evidence_items)
    id_to_text=dict(zip(df["ids"].astype(str),df["text_clean"]))
    schema_valid=(evidence_df["source_id"].astype(bool)&evidence_df["extractive_evidence"].astype(bool)).mean()
    grounded=np.mean([r.extractive_evidence in id_to_text[r.source_id] for r in evidence_df.itertuples()])
    energy_docs=int(df["cat_list"].map(lambda c: bool(set(c)&ENERGY_CATEGORIES)).sum())

    metrics={
        "documents":int(len(df)),
        "energy_documents":energy_docs,
        "controlled_queries":len(BENCHMARK_QUERIES),
        "hit_at_1":float(eval_df["hit1"].mean()),
        "hit_at_3":float(eval_df["hit3"].mean()),
        "hit_at_5":float(eval_df["hit5"].mean()),
        "hit_at_10":float(eval_df["hit10"].mean()),
        "mrr":float(eval_df["rr"].mean()),
        "mean_top5_relevance_rate":float(eval_df["top5_relevance_rate"].mean()),
        "vectorizer_fit_seconds":fit_seconds,
        "workflow_evidence_items":int(len(evidence_df)),
        "schema_valid_rate":float(schema_valid),
        "extractive_groundedness_rate":float(grounded),
        "limitations":[
            "The uploaded Reuters corpus is historical and is not a modern European power/emissions feed.",
            "The 20 benchmark queries are controlled, expert-written prompts rather than a production query distribution.",
            "Verified metrics cover hybrid retrieval and deterministic extractive validation only.",
            "No real LLM output is included in verified metrics unless an API-backed run is separately executed."
        ]
    }
    eval_df.to_csv(out/"query_evaluation.csv",index=False)
    evidence_df.to_csv(out/"evidence_validation_sample.csv",index=False)
    with open(out/"metrics.json","w") as f: json.dump(metrics,f,indent=2)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":
    main()
