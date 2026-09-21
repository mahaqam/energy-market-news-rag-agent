# Energy Market News RAG & Agentic Research Assistant

Portfolio project for source-grounded energy-market research. It evaluates hybrid retrieval over the uploaded Reuters news corpus and implements a multi-step retrieve → extract → cross-check workflow. An optional OpenAI Responses API adapter is included for source-grounded summarisation, but the verified metrics below cover only the fully executed deterministic retrieval and validation path.

## What was built

- Parsed 10,788 Reuters documents and category labels.
- Identified 683 documents tagged with one or more of `crude`, `nat-gas`, `gas`, `fuel`, or `pet-chem`.
- Hybrid retrieval using 70% word TF-IDF + 30% character TF-IDF.
- Controlled 20-query benchmark across five energy categories.
- Multi-step workflow that retrieves evidence, extracts source sentences, attaches source IDs/categories, and cross-checks that evidence is an exact source substring.
- Optional programmatic OpenAI Responses API adapter with citation-ID validation.
- Streamlit demo over a bundled 15-document corpus and a FastAPI retrieval endpoint.

## Verified results

| Metric | Verified result |
| --- | ---: |
| Reuters documents | 10,788 |
| Energy-category documents | 683 |
| Controlled queries | 20 |
| Hit@1 | 70% |
| Hit@3 | 85% |
| Hit@5 | 85% |
| Hit@10 | 100% |
| MRR | 0.794 |
| Mean top-5 target-category relevance | 65% |
| Evidence items validated | 100 |
| Schema-valid evidence items | 100% |
| Extractive groundedness | 100% |

The 20 queries are controlled research prompts, not a production user-query distribution.

## Programmatic LLM path

`src/assistant.py` includes an optional Responses API call. It sends only retrieved evidence, instructs the model to stay within that evidence, requires source-ID citations, and validates returned citation IDs against the retrieved sources.

The LLM path is **not included in the verified metrics** because no API-backed generation run was executed in this analysis environment. The Streamlit demo therefore defaults to the deterministic extractive workflow. If you enable the API-backed option in your own deployment, store `OPENAI_API_KEY` and an explicit `OPENAI_MODEL` in platform secrets rather than in the repository. No model name is hard-coded so the optional adapter does not imply a particular model was evaluated.

## Reproduce the full benchmark

Dataset:

- Reuters NLTK corpus: https://www.kaggle.com/datasets/boldy717/reutersnltk

Run:

```bash
pip install -r requirements.txt
python src/evaluate.py --data reutersNLTK.xlsx --output-dir results
python -m unittest tests/test_assistant.py
```

A ZIP containing one XLS/XLSX file is also accepted.

## Demo

```bash
streamlit run app/dashboard.py
```

The public demo uses `data/demo_corpus.csv`, a fixed 15-document subset so the repository does not redistribute the full uploaded workbook. `results/evidence_validation_sample.csv` contains 25 of the 100 evidence items checked in the full validation run.

FastAPI:

```bash
uvicorn api.main:app --reload
```

## Scope and limitations

The uploaded Reuters corpus is historical and is not a current European gas/power/emissions news feed. The project demonstrates retrieval, structured research orchestration, source validation, and an optional API integration pattern; it does not claim a production trading-news system or verified LLM-generation quality.

## Streamlit deployment

- Repository: `mahaqam/energy-market-news-rag-agent`
- Branch: `main`
- Main file: `app/dashboard.py`

The deterministic retrieval workflow runs without secrets. The optional API-backed summarisation path requires `OPENAI_API_KEY` and `OPENAI_MODEL` in Streamlit secrets.
