import sys, unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from assistant import EnergyNewsAssistant

class TestAssistant(unittest.TestCase):
    def test_retrieval_schema_and_grounding(self):
        a=EnergyNewsAssistant(pd.read_csv(ROOT/"data/demo_corpus.csv"))
        out=a.run("crude oil production and prices",k=3)
        self.assertEqual(len(out["evidence"]),3)
        source=dict(zip(a.df["ids"].astype(str),a.df["text_clean"]))
        for e in out["evidence"]:
            self.assertTrue(e["source_id"])
            self.assertIn(e["evidence"],source[e["source_id"]])

if __name__=="__main__":
    unittest.main()
