import pandas as pd
from core.data_engine import DataEngine

def test_profile():
    df = pd.DataFrame({"region": ["A", "B"], "revenue": [10, 20]})
    e = DataEngine({"test.csv": df})
    p = e.profile().iloc[0]
    assert p["rows"] == 2
    assert p["columns"] == 2
