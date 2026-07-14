import pandas as pd
from src.transform.build_gold_features import build_gold_features

def test_lag_feature_shifts_correctly():
    config = {
        "forecast": {
            "group_columns": ["category"],
            "target_column": "qty",
            "lag_days": 1,
        }
    }
    df = pd.DataFrame({
        "Posting Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "category": ["A", "A", "A"],
        "Shipped Qty.": [10, 20, 30],
    })

    result = build_gold_features(df, config)

    # Second row's lag_1 should equal first row's qty (10)
    assert result.iloc[1]["lag_1"] == 10