import pandas as pd
from src.training.train_model import prepare_features

def test_prepare_features_drops_incomplete_rows():
    config = {
        "forecast": {"target_column": "qty", "lag_days": 1},
    }
    df = pd.DataFrame({
        "qty": [10, 20, 30],
        "day_of_week": [0, 1, 2],
        "month": [1, 1, 1],
        "is_weekend": [0, 0, 0],
        "lag_1": [None, 10, 20],
        "rolling_avg_7": [None, 10, 15],
    })

    X, y, feature_cols = prepare_features(df, config)

    # Row 0 has NaNs in lag_1/rolling_avg_7 and should be dropped
    assert len(X) == 2
    assert len(y) == 2