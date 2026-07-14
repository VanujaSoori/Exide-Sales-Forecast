import pandas as pd
import yaml


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def build_gold_features(df_silver: pd.DataFrame, config: dict) -> pd.DataFrame:
    group_cols = config["forecast"]["group_columns"]
    target_col = config["forecast"]["target_column"]
    lag_days = config["forecast"]["lag_days"]

    # Aggregate to daily totals per group (e.g. category + region)
    df_gold = (
        df_silver
        .groupby(["Posting Date"] + group_cols)
        .agg(**{target_col: ("Shipped Qty.", "sum")})
        .reset_index()
    )

    # Time-based features
    df_gold["day_of_week"] = df_gold["Posting Date"].dt.dayofweek
    df_gold["month"] = df_gold["Posting Date"].dt.month
    df_gold["is_weekend"] = df_gold["day_of_week"].isin([5, 6]).astype(int)

    # Sort before computing lag/rolling features — critical, or values will be wrong
    df_gold = df_gold.sort_values(["Posting Date"] + group_cols)

    # Lag feature: same group's value N days ago
    df_gold[f"lag_{lag_days}"] = (
        df_gold.groupby(group_cols)[target_col].shift(lag_days)
    )

    # Rolling 7-day average, per group, excluding current day (shift(1) first)
    df_gold["rolling_avg_7"] = (
        df_gold.groupby(group_cols)[target_col]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )

    return df_gold


if __name__ == "__main__":
    config = load_config()
    df_silver = pd.read_parquet(config["paths"]["silver"])

    df_gold = build_gold_features(df_silver, config)
    df_gold.to_parquet(config["paths"]["gold"], index=False)

    print(f"Wrote {len(df_gold)} rows to gold")
    print(df_gold.head())