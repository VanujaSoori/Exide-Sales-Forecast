import pandas as pd


def aggregate_overall_daily(df_silver: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: aggregate to one row per day, overall battery sales."""
    daily = (
        df_silver
        .groupby("posting_date")
        .agg(total_units_sold=("units_sold", "sum"))
        .reset_index()
    )
    return daily


def fill_missing_dates(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Ensure every day in the range has a row, filling gaps with 0 sales."""
    full_range = pd.date_range(df_daily["posting_date"].min(), df_daily["posting_date"].max())
    full_df = pd.DataFrame({"posting_date": full_range})

    merged = full_df.merge(df_daily, on="posting_date", how="left")
    merged["was_filled"] = merged["total_units_sold"].isna()
    merged["total_units_sold"] = merged["total_units_sold"].fillna(0)

    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["posting_date"].dt.dayofweek
    df["month"] = df["posting_date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    df = df.sort_values("posting_date").copy()
    df[f"lag_{lag_days}"] = df["total_units_sold"].shift(lag_days)
    df["rolling_avg_7"] = (
        df["total_units_sold"].shift(1).rolling(window=7, min_periods=1).mean()
    )
    return df


def build_gold_overall(df_silver: pd.DataFrame) -> pd.DataFrame:
    daily = aggregate_overall_daily(df_silver)
    daily = fill_missing_dates(daily)
    daily = add_time_features(daily)
    daily = add_lag_and_rolling_features(daily)
    return daily