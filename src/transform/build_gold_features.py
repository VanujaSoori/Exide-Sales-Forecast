import pandas as pd

def aggregate_overall_weekly(df_silver: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per week (ISO week, Monday start)."""
    df = df_silver.copy()
    df["week_start"] = df["posting_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    weekly = (
        df.groupby("week_start")
        .agg(total_units_sold=("units_sold", "sum"))
        .reset_index()
    )
    return weekly


def fill_missing_weeks(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Ensure every week in the range has a row, filling gaps with 0 sales."""
    full_range = pd.date_range(df_weekly["week_start"].min(), df_weekly["week_start"].max(), freq="W-MON")
    full_df = pd.DataFrame({"week_start": full_range})

    merged = full_df.merge(df_weekly, on="week_start", how="left")
    merged["was_filled"] = merged["total_units_sold"].isna()
    merged["total_units_sold"] = merged["total_units_sold"].fillna(0)

    return merged


def add_time_features_weekly(df_weekly: pd.DataFrame) -> pd.DataFrame:
    df_weekly = df_weekly.copy()
    df_weekly["week_of_year"] = df_weekly["week_start"].dt.isocalendar().week.astype(int)
    df_weekly["month"] = df_weekly["week_start"].dt.month

    # Does this week contain a month-end day? (e.g. week spanning the 28th-31st)
    week_end = df_weekly["week_start"] + pd.Timedelta(days=6)
    df_weekly["contains_month_end"] = (
        df_weekly["week_start"].dt.month != week_end.dt.month
    ).astype(int)

    return df_weekly


def add_lag_and_rolling_features_weekly(df_weekly: pd.DataFrame, lag_weeks: int = 4) -> pd.DataFrame:
    df_weekly = df_weekly.sort_values("week_start").copy()

    df_weekly[f"lag_{lag_weeks}w"] = df_weekly["total_units_sold"].shift(lag_weeks)
    df_weekly["rolling_avg_4w"] = (
        df_weekly["total_units_sold"].shift(1).rolling(window=4, min_periods=1).mean()
    )

    return df_weekly


def build_gold_overall_weekly(df_silver: pd.DataFrame) -> pd.DataFrame:
    weekly = aggregate_overall_weekly(df_silver)
    weekly = fill_missing_weeks(weekly)
    weekly = add_time_features_weekly(weekly)
    weekly = add_lag_and_rolling_features_weekly(weekly)
    return weekly