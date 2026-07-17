import pandas as pd


# ── Daily aggregation ──────────────────────────────────────────────

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


def add_time_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    df_daily = df_daily.copy()
    df_daily["day_of_week"] = df_daily["posting_date"].dt.dayofweek
    df_daily["month"] = df_daily["posting_date"].dt.month
    df_daily["is_weekend"] = df_daily["day_of_week"].isin([5, 6]).astype(int)

    df_daily["day_of_month"] = df_daily["posting_date"].dt.day
    df_daily["days_until_month_end"] = (
        df_daily["posting_date"] + pd.offsets.MonthEnd(0) - df_daily["posting_date"]
    ).dt.days
    df_daily["is_month_end"] = (df_daily["days_until_month_end"] == 0).astype(int)

    return df_daily


def add_lag_and_rolling_features(df_daily: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    df_daily = df_daily.sort_values("posting_date").copy()

    df_daily[f"lag_{lag_days}"] = df_daily["total_units_sold"].shift(lag_days)
    df_daily["rolling_avg_7"] = (
        df_daily["total_units_sold"].shift(1).rolling(window=7, min_periods=1).mean()
    )
    df_daily["rolling_avg_30"] = (
        df_daily["total_units_sold"].shift(1).rolling(window=30, min_periods=1).mean()
    )

    # Previous calendar month's peak sales day
    df_daily["year_month"] = df_daily["posting_date"].dt.to_period("M")
    monthly_peak = df_daily.groupby("year_month")["total_units_sold"].max().reset_index()
    monthly_peak.columns = ["year_month", "prev_month_peak"]
    monthly_peak["year_month"] = monthly_peak["year_month"] + 1

    df_daily = df_daily.merge(monthly_peak, on="year_month", how="left")
    df_daily = df_daily.drop(columns=["year_month"])

    return df_daily


def build_gold_overall(df_silver: pd.DataFrame) -> pd.DataFrame:
    daily = aggregate_overall_daily(df_silver)
    daily = fill_missing_dates(daily)
    daily = add_time_features(daily)
    daily = add_lag_and_rolling_features(daily)
    return daily


# ── Weekly aggregation ─────────────────────────────────────────────

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