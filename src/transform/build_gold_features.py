import pandas as pd


# ── Weekly ──────────────────────────────────────────────────────────

def aggregate_overall_weekly(df_silver: pd.DataFrame) -> pd.DataFrame:
    df = df_silver.copy()
    df["week_start"] = df["posting_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    weekly = df.groupby("week_start").agg(total_units_sold=("net_units", "sum")).reset_index()
    return weekly


def fill_missing_weeks(df_weekly: pd.DataFrame) -> pd.DataFrame:
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
    df_weekly["contains_month_end"] = (df_weekly["week_start"].dt.month != week_end.dt.month).astype(int)
    return df_weekly


def add_lag_and_rolling_features_weekly(df_weekly: pd.DataFrame, lag_weeks: int = 4) -> pd.DataFrame:
    df_weekly = df_weekly.sort_values("week_start").copy()
    df_weekly[f"lag_{lag_weeks}w"] = df_weekly["total_units_sold"].shift(lag_weeks)
    df_weekly["rolling_avg_4w"] = df_weekly["total_units_sold"].shift(1).rolling(4, min_periods=1).mean()
    return df_weekly


def build_gold_overall_weekly(df_silver: pd.DataFrame) -> pd.DataFrame:
    weekly = aggregate_overall_weekly(df_silver)
    weekly = fill_missing_weeks(weekly)
    weekly = add_time_features_weekly(weekly)
    weekly = add_lag_and_rolling_features_weekly(weekly)
    return weekly


# ── Monthly ─────────────────────────────────────────────────────────

def aggregate_overall_monthly(df_silver: pd.DataFrame) -> pd.DataFrame:
    df = df_silver.copy()
    df["month_start"] = df["posting_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month_start").agg(total_units_sold=("net_units", "sum")).reset_index()
    return monthly


def fill_missing_months(df_monthly: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(df_monthly["month_start"].min(), df_monthly["month_start"].max(), freq="MS")
    full_df = pd.DataFrame({"month_start": full_range})
    merged = full_df.merge(df_monthly, on="month_start", how="left")
    merged["was_filled"] = merged["total_units_sold"].isna()
    merged["total_units_sold"] = merged["total_units_sold"].fillna(0)
    return merged


def add_time_features_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    df_monthly = df_monthly.copy()
    df_monthly["month_of_year"] = df_monthly["month_start"].dt.month
    df_monthly["quarter"] = df_monthly["month_start"].dt.quarter
    # Your 3 seasonal blocks: Jan-Apr, May-Aug, Sep-Dec
    df_monthly["season_block"] = pd.cut(
        df_monthly["month_of_year"], bins=[0, 4, 8, 12], labels=["first_4mo", "mid_4mo", "last_4mo"]
    )
    return df_monthly


def add_lag_and_rolling_features_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    df_monthly = df_monthly.sort_values("month_start").copy()
    df_monthly["lag_1m"] = df_monthly["total_units_sold"].shift(1)
    df_monthly["lag_12m"] = df_monthly["total_units_sold"].shift(12)  # year-over-year
    df_monthly["rolling_avg_3m"] = df_monthly["total_units_sold"].shift(1).rolling(3, min_periods=1).mean()
    return df_monthly


def build_gold_overall_monthly(df_silver: pd.DataFrame) -> pd.DataFrame:
    monthly = aggregate_overall_monthly(df_silver)
    monthly = fill_missing_months(monthly)
    monthly = add_time_features_monthly(monthly)
    monthly = add_lag_and_rolling_features_monthly(monthly)
    return monthly

# ── Weekly, by brand ────────────────────────────────────────────────

def aggregate_by_brand_weekly(df_silver: pd.DataFrame) -> pd.DataFrame:
    df = df_silver.copy()
    df["week_start"] = df["posting_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    weekly = (
        df.groupby(["week_start", "brand_code"])
        .agg(total_units_sold=("net_units", "sum"))
        .reset_index()
    )
    return weekly


def fill_missing_weeks_by_brand(df_weekly: pd.DataFrame) -> pd.DataFrame:
    full_weeks = pd.date_range(df_weekly["week_start"].min(), df_weekly["week_start"].max(), freq="W-MON")
    brands = df_weekly["brand_code"].unique()
    full_grid = pd.MultiIndex.from_product([full_weeks, brands], names=["week_start", "brand_code"]).to_frame(index=False)
    merged = full_grid.merge(df_weekly, on=["week_start", "brand_code"], how="left")
    merged["was_filled"] = merged["total_units_sold"].isna()
    merged["total_units_sold"] = merged["total_units_sold"].fillna(0)
    return merged


def add_time_features_brand_weekly(df_weekly: pd.DataFrame) -> pd.DataFrame:
    df_weekly = df_weekly.copy()
    df_weekly["week_of_year"] = df_weekly["week_start"].dt.isocalendar().week.astype(int)
    df_weekly["month"] = df_weekly["week_start"].dt.month
    week_end = df_weekly["week_start"] + pd.Timedelta(days=6)
    df_weekly["contains_month_end"] = (df_weekly["week_start"].dt.month != week_end.dt.month).astype(int)
    return df_weekly


def add_lag_and_rolling_features_brand_weekly(df_weekly: pd.DataFrame, lag_weeks: int = 4) -> pd.DataFrame:
    df_weekly = df_weekly.sort_values(["brand_code", "week_start"]).copy()
    df_weekly[f"lag_{lag_weeks}w"] = df_weekly.groupby("brand_code")["total_units_sold"].shift(lag_weeks)
    df_weekly["rolling_avg_4w"] = (
        df_weekly.groupby("brand_code")["total_units_sold"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    )
    return df_weekly


def build_gold_brand_weekly(df_silver: pd.DataFrame) -> pd.DataFrame:
    weekly = aggregate_by_brand_weekly(df_silver)
    weekly = fill_missing_weeks_by_brand(weekly)
    weekly = add_time_features_brand_weekly(weekly)
    weekly = add_lag_and_rolling_features_brand_weekly(weekly)
    return weekly