import pandas as pd


def classify_items(silver: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-item volume and gross profit, with precomputed percentile ranks (0-100)
    for each metric. Thresholds are applied later as pure filters, not recomputed here.
    """
    item_agg = silver.groupby("item_no").agg(
        total_units=("net_units", "sum"),
        total_revenue=("salesAmountActual", "sum"),
        total_profit=("profit", "sum"),
        first_sale=("posting_date", "min"),
        last_sale=("posting_date", "max"),
    ).reset_index()

    item_agg = item_agg[item_agg["total_units"] > 0].copy()

    item_agg["volume_percentile"] = item_agg["total_units"].rank(pct=True) * 100
    item_agg["gp_percentile"] = item_agg["total_profit"].rank(pct=True) * 100

    return item_agg.sort_values("total_units", ascending=False)


def filter_by_percentile(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    """Pure filter on a precomputed percentile column — instant, no recomputation."""
    percentile_col = f"{metric}_percentile"
    return df[df[percentile_col] >= threshold]