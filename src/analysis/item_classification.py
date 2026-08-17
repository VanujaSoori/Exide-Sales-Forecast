import pandas as pd


def classify_items(silver: pd.DataFrame, volume_pct: float, velocity_pct: float, price_pct: float, margin_pct: float) -> pd.DataFrame:
    """
    Classifies each item into High/Low tiers for volume, movement speed, price, and margin,
    using user-supplied percentile thresholds (0-100).
    """
    item_agg = silver.groupby("item_no").agg(
        total_units=("net_units", "sum"),
        total_revenue=("salesAmountActual", "sum"),
        total_profit=("profit", "sum"),
        first_sale=("posting_date", "min"),
        last_sale=("posting_date", "max"),
    ).reset_index()

    item_agg["active_weeks"] = ((item_agg["last_sale"] - item_agg["first_sale"]).dt.days / 7).clip(lower=1)
    item_agg["velocity_units_per_week"] = item_agg["total_units"] / item_agg["active_weeks"]
    item_agg["avg_unit_price"] = item_agg["total_revenue"] / item_agg["total_units"].replace(0, pd.NA)

    item_agg["has_zero_revenue"] = item_agg["total_revenue"] == 0
    item_agg["gross_margin_pct"] = item_agg.apply(
        lambda row: (row["total_profit"] / row["total_revenue"]) * 100 if row["total_revenue"] != 0 else pd.NA,
        axis=1
    )

    item_agg = item_agg[item_agg["total_units"] > 0].copy()

    def classify(series, percentile):
        valid = series.dropna()
        cutoff = valid.quantile(percentile / 100)
        def label(x):
            if pd.isna(x):
                return "N/A"
            return "High" if x > cutoff else "Low"
        return series.apply(label), cutoff

    item_agg["volume_tier"], _ = classify(item_agg["total_units"], volume_pct)
    item_agg["velocity_tier"], _ = classify(item_agg["velocity_units_per_week"], velocity_pct)
    item_agg["price_tier"], _ = classify(item_agg["avg_unit_price"], price_pct)
    item_agg["margin_tier"], _ = classify(item_agg["gross_margin_pct"], margin_pct)

    item_agg["movement_tier"] = item_agg["velocity_tier"].map({"High": "Fast Moving", "Low": "Slow Moving"})

    return item_agg[[
        "item_no", "total_units", "volume_tier",
        "velocity_units_per_week", "movement_tier",
        "avg_unit_price", "price_tier",
        "gross_margin_pct", "margin_tier",
        "has_zero_revenue",
        "first_sale", "last_sale"
    ]].sort_values("total_units", ascending=False)