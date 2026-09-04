import pandas as pd


def compute_item_velocity(analysis_silver: pd.DataFrame, min_matched_events: int = 30) -> pd.DataFrame:
    """
    Computes fast/slow-moving velocity per item, by matching each sale to its
    originating purchase lot (item_no + lot_no) and measuring days to sell.
    Items below min_matched_events are excluded as unreliable sample sizes.
    """
    purchases_first = analysis_silver[analysis_silver["entryType"] == "Purchase"].groupby(
        ["item_no", "lot_no"]
    )["posting_date"].min().reset_index().rename(columns={"posting_date": "purchase_date"})

    shipments = analysis_silver[
        (analysis_silver["entryType"] == "Sale") & (analysis_silver["documentType"] == "Sales_x0020_Shipment")
    ][["item_no", "lot_no", "posting_date"]].rename(columns={"posting_date": "sale_date"})

    lot_matched = shipments.merge(purchases_first, on=["item_no", "lot_no"], how="inner")
    lot_matched["days_to_sell"] = (lot_matched["sale_date"] - lot_matched["purchase_date"]).dt.days
    lot_matched = lot_matched[lot_matched["days_to_sell"] >= 0].copy()

    item_velocity = lot_matched.groupby("item_no").agg(
        matched_events=("days_to_sell", "count"),
        avg_days_to_sell=("days_to_sell", "mean"),
        median_days_to_sell=("days_to_sell", "median"),
    ).reset_index()

    item_velocity = item_velocity[item_velocity["matched_events"] >= min_matched_events].copy()
    item_velocity["velocity_percentile"] = (1 - item_velocity["avg_days_to_sell"].rank(pct=True)) * 100

    item_names = analysis_silver[["item_no", "item_description"]].drop_duplicates(subset="item_no")
    item_velocity = item_velocity.merge(item_names, on="item_no", how="left")

    return item_velocity.sort_values("avg_days_to_sell")


def filter_by_percentile(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    percentile_col = f"{metric}_percentile"
    return df[df[percentile_col] >= threshold]