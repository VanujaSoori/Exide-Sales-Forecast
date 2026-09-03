import pandas as pd


def classify_items(analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-item volume and gross profit from analysis silver (Sale entries only,
    net of returns), excluding warranty claims (zero-revenue shipments).
    Precomputes percentile ranks (0-100) for each metric.
    """
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()

    is_warranty_claim = (
        (sales_only["documentType"] == "Sales_x0020_Shipment")
        & (sales_only["salesAmountActual"] == 0)
    )
    genuine_sales = sales_only[~is_warranty_claim].copy()

    genuine_sales["net_units"] = -genuine_sales["quantity"]
    genuine_sales["profit"] = genuine_sales["salesAmountActual"] + genuine_sales["costAmountActual"]

    item_agg = genuine_sales.groupby("item_no").agg(
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

def summarize_warranty_claims(analysis_silver: pd.DataFrame) -> pd.DataFrame:
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()
    warranty = sales_only[
        (sales_only["documentType"] == "Sales_x0020_Shipment") & (sales_only["salesAmountActual"] == 0)
    ].copy()

    return warranty.groupby("item_no").agg(
        warranty_claim_count=("quantity", "count"),
        warranty_claim_units=("quantity", lambda x: x.abs().sum()),
        warranty_claim_cost=("costAmountActual", lambda x: x.abs().sum()),
    ).reset_index().sort_values("warranty_claim_cost", ascending=False)


def filter_by_percentile(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    percentile_col = f"{metric}_percentile"
    return df[df[percentile_col] >= threshold]