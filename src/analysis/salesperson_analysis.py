import pandas as pd


def build_salesperson_summary(analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a per-salesperson summary with precomputed sales/profit/volume percentile ranks,
    plus a recency flag distinguishing active top performers from lapsed ones.
    """
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()
    # Exclude rows with no salesperson recorded
    sales_only = sales_only[sales_only["sales_person_code"].astype(str).str.strip() != ""].copy()

    sales_only["profit"] = sales_only["salesAmountActual"] + sales_only["costAmountActual"]
    sales_only["net_units"] = -sales_only["quantity"]

    sales_only_sorted = sales_only.sort_values("posting_date")

    salesperson_summary = sales_only_sorted.groupby("sales_person_code").agg(
        salesperson_name=("salesperson_name", "last"),
        total_sales=("salesAmountActual", "sum"),
        total_profit=("profit", "sum"),
        total_units=("net_units", "sum"),
        order_count=("posting_date", "count"),
        unique_customers=("resolved_customer_no", "nunique"),
        first_sale=("posting_date", "min"),
        last_sale=("posting_date", "max"),
    ).reset_index()

    salesperson_summary["sales_percentile"] = salesperson_summary["total_sales"].rank(pct=True) * 100
    salesperson_summary["profit_percentile"] = salesperson_summary["total_profit"].rank(pct=True) * 100
    salesperson_summary["volume_percentile"] = salesperson_summary["total_units"].rank(pct=True) * 100

    salesperson_summary["days_since_last_sale"] = (pd.Timestamp.today() - salesperson_summary["last_sale"]).dt.days
    salesperson_summary["is_active"] = salesperson_summary["days_since_last_sale"] <= 90

    return salesperson_summary.sort_values("sales_percentile", ascending=False)


def filter_by_percentile(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    percentile_col = f"{metric}_percentile"
    return df[df[percentile_col] >= threshold]