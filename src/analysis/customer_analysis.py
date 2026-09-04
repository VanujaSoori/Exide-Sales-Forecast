import pandas as pd


def build_customer_summary(analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a per-customer summary with precomputed sales/profit/volume percentile ranks,
    plus a recency flag distinguishing active top customers from lapsed ones.
    Relies on is_identifiable_customer, already resolved in clean_to_silver_analysis().
    """
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()
    identifiable = sales_only[sales_only["is_identifiable_customer"]].copy()

    identifiable["profit"] = identifiable["salesAmountActual"] + identifiable["costAmountActual"]
    identifiable["net_units"] = -identifiable["quantity"]

    identifiable_sorted = identifiable.sort_values("posting_date")

    customer_summary = identifiable_sorted.groupby("resolved_customer_no").agg(
        customer_name=("resolved_customer_name", "last"),
        customer_address=("resolved_customer_address", "last"),
        customer_phone1=("resolved_customer_phone1", "last"),
        customer_phone2=("resolved_customer_phone2", "last"),
        customer_email=("resolved_customer_email", "last"),
        total_sales=("salesAmountActual", "sum"),
        total_profit=("profit", "sum"),
        total_units=("net_units", "sum"),
        order_count=("posting_date", "count"),
        first_purchase=("posting_date", "min"),
        last_purchase=("posting_date", "max"),
    ).reset_index()

    customer_summary = customer_summary[customer_summary["total_units"] > 0].copy()

    customer_summary["sales_percentile"] = customer_summary["total_sales"].rank(pct=True) * 100
    customer_summary["profit_percentile"] = customer_summary["total_profit"].rank(pct=True) * 100
    customer_summary["volume_percentile"] = customer_summary["total_units"].rank(pct=True) * 100

    # Recency: distinguishes active top customers from lapsed ones
    customer_summary["days_since_last_purchase"] = (pd.Timestamp.today() - customer_summary["last_purchase"]).dt.days
    customer_summary["is_active"] = customer_summary["days_since_last_purchase"] <= 180

    return customer_summary.sort_values("sales_percentile", ascending=False)


def filter_by_percentile(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    percentile_col = f"{metric}_percentile"
    return df[df[percentile_col] >= threshold]


def get_lapsed_top_customers(df: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    """Top customers by the given metric who haven't purchased recently — re-engagement candidates."""
    individual_prefixes = ("MR.", "MRS.", "MS.", "DR.", "MR ", "MRS ", "MS ", "DR ")

    top = filter_by_percentile(df, metric, threshold)
    lapsed = top[~top["is_active"]].copy()
    lapsed["is_individual"] = lapsed["customer_name"].str.strip().str.upper().str.startswith(individual_prefixes)

    return lapsed.sort_values("days_since_last_purchase", ascending=False)