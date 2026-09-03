def get_top_customers(customer_summary: pd.DataFrame, metric: str, percentile_threshold: float) -> pd.DataFrame:
    """Pure filter — instant, works on the precomputed table, no recomputation."""
    percentile_col = f"{metric}_percentile"
    return customer_summary[customer_summary[percentile_col] >= percentile_threshold]