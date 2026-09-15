import pandas as pd


def match_shipment_return_pairs(analysis_silver: pd.DataFrame, quantity_tolerance: float = 0.20) -> pd.DataFrame:
    """
    Matches Sales Shipments near month-end to Sales Returns shortly after,
    using exact lot matching (strong signal). One-to-one matching to avoid
    duplicate/inflated pair counts.
    """
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()

    shipments = sales_only[sales_only["documentType"] == "Sales_x0020_Shipment"].copy()
    returns = sales_only[sales_only["documentType"] == "Sales_x0020_Return_x0020_Receipt"].copy()

    shipments["day_of_month"] = shipments["posting_date"].dt.day
    shipments["days_in_month"] = shipments["posting_date"].dt.days_in_month
    month_end_shipments = shipments[shipments["day_of_month"] > shipments["days_in_month"] - 3].copy()

    matched_pairs = []
    used_return_indices = set()

    grouped_shipments = month_end_shipments.groupby(["item_no", "lot_no", "sales_person_code"])

    for (item_no, lot_no, sp_code), ship_group in grouped_shipments:
        candidate_returns = returns[
            (returns["item_no"] == item_no) &
            (returns["lot_no"] == lot_no) &
            (returns["sales_person_code"] == sp_code) &
            (~returns.index.isin(used_return_indices))
        ]
        if candidate_returns.empty:
            continue

        for _, ship in ship_group.sort_values("quantity").iterrows():
            avail = candidate_returns[
                (candidate_returns["posting_date"] > ship["posting_date"]) &
                (~candidate_returns.index.isin(used_return_indices))
            ]
            if avail.empty:
                continue

            ship_qty = abs(ship["quantity"])
            qty_diff = (avail["quantity"].abs() - ship_qty).abs()
            best_idx = qty_diff.idxmin()

            if qty_diff.loc[best_idx] / ship_qty <= quantity_tolerance:
                best = avail.loc[best_idx]
                matched_pairs.append({
                    "item_no": item_no,
                    "lot_no": lot_no,
                    "sales_person_code": sp_code,
                    "customer_no_shipment": ship["resolved_customer_no"],
                    "customer_name_shipment": ship["resolved_customer_name"],
                    "is_identifiable_customer": ship["is_identifiable_customer"],
                    "shipment_date": ship["posting_date"],
                    "shipment_units": ship_qty,
                    "shipment_value": ship["salesAmountActual"],
                    "return_date": best["posting_date"],
                    "return_units": abs(best["quantity"]),
                    "days_between": (best["posting_date"] - ship["posting_date"]).days,
                })
                used_return_indices.add(best_idx)

    return pd.DataFrame(matched_pairs)

def summarize_by_salesperson(matched_pairs: pd.DataFrame, analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates matched round-trips per salesperson, normalized by their total
    month-end shipment activity (not raw counts) to avoid conflating high-volume
    salespeople with genuinely suspicious behavior.
    """
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()
    shipments = sales_only[sales_only["documentType"] == "Sales_x0020_Shipment"].copy()
    shipments["day_of_month"] = shipments["posting_date"].dt.day
    shipments["days_in_month"] = shipments["posting_date"].dt.days_in_month
    month_end_shipments = shipments[shipments["day_of_month"] > shipments["days_in_month"] - 3]

    total_month_end_shipments = month_end_shipments.groupby("sales_person_code").size()

    summary = matched_pairs.groupby("sales_person_code").agg(
        round_trip_count=("item_no", "count"),
        total_value_involved=("shipment_value", "sum"),
        avg_days_between=("days_between", "mean"),
        identifiable_customer_share=("is_identifiable_customer", "mean"),
        active_months=("shipment_date", lambda x: x.dt.to_period("M").nunique()),
    ).reset_index()

    summary = summary.merge(
        total_month_end_shipments.rename("total_month_end_shipments"),
        left_on="sales_person_code", right_index=True, how="left"
    )
    summary["round_trip_rate"] = summary["round_trip_count"] / summary["total_month_end_shipments"]

    summary["rate_percentile"] = summary["round_trip_rate"].rank(pct=True) * 100
    summary["value_percentile"] = summary["total_value_involved"].rank(pct=True) * 100

    return summary.sort_values("round_trip_rate", ascending=False)

def summarize_by_customer(matched_pairs: pd.DataFrame, analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates matched round-trips per customer, normalized by their total
    month-end shipment activity. Restricted to identifiable customers only.
    """
    sales_only = analysis_silver[analysis_silver["entryType"] == "Sale"].copy()
    shipments = sales_only[sales_only["documentType"] == "Sales_x0020_Shipment"].copy()
    shipments["day_of_month"] = shipments["posting_date"].dt.day
    shipments["days_in_month"] = shipments["posting_date"].dt.days_in_month
    month_end_shipments = shipments[shipments["day_of_month"] > shipments["days_in_month"] - 3]

    total_month_end_by_customer = month_end_shipments[
        month_end_shipments["is_identifiable_customer"]
    ].groupby("resolved_customer_no").size()

    identifiable_pairs = matched_pairs[matched_pairs["is_identifiable_customer"]].copy()

    summary = identifiable_pairs.groupby("customer_no_shipment").agg(
        customer_name=("customer_name_shipment", "last"),
        round_trip_count=("item_no", "count"),
        total_value_involved=("shipment_value", "sum"),
        avg_days_between=("days_between", "mean"),
        distinct_salespeople=("sales_person_code", "nunique"),
        active_months=("shipment_date", lambda x: x.dt.to_period("M").nunique()),
    ).reset_index()

    summary = summary.merge(
        total_month_end_by_customer.rename("total_month_end_shipments"),
        left_on="customer_no_shipment", right_index=True, how="left"
    )
    summary["round_trip_rate"] = summary["round_trip_count"] / summary["total_month_end_shipments"]

    summary["rate_percentile"] = summary["round_trip_rate"].rank(pct=True) * 100
    summary["value_percentile"] = summary["total_value_involved"].rank(pct=True) * 100

    return summary.sort_values("round_trip_rate", ascending=False)