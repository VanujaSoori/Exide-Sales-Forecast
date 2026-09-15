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