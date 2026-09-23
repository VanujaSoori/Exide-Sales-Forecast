import pandas as pd


def compute_current_stock(analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Computes current stock on hand per item + location, from cumulative
    signed quantity across Purchase, Sale, and Transfer entries.
    """
    stock_moves = analysis_silver.copy()
    stock_moves["signed_qty"] = stock_moves["quantity"]

    current_stock = stock_moves.groupby(["item_no", "location_code"])["signed_qty"].sum().reset_index()
    current_stock = current_stock.rename(columns={"signed_qty": "current_stock"})
    current_stock["current_stock"] = current_stock["current_stock"].clip(lower=0)

    return current_stock


def compute_incoming_stock(purchase_orders_bronze: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates open purchase order quantities per item + location.
    """
    open_orders = purchase_orders_bronze[purchase_orders_bronze["documentType"] == "Order"].copy()

    incoming_stock = open_orders.groupby(["itemNo", "locationCode"])["qtyToReceive"].sum().reset_index()
    incoming_stock = incoming_stock.rename(columns={
        "itemNo": "item_no", "locationCode": "location_code", "qtyToReceive": "incoming_stock"
    })

    return incoming_stock


def allocate_item_forecast_to_locations(item_monthly_forecast: pd.DataFrame, analysis_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Splits each item's company-wide forecast across locations, using each
    location's historical share of that item's sales (last 12 months).
    """
    sales_only = analysis_silver[
        (analysis_silver["entryType"] == "Sale") & (analysis_silver["documentType"] == "Sales_x0020_Shipment")
    ].copy()
    sales_only["units"] = sales_only["quantity"].abs()

    recency_cutoff = sales_only["posting_date"].max() - pd.Timedelta(days=365)
    recent_sales = sales_only[sales_only["posting_date"] >= recency_cutoff]

    item_location_share = recent_sales.groupby(["item_no", "location_code"])["units"].sum().reset_index()
    item_totals = item_location_share.groupby("item_no")["units"].transform("sum")
    item_location_share["share"] = item_location_share["units"] / item_totals

    allocated = item_location_share.merge(item_monthly_forecast, on="item_no", how="inner")
    allocated["location_forecast_units"] = allocated["share"] * allocated["predicted_units"]

    return allocated[["item_no", "location_code", "location_forecast_units"]]


def compute_stock_safety(current_stock: pd.DataFrame, incoming_stock: pd.DataFrame, allocated_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Combines current + incoming stock, compares against allocated item-level
    forecasted demand per location. Flags Safe / Not Safe.
    """
    combined = current_stock.merge(incoming_stock, on=["item_no", "location_code"], how="outer")
    combined["current_stock"] = combined["current_stock"].fillna(0)
    combined["incoming_stock"] = combined["incoming_stock"].fillna(0)

    combined = combined.merge(allocated_forecast, on=["item_no", "location_code"], how="left")
    combined["location_forecast_units"] = combined["location_forecast_units"].fillna(0)

    combined["available_stock"] = combined["current_stock"] + combined["incoming_stock"]

    combined["safety_ratio"] = combined.apply(
        lambda row: row["available_stock"] / row["location_forecast_units"] if row["location_forecast_units"] > 0 else None,
        axis=1
    )
    combined["status"] = combined.apply(
        lambda row: "Safe" if (pd.notna(row["safety_ratio"]) and row["safety_ratio"] >= 1.0)
        else ("Not Safe" if pd.notna(row["safety_ratio"]) else "No forecast"),
        axis=1
    )

    return combined.sort_values("safety_ratio")