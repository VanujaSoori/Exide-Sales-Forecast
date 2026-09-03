import pandas as pd


def clean_to_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasting silver - Sale/Shipment only, net of anomalies.
    Used by the daily forecasting pipeline (06_build_gold_live onward).
    """
    bronze = bronze[bronze["entryType"] == "Sale"].copy()

    # Filter to battery items only
    bronze = bronze[bronze["itemCategoryCode"] == "BATTERY"].copy()

    # Filter to EXIDE, DAGENITE brands
    bronze = bronze[bronze["brandCode"].isin(["EXIDE", "DAGENITE"])].copy()

    # documentType values are OData-encoded (spaces become _x0020_)
    bronze = bronze[bronze["documentType"].isin([
        "Sales_x0020_Shipment", "Sales_x0020_Return_x0020_Receipt"
    ])].copy()

    # Filter to Sri Lanka only - exclude export/foreign sales
    bronze = bronze[bronze["countryRegionCode"] == "LK"].copy()

    # Remove anomalous zero-value transactions with a quantity sign that
    # contradicts their document type (bad data / voided entries)
    anomaly_shipment = (
        (bronze["salesAmountActual"] == 0)
        & (bronze["documentType"] == "Sales_x0020_Shipment")
        & (bronze["quantity"] > 0)
    )
    anomaly_return = (
        (bronze["salesAmountActual"] == 0)
        & (bronze["documentType"] == "Sales_x0020_Return_x0020_Receipt")
        & (bronze["quantity"] < 0)
    )
    removed_count = (anomaly_shipment | anomaly_return).sum()
    bronze = bronze[~(anomaly_shipment | anomaly_return)].copy()
    print(f"Removed {removed_count} anomalous zero-value transactions")

    # Forecasting uses actual shipments only - returns excluded from model training
    bronze = bronze[bronze["documentType"] == "Sales_x0020_Shipment"].copy()

    bronze["gross_units"] = bronze["quantity"].abs()
    bronze["net_units"] = -bronze["quantity"]
    bronze["profit"] = bronze["salesAmountActual"] + bronze["costAmountActual"]
    bronze["postingDate"] = pd.to_datetime(bronze["postingDate"])

    bronze = bronze.rename(columns={
        "itemCategory2": "vehicle_type",
        "locationCode": "location_code",
        "locationDescription": "location_description",
        "postingDate": "posting_date",
        "itemNo": "item_no",
        "salesPersonCode": "sales_person_code",
        "salespersonName": "salesperson_name",
        "brandCode": "brand_code",
        "brandDescription": "brand_description",
    })

    bronze = bronze[[
        "posting_date", "item_no", "itemCategoryCode",
        "vehicle_type", "location_code", "location_description",
        "sales_person_code", "salesperson_name",
        "brand_code", "brand_description",
        "documentType",
        "gross_units", "net_units", "profit",
        "costAmountActual", "salesAmountActual"
    ]]

    bronze = bronze.dropna(subset=["posting_date", "net_units"])

    return bronze


def clean_to_silver_analysis(bronze: pd.DataFrame) -> pd.DataFrame:
    """
    Broader cleaning for analysis/fraud detection - keeps Sale (shipments + returns),
    Purchase, and Transfer entries, plus resolved customer identity and lot tracking.
    Separate from clean_to_silver(), which is forecasting-only (Sale/Shipment).
    """
    bronze = bronze[bronze["entryType"].isin(["Sale", "Purchase", "Transfer"])].copy()
    bronze = bronze[bronze["itemCategoryCode"] == "BATTERY"].copy()
    bronze = bronze[bronze["brandCode"].isin(["EXIDE", "DAGENITE"])].copy()
    bronze = bronze[bronze["countryRegionCode"] == "LK"].copy()

    # Remove anomalous zero-value transactions with a contradictory quantity sign
    anomaly_shipment = (
        (bronze["salesAmountActual"] == 0)
        & (bronze["documentType"] == "Sales_x0020_Shipment")
        & (bronze["quantity"] > 0)
    )
    anomaly_return = (
        (bronze["salesAmountActual"] == 0)
        & (bronze["documentType"] == "Sales_x0020_Return_x0020_Receipt")
        & (bronze["quantity"] < 0)
    )
    bronze = bronze[~(anomaly_shipment | anomaly_return)].copy()

    bronze["postingDate"] = pd.to_datetime(bronze["postingDate"])

    bronze["has_sub_customer"] = bronze["subCustomerName"].astype(str).str.strip() != ""
    bronze["subCustomerName_clean"] = bronze["subCustomerName"].astype(str).str.strip().str.upper()

    bronze["resolved_customer_no"] = bronze["subCustomerCode"].where(
        bronze["has_sub_customer"] & (bronze["subCustomerCode"].astype(str).str.strip() != ""),
        bronze["subCustomerName_clean"].where(bronze["has_sub_customer"], bronze["customerNo"])
    )
    bronze["resolved_customer_name"] = bronze["subCustomerName"].where(
        bronze["has_sub_customer"], bronze["customerName"]
    )
        # Build a combined full address from the main customer fields
    bronze["customer_full_address"] = (
        bronze["customerAddress"].astype(str).str.strip() + ", " +
        bronze["customerAddress2"].astype(str).str.strip() + ", " +
        bronze["customerCity"].astype(str).str.strip()
    )
    # Clean up cases where one or more parts were empty (avoid dangling ", ,")
    bronze["customer_full_address"] = (
        bronze["customer_full_address"]
        .str.replace(r",\s*,", ",", regex=True)
        .str.replace(r"^,\s*|,\s*$", "", regex=True)
        .str.strip()
    )

    bronze["resolved_customer_address"] = bronze["subCustomerAddress"].where(
        bronze["has_sub_customer"], bronze["customer_full_address"]
    )
    bronze["resolved_customer_address"] = bronze["subCustomerAddress"].where(
    bronze["has_sub_customer"], bronze["customer_full_address"]
    )
    bronze["resolved_customer_address"] = bronze["resolved_customer_address"].str.replace(r"\s+", " ", regex=True).str.strip()
    bronze["resolved_customer_phone1"] = bronze["subPhoneNo1"].where(
        bronze["has_sub_customer"], bronze["phoneNo1"]
    )
    bronze["resolved_customer_phone2"] = bronze["subPhoneNo2"].where(
        bronze["has_sub_customer"], bronze["phoneNo2"]
    )
    bronze["resolved_customer_email"] = bronze["subEmail"].where(
        bronze["has_sub_customer"], bronze["email"]
    )

    GENERIC_CUSTOMER_NAMES = {
        "CASH CUSTOMER", "DEALER- CASH CUSTOMER", "HYBRID CASH CUSTOMER", "CORPORATE - CASH CUSTOMER"
    }
    bronze["is_identifiable_customer"] = ~bronze["resolved_customer_name"].astype(str).str.strip().str.upper().isin(GENERIC_CUSTOMER_NAMES)

    bronze = bronze.rename(columns={
        "itemCategory2": "vehicle_type",
        "locationCode": "location_code",
        "locationDescription": "location_description",
        "postingDate": "posting_date",
        "itemNo": "item_no",
        "salesPersonCode": "sales_person_code",
        "salespersonName": "salesperson_name",
        "brandCode": "brand_code",
        "brandDescription": "brand_description",
        "LotNo": "lot_no",
        "customerCity": "customer_city",
    })

    bronze = bronze[[
        "posting_date", "item_no", "itemCategoryCode",
        "vehicle_type", "location_code", "location_description",
        "sales_person_code", "salesperson_name",
        "brand_code", "brand_description",
        "entryType", "documentType", "lot_no",
        "quantity", "costAmountActual", "salesAmountActual",
        "resolved_customer_no", "resolved_customer_name", "resolved_customer_address",
        "resolved_customer_phone1", "resolved_customer_phone2", "resolved_customer_email",
        "customer_city", "is_identifiable_customer",
    ]]

    bronze = bronze.dropna(subset=["posting_date", "quantity"])

    return bronze