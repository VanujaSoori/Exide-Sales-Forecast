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
    bronze = bronze[bronze["entryType"].isin(["Sale", "Purchase", "Transfer"])].copy()
    bronze = bronze[bronze["itemCategoryCode"] == "BATTERY"].copy()
    bronze = bronze[bronze["brandCode"].isin(["EXIDE", "DAGENITE"])].copy()
    bronze = bronze[bronze["countryRegionCode"] == "LK"].copy()

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

    # Customer resolution: prefer subCustomer fields, fall back to customer fields
    bronze["resolved_customer_no"] = bronze["subCustomerCode"].where(
        bronze["subCustomerName"].str.strip().ne(""), bronze["customerNo"]
    )
    bronze["resolved_customer_name"] = bronze["subCustomerName"].where(
        bronze["subCustomerName"].str.strip().ne(""), bronze["customerName"]
    )
    bronze["resolved_customer_address"] = bronze["subCustomerAddress"].where(
        bronze["subCustomerName"].str.strip().ne(""), bronze["customerAddress"]
    )
    bronze["resolved_customer_phone1"] = bronze["subPhoneNo1"].where(
        bronze["subCustomerName"].str.strip().ne(""), bronze["phoneNo1"]
    )
    bronze["resolved_customer_phone2"] = bronze["subPhoneNo2"].where(
        bronze["subCustomerName"].str.strip().ne(""), bronze["phoneNo2"]
    )
    bronze["resolved_customer_email"] = bronze["subEmail"].where(
        bronze["subCustomerName"].str.strip().ne(""), bronze["email"]
    )

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
        "customerCity",
    ]]

    bronze = bronze.dropna(subset=["posting_date", "quantity"])

    return bronze