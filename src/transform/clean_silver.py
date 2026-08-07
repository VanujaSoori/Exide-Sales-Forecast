import pandas as pd

def clean_to_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    # Filter to battery items only
    bronze = bronze[bronze["itemCategoryCode"] == "BATTERY"].copy()

    # Filter to EXIDE, DAGENITE and LUCAS brands 
    bronze = bronze[bronze["brandCode"].isin(["EXIDE", "DAGENITE", "LUCAS"])].copy()

    # Filter to actual sales/returns only — exclude Service Shipment and Service Credit Memo
    bronze = bronze[bronze["documentType"].isin(["Sales Shipment", "Sales Return Receipt"])].copy()

    # Filter to Sri Lanka only — exclude export/foreign sales
    bronze = bronze[bronze["countryRegionCode"] == "LK"].copy()

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