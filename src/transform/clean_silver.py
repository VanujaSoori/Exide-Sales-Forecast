import pandas as pd


def clean_to_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    # Filter to battery items only
    bronze = bronze[bronze["itemCategoryCode"] == "BATTERY"].copy()

    # Filter to EXIDE brand only
    bronze = bronze[bronze["brandCode"] == "EXIDE"].copy()

    # quantity: negative = sale, e.g. -2 means 2 units sold
    bronze["units_sold"] = bronze["quantity"].abs()

    # Types
    bronze["postingDate"] = pd.to_datetime(bronze["postingDate"])

    # Rename for clarity downstream
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

    # Keep everything relevant for now — narrow further once you've reviewed brand-level detail
    bronze = bronze[[
        "posting_date", "item_no", "itemCategoryCode",
        "vehicle_type", "location_code", "location_description",
        "sales_person_code", "salesperson_name",
        "brand_code", "brand_description",
        "documentType",
        "units_sold", "costAmountActual", "salesAmountActual"
    ]]

    # Drop rows missing critical fields
    bronze = bronze.dropna(subset=["posting_date", "units_sold"])

    return bronze