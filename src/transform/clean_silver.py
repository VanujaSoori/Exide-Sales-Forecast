import pandas as pd


def clean_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to battery items only
    df = df[df["itemCategoryCode"] == "BATTERY"].copy()

    # Filter to EXIDE brand only
    df = df[df["brandCode"] == "EXIDE"].copy()

    # quantity: negative = sale, e.g. -2 means 2 units sold
    df["units_sold"] = df["quantity"].abs()

    # Types
    df["postingDate"] = pd.to_datetime(df["postingDate"])

    # Rename for clarity downstream
    df = df.rename(columns={
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
    df = df[[
        "posting_date", "item_no", "itemCategoryCode",
        "vehicle_type", "location_code", "location_description",
        "sales_person_code", "salesperson_name",
        "brand_code", "brand_description",
        "documentType",
        "units_sold", "costAmountActual", "salesAmountActual"
    ]]

    # Drop rows missing critical fields
    df = df.dropna(subset=["posting_date", "units_sold"])

    return df