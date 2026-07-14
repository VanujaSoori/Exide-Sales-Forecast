import pandas as pd
import yaml

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def clean_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Document Type"] == "Sales Shipment"].copy()
    df = df[["Posting Date", "Item Category Code", "Country/Region Code",
             "Site/Province", "Shipped Qty.", "Cost"]]

    df["Posting Date"] = pd.to_datetime(df["Posting Date"])
    df["Shipped Qty."] = pd.to_numeric(df["Shipped Qty."], errors="coerce")
    df = df.dropna(subset=["Posting Date", "Shipped Qty."])

    return df

if __name__ == "__main__":
    from src.ingestion.fetch_excel_data import load_from_blob
    raw = load_from_blob("sales_export.xlsx")
    silver = clean_to_silver(raw)
    silver.to_parquet("silver/sales_clean.parquet", index=False)
    print(f"Wrote {len(silver)} rows to silver")