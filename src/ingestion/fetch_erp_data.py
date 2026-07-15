import os
import json
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()


def load_from_blob(blob_name: str = "erp/battery/battery.json") -> pd.DataFrame:
    """Read the battery ledger JSON from the bronze container."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("STORAGE_CONTAINER_BRONZE", "bronze")

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    raw_bytes = blob_client.download_blob().readall()
    data = json.loads(raw_bytes)

    # Handle the OData wrapper if present, otherwise assume a flat list
    if isinstance(data, dict) and "value" in data:
        return pd.DataFrame(data["value"])
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = load_from_blob()
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())