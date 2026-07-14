import os
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

def load_from_local(path: str) -> pd.DataFrame:
    """Prototype path: read Excel directly from local disk."""
    return pd.read_excel(path)

def load_from_blob(blob_name: str = "exide/sales.xlsx") -> pd.DataFrame:
    """Production-shaped path: read Excel from the bronze container."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("STORAGE_CONTAINER_BRONZE", "bronze")

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    stream = blob_client.download_blob().readall()
    return pd.read_excel(stream)

if __name__ == "__main__":
    df = load_from_blob("exide/sales.xlsx")
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())