import io
import json
import pandas as pd
from azure.storage.blob import BlobServiceClient


def get_blob_service(storage_account_name, storage_account_key):
    conn_str = (
        f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};"
        f"AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
    )
    return BlobServiceClient.from_connection_string(conn_str)


def read_bronze(blob_service, blob_path):
    blob_client = blob_service.get_blob_client(container="bronze", blob=blob_path)
    stream = blob_client.download_blob().readall()
    data = json.loads(stream)

    # Case 1: top-level dict with a "value" key (single-page OData response)
    if isinstance(data, dict) and "value" in data:
        return pd.DataFrame(data["value"])

    # Case 2: list containing one wrapper dict (what some daily pulls produced)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "value" in data[0]:
        return pd.DataFrame(data[0]["value"])

    # Case 3: already a flat list of real records
    return pd.DataFrame(data)


def save_silver(blob_service, df, blob_path):
    payload = df.to_json(orient="records", date_format="iso", lines=False)
    blob_client = blob_service.get_blob_client(container="silver", blob=blob_path)
    blob_client.upload_blob(payload, overwrite=True)


def read_silver(blob_service, blob_path):
    blob_client = blob_service.get_blob_client(container="silver", blob=blob_path)
    stream = blob_client.download_blob().readall()
    return pd.DataFrame(json.loads(stream))


def save_gold(blob_service, df, blob_path):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    blob_client = blob_service.get_blob_client(container="gold", blob=blob_path)
    blob_client.upload_blob(buffer, overwrite=True)


def read_gold(blob_service, blob_path):
    blob_client = blob_service.get_blob_client(container="gold", blob=blob_path)
    stream = blob_client.download_blob().readall()
    return pd.read_parquet(io.BytesIO(stream))

def append_json_history(blob_service, new_records, blob_path):
    """Reads existing history (if any), appends new records, writes back."""
    blob_client = blob_service.get_blob_client(container="gold", blob=blob_path)
    try:
        stream = blob_client.download_blob().readall()
        existing = json.loads(stream)
    except Exception:
        existing = []

    combined = existing + new_records
    blob_client.upload_blob(json.dumps(combined, indent=2), overwrite=True)
    return combined


def read_json_history(blob_service, blob_path):
    blob_client = blob_service.get_blob_client(container="gold", blob=blob_path)
    try:
        stream = blob_client.download_blob().readall()
        return json.loads(stream)
    except Exception:
        return []
    
def save_history_as_excel(blob_service, history_records, blob_path):
    """Converts a list of forecast history records to Excel and saves to blob storage."""
    df = pd.DataFrame(history_records)
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    blob_client = blob_service.get_blob_client(container="gold", blob=blob_path)
    blob_client.upload_blob(buffer, overwrite=True)
    return len(df)