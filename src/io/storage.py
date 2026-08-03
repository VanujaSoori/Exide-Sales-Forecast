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
    return pd.DataFrame(data.get("value", data)) if isinstance(data, dict) else pd.DataFrame(data)


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