import requests
import pandas as pd


def fetch_holidays(year: int, api_base_url: str, api_token: str) -> pd.DataFrame:
    """Fetches Sri Lankan public holidays for a given year."""
    url = f"{api_base_url}?year={year}"
    headers = {"Authorization": f"Bearer {api_token}"}

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    if not data.get("ok"):
        raise ValueError(f"API returned ok=false for year {year}")

    holidays = data["data"]["holidays"]
    df = pd.DataFrame(holidays)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = year
    return df[["date", "name", "type", "public", "bank", "mercantile", "year"]]