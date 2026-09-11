import requests
import pandas as pd

def get_crypto_raw_data(coin_id="bitcoin", vs_currency="usd", days=365):
    """Mengambil data mentah pasar dari CoinGecko API."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    headers = {"accept": "application/json"}

    res = requests.get(url, params=params, headers=headers, timeout=15)
    res.raise_for_status()
    data = res.json()

    df_price = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df_vol = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume_24h"])
    df_mc = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])

    df = df_price.merge(df_vol, on="timestamp").merge(df_mc, on="timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    return df[["datetime", "price", "volume_24h", "market_cap"]]

if __name__ == "__main__":
    df = get_crypto_raw_data()
    df.to_csv("data/raw/bitcoin_raw_market_data.csv", index=False)
    print("Data mentah berhasil disimpan di data/raw/bitcoin_raw_market_data.csv")
