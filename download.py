"""Download the public 'Daily News for Stock Market Prediction' table: top-25 Reddit r/worldnews
headlines per day (2008-2016) with a label = 1 if the Dow Jones closed up/flat vs the prior day."""
import urllib.request
from config import RAW_CSV, DATA_URL

def main():
    RAW_CSV.parent.mkdir(exist_ok=True)
    if RAW_CSV.exists(): print("already downloaded"); return
    urllib.request.urlretrieve(DATA_URL, RAW_CSV); print(f"downloaded {RAW_CSV.stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
