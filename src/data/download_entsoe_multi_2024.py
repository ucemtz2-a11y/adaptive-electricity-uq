# Download the separate 2024 files used only for the final evaluation.

import sys
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.entsoe_config import get_api_key, load_config
from src.protocol import ENTSOE_MARKETS


# Stop immediately if a final-year file cannot be downloaded completely.
def safe_download(name, query_func, out_path):
    try:
        print(f"Downloading {name}...")
        data = query_func()
        data.to_csv(out_path)
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        raise


# Use fixed 2024 UTC dates so the final dataset cannot drift with the YAML settings.
def main():
    """
    Exact extension of the original multi-market ENTSO-E downloader.

    The only intentional changes relative to the original script are:
      1. force the date range to calendar year 2024;
      2. write to data/raw/entsoe_multi_2024 so nothing from 2022-2023 is overwritten.

    API-key handling, market list, timezone handling and ENTSO-E queries are
    otherwise kept the same as the original pipeline.
    """
    config = load_config()

    api_key = get_api_key(config)
    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    raw_dir = Path("data/raw/entsoe_multi_2024")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Keep 2024 in a separate folder to protect the already checked historical data.
    for market in ENTSOE_MARKETS:
        print("\n" + "=" * 60)
        print(f"Market: {market}")
        print(f"Period: {start} -> {end}")
        print("=" * 60)

        market_dir = raw_dir / market
        market_dir.mkdir(parents=True, exist_ok=True)

        safe_download(
            name=f"{market} day-ahead prices",
            query_func=lambda m=market: client.query_day_ahead_prices(
                m, start=start, end=end
            ),
            out_path=market_dir / f"{market}_prices.csv",
        )

        safe_download(
            name=f"{market} load",
            query_func=lambda m=market: client.query_load(m, start=start, end=end),
            out_path=market_dir / f"{market}_load.csv",
        )

        safe_download(
            name=f"{market} generation",
            query_func=lambda m=market: client.query_generation(
                m, start=start, end=end
            ),
            out_path=market_dir / f"{market}_generation.csv",
        )


if __name__ == "__main__":
    main()
