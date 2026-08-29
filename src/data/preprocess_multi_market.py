# Module purpose: Clean and merge multi-market ENTSO-E files into modelling data.

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol import ENTSOE_MARKETS


# Read ENTSO-E CSV.
def read_entsoe_csv(path: Path) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Read an ENTSO-E CSV file robustly.

    Some ENTSO-E generation files contain a first metadata row such as
    'Actual Aggregated'. This function separates that row from real
    timestamped observations.
    """
    raw = pd.read_csv(path, low_memory=False)

    time_col = raw.columns[0]
    raw[time_col] = pd.to_datetime(raw[time_col], utc=True, errors="coerce")

    metadata = None

    # Skip the first row as ENTSO-E metadata when it does not contain a timestamp.
    if raw[time_col].isna().iloc[0]:
        metadata = raw.iloc[0].copy()

    raw = raw.dropna(subset=[time_col])
    raw = raw.set_index(time_col)
    raw = raw.sort_index()

    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return raw, metadata


# To hourly.
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert quarter-hourly or hourly data to hourly frequency.
    If already hourly, this keeps the same frequency.
    """
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("h").mean()
    return df


# Read single series.
def read_single_series(path: Path, value_name: str) -> pd.DataFrame:
    """
    Read price or load series.

    If the data are quarter-hourly, convert them to hourly average.
    """
    df, _ = read_entsoe_csv(path)
    df = to_hourly(df)

    if df.shape[1] == 1:
        out = df.copy()
        out.columns = [value_name]
    else:
        out = pd.DataFrame(index=df.index)
        out[value_name] = df.mean(axis=1)

    return out


# Read generation.
def read_generation(path: Path) -> pd.DataFrame:
    """
    Extract hourly wind and solar generation.

    For files with metadata rows, use only columns marked as
    'Actual Aggregated'. This avoids adding 'Actual Consumption' columns
    such as Wind Onshore.1 or Solar.1 when they are not generation.
    """
    gen, metadata = read_entsoe_csv(path)

    valid_cols = list(gen.columns)

    if metadata is not None:
        actual_aggregated_cols = []

        for col in gen.columns:
            meta_value = str(metadata.get(col, ""))
            if "Actual Aggregated" in meta_value:
                actual_aggregated_cols.append(col)

        if actual_aggregated_cols:
            valid_cols = actual_aggregated_cols

    wind_cols = [c for c in valid_cols if "Wind" in str(c) or "wind" in str(c)]

    solar_cols = [c for c in valid_cols if "Solar" in str(c) or "solar" in str(c)]

    out = pd.DataFrame(index=gen.index)

    if wind_cols:
        out["wind"] = gen[wind_cols].sum(axis=1)
    else:
        out["wind"] = 0.0

    if solar_cols:
        out["solar"] = gen[solar_cols].sum(axis=1)
    else:
        out["solar"] = 0.0

    out = to_hourly(out)

    return out


# Preprocess market.
def preprocess_market(market: str):
    raw_dir = Path("data/raw/entsoe_multi") / market
    out_dir = Path("data/processed/multi_market")
    out_dir.mkdir(parents=True, exist_ok=True)

    price_path = raw_dir / f"{market}_prices.csv"
    load_path = raw_dir / f"{market}_load.csv"
    generation_path = raw_dir / f"{market}_generation.csv"

    if not price_path.exists():
        raise FileNotFoundError(f"Missing price file: {price_path}")
    if not load_path.exists():
        raise FileNotFoundError(f"Missing load file: {load_path}")
    if not generation_path.exists():
        raise FileNotFoundError(f"Missing generation file: {generation_path}")

    print(f"\nProcessing {market}...")

    prices = read_single_series(price_path, "price")
    load = read_single_series(load_path, "load")
    generation = read_generation(generation_path)

    df = prices.join(load, how="outer").join(generation, how="outer")
    df = df.sort_index()

    df = df.ffill()

    for col in ["price", "load", "wind", "solar"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "load", "wind", "solar"])

    df["residual_load"] = df["load"] - df["wind"] - df["solar"]

    df = df.dropna(subset=["price", "load", "wind", "solar", "residual_load"])

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "datetime"})

    out_path = out_dir / f"{market}_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Start: {df['datetime'].min()}")
    print(f"End:   {df['datetime'].max()}")
    print(df.head())

    return out_path


# Main.
def main():
    for market in ENTSOE_MARKETS:
        try:
            preprocess_market(market)
        except Exception as e:
            print(f"Failed to process {market}: {e}")


if __name__ == "__main__":
    main()
