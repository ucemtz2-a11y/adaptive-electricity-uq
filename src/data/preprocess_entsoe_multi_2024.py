# Module purpose: Convert multi-market 2024 ENTSO-E files into hourly modelling data.

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
    Exact copy of the original preprocessing logic.

    Some ENTSO-E generation files contain a first metadata row such as
    'Actual Aggregated'. This function separates that row from real
    timestamped observations.
    """
    raw = pd.read_csv(path, low_memory=False)

    time_col = raw.columns[0]
    raw[time_col] = pd.to_datetime(raw[time_col], utc=True, errors="coerce")

    metadata = None

    if len(raw) > 0 and raw[time_col].isna().iloc[0]:
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
    This is deliberately identical to the original pipeline:
        - drop duplicate timestamps, keep first;
        - resample hourly;
        - take arithmetic mean.
    """
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("h").mean()
    return df


# Read single series.
def read_single_series(path: Path, value_name: str) -> pd.DataFrame:
    """
    Read price or load series using the original pipeline.
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
    Extract hourly wind and solar generation exactly as in the original code.

    If a metadata row exists and contains 'Actual Aggregated', only those
    columns are retained. This avoids accidentally including consumption
    columns in generation.
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
    """
    Exact 2024 extension of the original preprocessor.

    Only the input/output directories differ from the old script:
      input : data/raw/entsoe_multi_2024/<MARKET>/
      output: data/processed/multi_market_2024/<MARKET>_dataset.csv
    """
    raw_dir = Path("data/raw/entsoe_multi_2024") / market
    out_dir = Path("data/processed/multi_market_2024")
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

    start_utc = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_utc = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    df = df.loc[(df.index >= start_utc) & (df.index < end_utc)].copy()
    
    # Preserve the original column selection and missing-value rules exactly.
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
            raise


if __name__ == "__main__":
    main()
