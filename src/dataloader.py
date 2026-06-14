from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "src" / "data"
SPX_RAW_PATH = DATA_DIR / "spx_raw.csv"
VIX_RAW_PATH = DATA_DIR / "vix_raw.csv"
MASTER_DATASET_PATH = DATA_DIR / "master_dataset.csv"


def _normalize_download_frame(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].rename(columns={"Close": column_name})


def download_raw_data(start: str = "2010-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading S&P 500 data...")
    spx = _normalize_download_frame(yf.download("^GSPC", start=start), "spx_close")
    spx.to_csv(SPX_RAW_PATH)

    print("Downloading VIX data...")
    vix = _normalize_download_frame(yf.download("^VIX", start=start), "vix_close")
    vix.to_csv(VIX_RAW_PATH)

    print(f"Saved raw datasets to {DATA_DIR}")
    return spx, vix


def build_master_dataset(
    spx_path: Path = SPX_RAW_PATH,
    vix_path: Path = VIX_RAW_PATH,
    output_path: Path = MASTER_DATASET_PATH,
) -> pd.DataFrame:
    spx = pd.read_csv(spx_path, index_col="Date", parse_dates=True)
    vix = pd.read_csv(vix_path, index_col="Date", parse_dates=True)

    merged = pd.merge(spx, vix, left_index=True, right_index=True, how="inner").sort_index()
    merged["spx_log_ret"] = np.log(merged["spx_close"] / merged["spx_close"].shift(1))
    merged["spx_roll_std"] = merged["spx_log_ret"].rolling(window=30).std()
    merged["spx_realized_vol"] = merged["spx_roll_std"] * np.sqrt(252)
    merged["vix_decimal"] = merged["vix_close"] / 100.0
    merged["forward_realized_vol"] = merged["spx_realized_vol"].shift(-30)

    master_dataset = merged.dropna(subset=["forward_realized_vol"])[
        ["spx_close", "vix_decimal", "forward_realized_vol"]
    ].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_dataset.to_csv(output_path)

    print(f"Built master dataset at {output_path}")
    print(f"Rows: {len(master_dataset):,}")
    return master_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw VRP inputs and build the Streamlit master dataset."
    )
    parser.add_argument(
        "--download-raw",
        action="store_true",
        help="Download fresh SPX and VIX closes into src/data.",
    )
    parser.add_argument(
        "--build-master-dataset",
        action="store_true",
        help="Build src/data/master_dataset.csv from the tracked raw inputs.",
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date for yfinance downloads when --download-raw is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_download = args.download_raw
    run_build = args.build_master_dataset

    if not run_download and not run_build:
        run_build = True

    if run_download:
        download_raw_data(start=args.start)

    if run_build:
        build_master_dataset()


if __name__ == "__main__":
    main()

