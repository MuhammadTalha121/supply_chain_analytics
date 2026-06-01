"""
Supply Chain Intelligence System — Data Pipeline
Day 1: Download, validate, chunk, and downcast the M5 Walmart Retail Dataset.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load project configuration.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Dictionary of config values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        Memory-optimised DataFrame.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_sales(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Load and melt the wide-format sales file into long format.

    Args:
        raw_dir: Path to raw data directory.
        nrows:   Optional row limit for testing.

    Returns:
        Long-format sales DataFrame with columns [id, d, sales].

    Raises:
        FileNotFoundError: If sales file is missing.
    """
    path = raw_dir / "sales_train_evaluation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path}")

    logger.info("Loading sales data...")
    df = pd.read_csv(path, nrows=nrows)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    df_long = df.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    df_long = downcast_dataframe(df_long)

    logger.info(f"Sales shape (long): {df_long.shape}")
    return df_long


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    """
    Load and clean the calendar file.

    Args:
        raw_dir: Path to raw data directory.

    Returns:
        Calendar DataFrame with date, event, and SNAP columns.

    Raises:
        FileNotFoundError: If calendar file is missing.
    """
    path = raw_dir / "calendar.csv"
    if not path.exists():
        raise FileNotFoundError(f"Calendar file not found: {path}")

    logger.info("Loading calendar data...")
    df = pd.read_csv(path, parse_dates=["date"])
    df = downcast_dataframe(df)
    logger.info(f"Calendar shape: {df.shape}")
    return df


def load_prices(raw_dir: Path) -> pd.DataFrame:
    """
    Load the sell prices file.

    Args:
        raw_dir: Path to raw data directory.

    Returns:
        Sell prices DataFrame.

    Raises:
        FileNotFoundError: If prices file is missing.
    """
    path = raw_dir / "sell_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prices file not found: {path}")

    logger.info("Loading prices data...")
    df = pd.read_csv(path)
    df = downcast_dataframe(df)
    logger.info(f"Prices shape: {df.shape}")
    return df


def validate_dataset(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> None:
    """
    Run basic integrity checks on the three source tables.

    Args:
        sales:    Long-format sales DataFrame.
        calendar: Calendar DataFrame.
        prices:   Sell prices DataFrame.

    Raises:
        AssertionError: If any validation check fails.
    """
    logger.info("Running validation checks...")

    assert sales["sales"].isna().sum() == 0,    "Null sales values detected."
    assert sales["sales"].min() >= 0,            "Negative sales values detected."
    assert calendar["d"].nunique() == len(calendar), "Duplicate day keys in calendar."
    assert prices["sell_price"].min() > 0,       "Non-positive prices detected."

    logger.info("All validation checks passed.")


def merge_and_save(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    processed_dir: Path,
    chunk_size: int,
) -> None:
    """
    Merge sales with calendar and prices, then save in chunks to avoid OOM errors.

    Args:
        sales:         Long-format sales DataFrame.
        calendar:      Calendar DataFrame.
        prices:        Sell prices DataFrame.
        processed_dir: Output directory for processed chunks.
        chunk_size:    Number of rows per saved chunk.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging calendar into sales...")
    df = sales.merge(calendar[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]],
                     on="d", how="left")

    logger.info("Merging prices into sales...")
    df["wm_yr_wk"]     = df["wm_yr_wk"].astype("int64")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int64")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df["date"] = pd.to_datetime(df["date"])
    df = downcast_dataframe(df)

    total_chunks = (len(df) // chunk_size) + 1
    logger.info(f"Saving {total_chunks} chunks to {processed_dir}...")

    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start: start + chunk_size]
        chunk.to_parquet(processed_dir / f"chunk_{i:04d}.parquet", index=False)

    logger.info("All chunks saved.")
    logger.info("Sample (last 5 rows):")
    print(df.tail(5).to_string())


def run_pipeline(config_path: str = "config.yaml", test_mode: bool = False) -> None:
    """
    Execute the full Day 1 data pipeline.

    Args:
        config_path: Path to config.yaml.
        test_mode:   If True, loads only a small subset for fast testing.
    """
    cfg        = load_config(config_path)
    raw_dir    = Path(cfg["paths"]["raw_data"])
    proc_dir   = Path(cfg["paths"]["processed_data"])
    chunk_size = cfg["pipeline"]["chunk_size"]
    nrows      = cfg["pipeline"]["test_nrows"] if test_mode else None

    sales    = load_sales(raw_dir, nrows=nrows)
    calendar = load_calendar(raw_dir)
    prices   = load_prices(raw_dir)

    validate_dataset(sales, calendar, prices)
    merge_and_save(sales, calendar, prices, proc_dir, chunk_size)

    logger.info("Day 1 pipeline complete.")


if __name__ == "__main__":
    run_pipeline(test_mode=False)
