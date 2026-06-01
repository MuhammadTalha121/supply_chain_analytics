"""
Synthetic M5-Like Dataset Generator
Generates sales_train_evaluation.csv, calendar.csv, sell_prices.csv
mimicking M5 Walmart schema and statistical distributions.

Demand model: Negative Binomial (intermittent demand)
Price model:  Gamma distribution
Reference:    Syntetos et al. (2005), IJF — intermittent demand classification
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

RAW_DIR    = Path("data/raw")
N_DAYS     = 730          # ~2 years
N_ITEMS    = 150          # items per store
N_STORES   = 10
START_DATE = "2020-01-01"

CATEGORIES = {
    "FOODS":      ["FOODS_1",    "FOODS_2",    "FOODS_3"],
    "HOBBIES":    ["HOBBIES_1",  "HOBBIES_2"],
    "HOUSEHOLD":  ["HOUSEHOLD_1","HOUSEHOLD_2"],
}
STORES = {
    "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
    "TX": ["TX_1", "TX_2", "TX_3"],
    "WI": ["WI_1", "WI_2", "WI_3"],
}
SNAP_DAYS_PER_MONTH = 10

EVENTS = [
    ("SuperBowl",       "Sporting"),
    ("ValentinesDay",   "Cultural"),
    ("Easter",          "Religious"),
    ("LaborDay",        "National"),
    ("Thanksgiving",    "National"),
    ("Christmas",       "Religious"),
    ("NewYear",         "National"),
    ("IndependenceDay", "National"),
]


def build_item_master() -> pd.DataFrame:
    """Build item-store hierarchy matching M5 schema."""
    rows = []
    item_counter = 1
    for cat, depts in CATEGORIES.items():
        items_per_dept = N_ITEMS // len(depts)
        for dept in depts:
            for _ in range(items_per_dept):
                item_id = f"{dept}_{item_counter:03d}"
                for state, stores in STORES.items():
                    for store in stores:
                        rows.append({
                            "id":       f"{item_id}_{store}_evaluation",
                            "item_id":  item_id,
                            "dept_id":  dept,
                            "cat_id":   cat,
                            "store_id": store,
                            "state_id": state,
                        })
                item_counter += 1
    return pd.DataFrame(rows)


def build_calendar() -> pd.DataFrame:
    """Build calendar table with events and SNAP flags."""
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    df    = pd.DataFrame({"date": dates})

    df["d"]        = [f"d_{i+1}" for i in range(N_DAYS)]
    df["wm_yr_wk"] = (df["date"].dt.year * 100 + df["date"].dt.isocalendar().week).astype(int)
    df["weekday"]  = df["date"].dt.day_name()
    df["wday"]     = df["date"].dt.dayofweek + 1
    df["month"]    = df["date"].dt.month
    df["year"]     = df["date"].dt.year

    event_rows_idx = np.random.choice(N_DAYS, size=len(EVENTS) * 2, replace=False)
    df["event_name_1"] = np.nan
    df["event_type_1"] = np.nan
    for i, (name, etype) in enumerate(EVENTS):
        df.loc[event_rows_idx[i * 2], "event_name_1"] = name
        df.loc[event_rows_idx[i * 2], "event_type_1"] = etype

    for state in ["CA", "TX", "WI"]:
        snap_col = f"snap_{state}"
        df[snap_col] = 0
        for _, grp in df.groupby([df["date"].dt.year, df["date"].dt.month]):
            snap_idx = np.random.choice(grp.index, size=min(SNAP_DAYS_PER_MONTH, len(grp)), replace=False)
            df.loc[snap_idx, snap_col] = 1

    return df


def build_prices(item_master: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Generate weekly sell prices per item-store using Gamma distribution."""
    weeks     = calendar["wm_yr_wk"].unique()
    all_items = item_master[["item_id", "store_id"]].drop_duplicates()
    rows      = []

    base_prices = {item: np.random.gamma(shape=3.0, scale=2.5)
                   for item in all_items["item_id"].unique()}

    for _, row in all_items.iterrows():
        base = base_prices[row["item_id"]]
        for wk in weeks:
            noise = np.random.normal(0, base * 0.05)
            rows.append({
                "store_id":   row["store_id"],
                "item_id":    row["item_id"],
                "wm_yr_wk":   wk,
                "sell_price": round(max(0.5, base + noise), 2),
            })
    return pd.DataFrame(rows)


def simulate_demand(
    n_days:     int,
    base_rate:  float,
    weekly_amp: float,
    yearly_amp: float,
    is_event:   np.ndarray,
    price_idx:  np.ndarray,
) -> np.ndarray:
    """
    Simulate daily demand using Negative Binomial with seasonal effects.

    Args:
        n_days:      Number of days to simulate.
        base_rate:   Mean demand baseline.
        weekly_amp:  Weekly seasonality amplitude.
        yearly_amp:  Yearly seasonality amplitude.
        is_event:    Binary array flagging event days.
        price_idx:   Price index (1.0 = average price) — higher price lowers demand.

    Returns:
        Integer array of daily sales.
    """
    t = np.arange(n_days)

    weekly  = 1 + weekly_amp * np.sin(2 * np.pi * t / 7)
    yearly  = 1 + yearly_amp * np.sin(2 * np.pi * t / 365)
    event   = 1 + 0.4 * is_event
    price   = np.clip(1.0 / price_idx, 0.5, 2.0)

    mu       = np.maximum(base_rate * weekly * yearly * event * price, 0.1)
    dispersion = 0.5
    p        = dispersion / (dispersion + mu)
    sales    = np.random.negative_binomial(dispersion, p)
    return sales.astype(np.int16)


def build_sales(
    item_master: pd.DataFrame,
    calendar:    pd.DataFrame,
    prices:      pd.DataFrame,
) -> pd.DataFrame:
    """Build wide-format sales table matching M5 schema."""
    day_cols    = calendar["d"].tolist()
    is_event    = (calendar["event_name_1"].notna()).astype(int).values
    wk_map      = dict(zip(calendar["d"], calendar["wm_yr_wk"]))

    price_lookup = prices.set_index(["item_id", "store_id", "wm_yr_wk"])["sell_price"].to_dict()

    records = []
    total   = len(item_master)

    for i, (_, row) in enumerate(item_master.iterrows()):
        if i % 500 == 0:
            print(f"  Generating series {i}/{total}...")

        base_rate   = np.random.gamma(shape=2.0, scale=3.0)
        weekly_amp  = np.random.uniform(0.1, 0.4)
        yearly_amp  = np.random.uniform(0.05, 0.25)

        price_series = np.array([
            price_lookup.get((row["item_id"], row["store_id"], wk_map[d]), base_rate * 2)
            for d in day_cols
        ])
        avg_price       = price_series.mean()
        price_idx       = price_series / avg_price

        sales = simulate_demand(N_DAYS, base_rate, weekly_amp, yearly_amp, is_event, price_idx)

        record = {
            "id":       row["id"],
            "item_id":  row["item_id"],
            "dept_id":  row["dept_id"],
            "cat_id":   row["cat_id"],
            "store_id": row["store_id"],
            "state_id": row["state_id"],
        }
        for d, s in zip(day_cols, sales):
            record[d] = s

        records.append(record)

    return pd.DataFrame(records)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Building item master...")
    item_master = build_item_master()
    print(f"  {len(item_master):,} item-store series")

    print("Building calendar...")
    calendar = build_calendar()
    calendar.to_csv(RAW_DIR / "calendar.csv", index=False)
    print(f"  Saved calendar.csv ({N_DAYS} days)")

    print("Building prices...")
    prices = build_prices(item_master, calendar)
    prices.to_csv(RAW_DIR / "sell_prices.csv", index=False)
    print(f"  Saved sell_prices.csv ({len(prices):,} rows)")

    print("Simulating demand (this takes ~1-2 min)...")
    sales = build_sales(item_master, calendar, prices)
    sales.to_csv(RAW_DIR / "sales_train_evaluation.csv", index=False)
    print(f"  Saved sales_train_evaluation.csv ({len(sales):,} rows x {len(sales.columns)} cols)")

    print("\nSample output (last 5 rows, first 10 cols):")
    print(sales.iloc[-5:, :10].to_string())
    print("\nDone. Files saved to data/raw/")


if __name__ == "__main__":
    main()
