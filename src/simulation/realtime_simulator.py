import time
import random
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "Data" / "processed" / "features.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_simulation_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"features.csv not found at:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Convert date
    if "interval_start" in df.columns:
        df["interval_start"] = pd.to_datetime(
            df["interval_start"],
            errors="coerce"
        )

    # Sort
    if "equipment_ID" in df.columns and "interval_start" in df.columns:
        df = df.sort_values(
            ["equipment_ID", "interval_start"]
        )

    return df.reset_index(drop=True)


# ============================================================
# SIMULATE ONE LIVE RECORD
# ============================================================

def simulate_record(row):

    record = row.to_dict()

    # Add small random variation to simulate live sensor movement
    if "%production" in record:
        record["%production"] = max(
            0,
            min(
                1,
                float(record["%production"])
                + random.uniform(-0.02, 0.02)
            )
        )

    if "%downtime" in record:
        record["%downtime"] = max(
            0,
            min(
                1,
                float(record["%downtime"])
                + random.uniform(-0.01, 0.01)
            )
        )

    if "%idle" in record:
        record["%idle"] = max(
            0,
            min(
                1,
                float(record["%idle"])
                + random.uniform(-0.01, 0.01)
            )
        )

    if "%performance_loss" in record:
        record["%performance_loss"] = max(
            0,
            min(
                1,
                float(record["%performance_loss"])
                + random.uniform(-0.01, 0.01)
            )
        )

    # Recalculate project health score
    production = record.get("%production", 0)
    downtime = record.get("%downtime", 0)
    idle = record.get("%idle", 0)
    performance_loss = record.get("%performance_loss", 0)

    record["health_score"] = max(
        0,
        min(
            100,
            100
            - downtime
            - idle
            - performance_loss
        )
    )

    return record


# ============================================================
# SIMPLE OPERATIONAL STATUS
# ============================================================

def determine_status(record):

    production = float(record.get("%production", 0))
    downtime = float(record.get("%downtime", 0))
    performance_loss = float(
        record.get("%performance_loss", 0)
    )

    if downtime >= 0.50:
        return "CRITICAL"

    elif production <= 0.30:
        return "WARNING"

    elif performance_loss >= 0.30:
        return "WARNING"

    else:
        return "HEALTHY"


# ============================================================
# REAL-TIME GENERATOR
# ============================================================

def realtime_stream(
    machine_id=None,
    interval_seconds=2,
    number_of_records=20
):

    df = load_simulation_data()

    # Filter machine
    if machine_id is not None:

        df = df[
            df["equipment_ID"].astype(str)
            == str(machine_id)
        ].copy()

    if df.empty:
        raise ValueError(
            f"No records found for machine: {machine_id}"
        )

    # Random starting point
    if len(df) > number_of_records:
        start_index = random.randint(
            0,
            len(df) - number_of_records
        )

        simulation_df = df.iloc[
            start_index:
            start_index + number_of_records
        ]

    else:
        simulation_df = df

    print("=" * 70)
    print("PACKAGING AI OPERATIONS CONTROL TOWER")
    print("REAL-TIME SIMULATION STARTED")
    print("=" * 70)

    for _, row in simulation_df.iterrows():

        live_record = simulate_record(row)

        machine = live_record.get(
            "equipment_ID",
            "UNKNOWN"
        )

        timestamp = live_record.get(
            "interval_start",
            "UNKNOWN"
        )

        status = determine_status(
            live_record
        )

        production = float(
            live_record.get("%production", 0)
        )

        downtime = float(
            live_record.get("%downtime", 0)
        )

        health = float(
            live_record.get("health_score", 0)
        )

        print()
        print("-" * 70)

        print(
            f"Machine       : {machine}"
        )

        print(
            f"Timestamp     : {timestamp}"
        )

        print(
            f"Production    : {production:.2%}"
        )

        print(
            f"Downtime      : {downtime:.2%}"
        )

        print(
            f"Health Score  : {health:.2f}"
        )

        print(
            f"Status        : {status}"
        )

        print("-" * 70)

        time.sleep(interval_seconds)

    print()
    print("=" * 70)
    print("REAL-TIME SIMULATION COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    realtime_stream(
        machine_id="s_1",
        interval_seconds=2,
        number_of_records=20
    )