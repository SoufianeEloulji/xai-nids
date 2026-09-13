"""
Simulates network traffic by reading flows from a CSV file and sending them to a specified URL as JSON payloads. 
The script allows for configurable delays between requests, shuffling of the input data, and looping through the dataset.

Usage:
    python simulator/simulate_traffic.py \
        --csv data/traffic_sample.csv \
        --url http://localhost:8000/predict \
        --min-delay 0.3 --max-delay 2.0 \
        --shuffle --loop
"""
import argparse
import random
import re
import time
import sys
import requests
import pandas as pd
from src import logger


def clean_columns(columns):
    """Cleans column names by replacing spaces, commas, semicolons, braces, parentheses, newlines, tabs, and equal signs with underscores."""
    return [re.sub(r"[ ,;{}()\n\t=]+", "_", c.strip()).strip("_") for c in columns]


def row_to_payload(row: pd.Series) -> dict:
    """Converts a row from the CSV into a JSON payload, ignoring NaN values."""
    payload = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, (int, float)):
            payload[key] = value
        else:
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                continue
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Chemin du CSV de flux réseau bruts")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--min-delay", type=float, default=0.3, help="Délai min entre deux flux (s)")
    parser.add_argument("--max-delay", type=float, default=2.0, help="Délai max entre deux flux (s)")
    parser.add_argument("--loop", action="store_true", help="Rejoue le fichier en boucle")
    parser.add_argument("--shuffle", action="store_true", help="Mélange les lignes avant rejeu")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = clean_columns(df.columns)

    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    logger.info(f"{len(df)} lines loaded from {args.csv}. Sending to {args.url}")

    sent, errors = 0, 0
    try:
        while True:
            for _, row in df.iterrows():
                payload = row_to_payload(row)
                try:
                    resp = requests.post(args.url, json=payload, timeout=5)
                    resp.raise_for_status()
                    result = resp.json()
                    sent += 1
                    tag = "red" if result.get("Label") != "BENIGN" else "green"
                    logger.info(
                        f"{tag} [{sent}] {result.get('Label')} "
                        f"(confiance={result.get('confidence', 0):.3f})"
                    )
                except requests.RequestException as e:
                    errors += 1
                    logger.error(f"Failed to send flow #{sent + errors} : {e}")

                time.sleep(random.uniform(args.min_delay, args.max_delay))

            if not args.loop:
                break
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")

    logger.info(f"Completed. Sent={sent}, Errors={errors}")
    sys.exit(0)


if __name__ == "__main__":
    main()