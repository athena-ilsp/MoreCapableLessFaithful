#!/usr/bin/env python3
"""Convert a parquet file to a pretty-printed JSON array.

Usage:
  python3 parquet_to_json.py input.parquet output.json
"""
import argparse
import json
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file", help="Path to the input .parquet file")
    parser.add_argument("json_file", help="Path to write the output .json file")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_file)
    with open(args.json_file, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

    print(f"Converted {args.parquet_file} -> {args.json_file}")


if __name__ == "__main__":
    main()
