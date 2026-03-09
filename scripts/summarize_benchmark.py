# ruff: noqa: E402
import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mpc.benchmark import aggregate_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument(
        "--input-glob",
        type=str,
        default=os.path.join(ROOT, "outputs", "benchmarks", "raw", "*_benchmark_*.csv"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "outputs", "benchmarks", "summary", "benchmark_summary.csv"),
    )
    args = parser.parse_args()

    files = glob.glob(args.input_glob)
    if not files:
        raise FileNotFoundError(f"No benchmark files found: {args.input_glob}")

    frames = [pd.read_csv(p) for p in files]
    raw = pd.concat(frames, ignore_index=True)
    summary = aggregate_records(raw.to_dict(orient="records"))
    summary_df = pd.DataFrame(summary)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print(f"Saved summary: {args.output}")
    print(summary_df)


if __name__ == "__main__":
    main()
