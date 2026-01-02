"""Simple profiling entrypoint for the trade normalization pipeline.

Usage:
    python scripts/profile_pipeline.py --path data/raw_file.csv --chunk-size 75000 --parallel-chunks

The script runs the normalization pipeline and prints per-chunk stage timings
so you can benchmark the cost of each phase on your sample data.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import List, Dict

from japantrade.tradefile import NormalizationConfig, TradeFile


def _summarize_timings(timings: List[Dict]) -> Dict:
    stage_totals = {}
    for chunk in timings:
        for metric in chunk.get("metrics", []):
            step = metric.get("step")
            duration = metric.get("duration", 0)
            stage_totals.setdefault(step, []).append(duration)

    return {
        step: {
            "avg_seconds": round(mean(values), 4),
            "runs": len(values)
        }
        for step, values in stage_totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Profile the trade normalization pipeline.")
    parser.add_argument("--path", required=True, help="Path to the raw trade file (csv or zip).")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size to use while streaming.")
    parser.add_argument("--parallel-chunks", action="store_true", help="Process chunks in parallel where safe.")
    parser.add_argument("--max-workers", type=int, default=None, help="Max workers for parallel chunk processing.")
    parser.add_argument("--use-tqdm", action="store_true", help="Enable tqdm progress bars if installed.")
    args = parser.parse_args()

    config = NormalizationConfig(
        chunk_size=args.chunk_size,
        use_tqdm=args.use_tqdm,
        parallel_chunks=args.parallel_chunks,
        max_workers=args.max_workers,
    )

    print("Running pipeline with config:", config)
    tf = TradeFile(
        source=args.path,
        raw=True,
        kind="infer",
        chunk_size=args.chunk_size,
        normalization_config=config,
    )

    summary = _summarize_timings(tf.latest_timings)
    summary_path = Path("profile_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Per-step average durations (seconds):")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
