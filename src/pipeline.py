"""
End-to-end reproducible pipeline for the London Air Quality Intelligence project.

Runs:
1. Raw data processing
2. Feature engineering
3. Model training
4. Model evaluation

Usage:
    python -m src.pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src import data_processing
from src import evaluate
from src import feature_engineering
from src import train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StageResult:
    """Execution result for one pipeline stage."""

    name: str
    duration_seconds: float


def _run_stage(
    name: str,
    function: Callable,
) -> StageResult:
    """
    Execute one pipeline stage with clear progress reporting.

    Any exception is allowed to propagate so the pipeline exits
    with a non-zero status code instead of silently continuing.
    """
    print()
    print("=" * 72)
    print(f"STARTING: {name}")
    print("=" * 72)

    start = time.perf_counter()

    function()

    duration = time.perf_counter() - start

    print()
    print(f"COMPLETED: {name}")
    print(f"DURATION: {duration:.2f} seconds")

    return StageResult(
        name=name,
        duration_seconds=duration,
    )


def run_full_pipeline(
    *,
    skip_data_processing: bool = False,
    skip_feature_engineering: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
) -> list[StageResult]:
    """
    Run the complete project pipeline.

    Stages can optionally be skipped when their required artifacts
    already exist.
    """
    results: list[StageResult] = []

    if not skip_data_processing:
        results.append(
            _run_stage(
                "Data processing",
                data_processing.run_pipeline,
            )
        )

    if not skip_feature_engineering:
        results.append(
            _run_stage(
                "Feature engineering",
                feature_engineering.run_pipeline,
            )
        )

    if not skip_training:
        results.append(
            _run_stage(
                "Model training",
                train.run_training,
            )
        )

    if not skip_evaluation:
        results.append(
            _run_stage(
                "Model evaluation",
                evaluate.run_evaluation,
            )
        )

    print()
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)

    total_duration = sum(
        result.duration_seconds
        for result in results
    )

    for result in results:
        print(
            f"{result.name:<24}"
            f"{result.duration_seconds:>10.2f} s"
        )

    print("-" * 72)
    print(
        f"{'Total':<24}"
        f"{total_duration:>10.2f} s"
    )

    return results


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the London Air Quality Intelligence "
            "end-to-end ML pipeline."
        )
    )

    parser.add_argument(
        "--skip-data-processing",
        action="store_true",
        help="Skip raw-data processing.",
    )

    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help="Skip feature engineering.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training.",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation.",
    )

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_full_pipeline(
            skip_data_processing=args.skip_data_processing,
            skip_feature_engineering=args.skip_feature_engineering,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
        )

    except Exception as exc:
        print()
        print("=" * 72)
        print("PIPELINE FAILED")
        print("=" * 72)
        print(f"{type(exc).__name__}: {exc}")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
