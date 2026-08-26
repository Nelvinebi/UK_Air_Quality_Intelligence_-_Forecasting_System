"""
End-to-end reproducible pipeline for the London Air Quality Intelligence project.

Runs:
1. Raw data processing
2. Processed-data validation
3. Feature engineering
4. Engineered-data validation
5. Model training
6. Saved-artifact validation
7. Model evaluation

Usage:
    python -m src.pipeline

Optional stage skipping:
    python -m src.pipeline --skip-data-processing
    python -m src.pipeline --skip-feature-engineering
    python -m src.pipeline --skip-training
    python -m src.pipeline --skip-evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import data_processing, evaluate, feature_engineering, train
from src.logging_config import configure_logging
from src.validation import (
    validate_model_artifacts,
    validate_model_dataset,
    validate_station_datetime_integrity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MERGED_DATA_PATH = feature_engineering.INPUT_PATH
FEATURE_DATA_PATH = feature_engineering.OUTPUT_PATH

MODEL_PATH = train.MODEL_PATH
IMPUTER_PATH = train.IMPUTER_PATH
METADATA_PATH = train.METADATA_PATH


# ---------------------------------------------------------------------------
# Pipeline result model
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Execution result for one pipeline stage."""

    name: str
    duration_seconds: float


# ---------------------------------------------------------------------------
# Validation checkpoints
# ---------------------------------------------------------------------------

def validate_processed_data() -> None:
    """
    Validate the merged air-quality/weather dataset produced by the
    data-processing stage.

    Checks that:
    - the processed file exists;
    - the dataset is not empty;
    - Station and Datetime are present;
    - Datetime values are valid;
    - Station values are present;
    - duplicate Station-Datetime rows do not exist.
    """
    if not MERGED_DATA_PATH.is_file():
        raise FileNotFoundError(
            "Processed merged dataset not found: "
            f"{MERGED_DATA_PATH}"
        )

    df = pd.read_csv(
        MERGED_DATA_PATH,
        parse_dates=["Datetime"],
    )

    if df.empty:
        raise ValueError(
            "Processed merged dataset is empty."
        )

    validate_station_datetime_integrity(df)


def validate_engineered_data() -> None:
    """
    Validate the model-ready dataset produced by feature engineering.

    Uses the canonical feature and target definitions from
    src.feature_engineering, so this validation does not depend on
    previously generated model metadata.
    """
    if not FEATURE_DATA_PATH.is_file():
        raise FileNotFoundError(
            "Engineered feature dataset not found: "
            f"{FEATURE_DATA_PATH}"
        )

    df = pd.read_csv(
        FEATURE_DATA_PATH,
        parse_dates=["Datetime"],
    )

    validate_model_dataset(
        df,
        features=feature_engineering.FEATURES,
        target=feature_engineering.TARGET,
    )


def validate_saved_artifacts() -> None:
    """
    Validate the persisted model, imputer, and metadata produced by
    the training stage.
    """
    validate_model_artifacts(
        model_path=MODEL_PATH,
        imputer_path=IMPUTER_PATH,
        metadata_path=METADATA_PATH,
    )


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------

def _run_stage(
    name: str,
    function: Callable,
) -> StageResult:
    """
    Execute one pipeline stage with progress and timing information.

    Exceptions intentionally propagate to main(), which records the
    failure with traceback information and returns a non-zero exit
    status.
    """
    logger.info(
        "stage_started name=%s",
        name,
    )

    start = time.perf_counter()

    function()

    duration = time.perf_counter() - start

    logger.info(
        "stage_completed name=%s duration_seconds=%.2f",
        name,
        duration,
    )

    return StageResult(
        name=name,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_full_pipeline(
    *,
    skip_data_processing: bool = False,
    skip_feature_engineering: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
) -> list[StageResult]:
    """
    Run the complete machine-learning pipeline.

    Validation checkpoints are executed after each artifact-producing
    stage.

    When an upstream stage is skipped but a downstream stage still
    depends on its existing artifact, the pre-existing artifact is
    validated before the downstream stage executes.
    """
    results: list[StageResult] = []

    # -----------------------------------------------------------------------
    # 1. Data processing
    # -----------------------------------------------------------------------

    if not skip_data_processing:
        results.append(
            _run_stage(
                "Data processing",
                data_processing.run_pipeline,
            )
        )

        results.append(
            _run_stage(
                "Validate processed data",
                validate_processed_data,
            )
        )

    # If processing is skipped but feature engineering will run,
    # validate the existing merged dataset first.
    elif not skip_feature_engineering:
        results.append(
            _run_stage(
                "Validate existing processed data",
                validate_processed_data,
            )
        )

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------

    if not skip_feature_engineering:
        results.append(
            _run_stage(
                "Feature engineering",
                feature_engineering.run_pipeline,
            )
        )

        results.append(
            _run_stage(
                "Validate engineered data",
                validate_engineered_data,
            )
        )

    # If feature engineering is skipped but training or evaluation
    # will use the existing engineered dataset, validate it first.
    elif not skip_training or not skip_evaluation:
        results.append(
            _run_stage(
                "Validate existing engineered data",
                validate_engineered_data,
            )
        )

    # -----------------------------------------------------------------------
    # 3. Model training
    # -----------------------------------------------------------------------

    if not skip_training:
        results.append(
            _run_stage(
                "Model training",
                train.run_training,
            )
        )

        results.append(
            _run_stage(
                "Validate model artifacts",
                validate_saved_artifacts,
            )
        )

    # If training is skipped but evaluation will use the existing
    # model bundle, validate the artifacts before evaluation.
    elif not skip_evaluation:
        results.append(
            _run_stage(
                "Validate existing model artifacts",
                validate_saved_artifacts,
            )
        )

    # -----------------------------------------------------------------------
    # 4. Model evaluation
    # -----------------------------------------------------------------------

    if not skip_evaluation:
        results.append(
            _run_stage(
                "Model evaluation",
                evaluate.run_evaluation,
            )
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    total_duration = sum(
        result.duration_seconds
        for result in results
    )

    if results:
        logger.info(
            "pipeline_completed stages=%d total_duration_seconds=%.2f",
            len(results),
            total_duration,
        )

        for result in results:
            logger.info(
                "stage_summary name=%s duration_seconds=%.2f",
                result.name,
                result.duration_seconds,
            )
    else:
        logger.warning(
            "pipeline_no_stages_executed"
        )

    return results


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the London Air Quality Intelligence "
            "end-to-end machine-learning pipeline."
        )
    )

    parser.add_argument(
        "--skip-data-processing",
        action="store_true",
        help=(
            "Skip raw-data processing and use the existing "
            "processed merged dataset."
        ),
    )

    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help=(
            "Skip feature engineering and use the existing "
            "engineered feature dataset."
        ),
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip model training and use the existing "
            "persisted model artifacts."
        ),
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip final model evaluation.",
    )

    return parser


def main() -> int:
    """Command-line entrypoint."""
    configure_logging()

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
        logger.exception(
            "pipeline_failed error_type=%s",
            type(exc).__name__,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
