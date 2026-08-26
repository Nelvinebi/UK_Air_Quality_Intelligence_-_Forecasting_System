import logging
from types import SimpleNamespace

import pytest

from src import pipeline


def patch_all_pipeline_stages(monkeypatch, calls):
    """Patch all heavy and validation stages with lightweight call trackers."""
    monkeypatch.setattr(
        pipeline.data_processing,
        "run_pipeline",
        lambda: calls.append("data_processing"),
    )

    monkeypatch.setattr(
        pipeline,
        "validate_processed_data",
        lambda: calls.append("validate_processed_data"),
    )

    monkeypatch.setattr(
        pipeline.feature_engineering,
        "run_pipeline",
        lambda: calls.append("feature_engineering"),
    )

    monkeypatch.setattr(
        pipeline,
        "validate_engineered_data",
        lambda: calls.append("validate_engineered_data"),
    )

    monkeypatch.setattr(
        pipeline.train,
        "run_training",
        lambda: calls.append("training"),
    )

    monkeypatch.setattr(
        pipeline,
        "validate_saved_artifacts",
        lambda: calls.append("validate_saved_artifacts"),
    )

    monkeypatch.setattr(
        pipeline.evaluate,
        "run_evaluation",
        lambda: calls.append("evaluation"),
    )


def test_run_full_pipeline_executes_all_stages_in_order(monkeypatch):
    calls = []

    patch_all_pipeline_stages(
        monkeypatch,
        calls,
    )

    results = pipeline.run_full_pipeline()

    assert calls == [
        "data_processing",
        "validate_processed_data",
        "feature_engineering",
        "validate_engineered_data",
        "training",
        "validate_saved_artifacts",
        "evaluation",
    ]

    assert [
        result.name
        for result in results
    ] == [
        "Data processing",
        "Validate processed data",
        "Feature engineering",
        "Validate engineered data",
        "Model training",
        "Validate model artifacts",
        "Model evaluation",
    ]


def test_skip_data_processing_validates_existing_processed_data(
    monkeypatch,
):
    calls = []

    patch_all_pipeline_stages(
        monkeypatch,
        calls,
    )

    pipeline.run_full_pipeline(
        skip_data_processing=True,
    )

    assert calls == [
        "validate_processed_data",
        "feature_engineering",
        "validate_engineered_data",
        "training",
        "validate_saved_artifacts",
        "evaluation",
    ]


def test_skip_feature_engineering_validates_existing_engineered_data(
    monkeypatch,
):
    calls = []

    patch_all_pipeline_stages(
        monkeypatch,
        calls,
    )

    pipeline.run_full_pipeline(
        skip_data_processing=True,
        skip_feature_engineering=True,
    )

    assert calls == [
        "validate_engineered_data",
        "training",
        "validate_saved_artifacts",
        "evaluation",
    ]


def test_skip_training_validates_existing_model_artifacts(
    monkeypatch,
):
    calls = []

    patch_all_pipeline_stages(
        monkeypatch,
        calls,
    )

    pipeline.run_full_pipeline(
        skip_data_processing=True,
        skip_feature_engineering=True,
        skip_training=True,
    )

    assert calls == [
        "validate_engineered_data",
        "validate_saved_artifacts",
        "evaluation",
    ]


def test_run_full_pipeline_can_skip_every_stage(monkeypatch):
    calls = []

    patch_all_pipeline_stages(
        monkeypatch,
        calls,
    )

    results = pipeline.run_full_pipeline(
        skip_data_processing=True,
        skip_feature_engineering=True,
        skip_training=True,
        skip_evaluation=True,
    )

    assert calls == []
    assert results == []


def test_run_stage_returns_stage_result(caplog):
    caplog.set_level(
        logging.INFO,
        logger="src.pipeline",
    )

    result = pipeline._run_stage(
        "Example stage",
        lambda: None,
    )

    assert isinstance(
        result,
        pipeline.StageResult,
    )

    assert result.name == "Example stage"
    assert result.duration_seconds >= 0

    assert "stage_started name=Example stage" in caplog.text
    assert "stage_completed name=Example stage" in caplog.text


def test_run_stage_propagates_failure():
    def failing_stage():
        raise RuntimeError(
            "stage failed"
        )

    with pytest.raises(
        RuntimeError,
        match="stage failed",
    ):
        pipeline._run_stage(
            "Failing stage",
            failing_stage,
        )


def test_pipeline_stops_after_validation_failure(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        pipeline.data_processing,
        "run_pipeline",
        lambda: calls.append("data_processing"),
    )

    def fail_validation():
        calls.append(
            "validate_processed_data"
        )
        raise ValueError(
            "processed dataset invalid"
        )

    monkeypatch.setattr(
        pipeline,
        "validate_processed_data",
        fail_validation,
    )

    monkeypatch.setattr(
        pipeline.feature_engineering,
        "run_pipeline",
        lambda: calls.append("feature_engineering"),
    )

    with pytest.raises(
        ValueError,
        match="processed dataset invalid",
    ):
        pipeline.run_full_pipeline()

    assert calls == [
        "data_processing",
        "validate_processed_data",
    ]


def test_main_returns_zero_on_success(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                skip_data_processing=False,
                skip_feature_engineering=False,
                skip_training=False,
                skip_evaluation=False,
            )
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "run_full_pipeline",
        lambda **kwargs: [],
    )

    assert pipeline.main() == 0


def test_main_returns_one_when_pipeline_fails(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        pipeline,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                skip_data_processing=False,
                skip_feature_engineering=False,
                skip_training=False,
                skip_evaluation=False,
            )
        ),
    )

    caplog.set_level(
        logging.ERROR,
        logger="src.pipeline",
    )

    def fail_pipeline(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        pipeline,
        "run_full_pipeline",
        fail_pipeline,
    )

    assert pipeline.main() == 1

    error_records = [
        record
        for record in caplog.records
        if (
            record.name == "src.pipeline"
            and record.levelno == logging.ERROR
        )
    ]

    assert any(
        "pipeline_failed error_type=RuntimeError"
        in record.getMessage()
        for record in error_records
    )
    assert any(
        record.exc_info is not None
        for record in error_records
    )
