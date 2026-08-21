from types import SimpleNamespace

import pytest

from src import pipeline


def test_run_full_pipeline_executes_all_stages_in_order(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline.data_processing,
        "run_pipeline",
        lambda: calls.append("data_processing"),
    )

    monkeypatch.setattr(
        pipeline.feature_engineering,
        "run_pipeline",
        lambda: calls.append("feature_engineering"),
    )

    monkeypatch.setattr(
        pipeline.train,
        "run_training",
        lambda: calls.append("training"),
    )

    monkeypatch.setattr(
        pipeline.evaluate,
        "run_evaluation",
        lambda: calls.append("evaluation"),
    )

    results = pipeline.run_full_pipeline()

    assert calls == [
        "data_processing",
        "feature_engineering",
        "training",
        "evaluation",
    ]

    assert [result.name for result in results] == [
        "Data processing",
        "Feature engineering",
        "Model training",
        "Model evaluation",
    ]


def test_run_full_pipeline_respects_skip_flags(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline.data_processing,
        "run_pipeline",
        lambda: calls.append("data_processing"),
    )

    monkeypatch.setattr(
        pipeline.feature_engineering,
        "run_pipeline",
        lambda: calls.append("feature_engineering"),
    )

    monkeypatch.setattr(
        pipeline.train,
        "run_training",
        lambda: calls.append("training"),
    )

    monkeypatch.setattr(
        pipeline.evaluate,
        "run_evaluation",
        lambda: calls.append("evaluation"),
    )

    pipeline.run_full_pipeline(
        skip_data_processing=True,
        skip_training=True,
    )

    assert calls == [
        "feature_engineering",
        "evaluation",
    ]


def test_run_full_pipeline_can_skip_every_stage(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline.data_processing,
        "run_pipeline",
        lambda: calls.append("data_processing"),
    )

    monkeypatch.setattr(
        pipeline.feature_engineering,
        "run_pipeline",
        lambda: calls.append("feature_engineering"),
    )

    monkeypatch.setattr(
        pipeline.train,
        "run_training",
        lambda: calls.append("training"),
    )

    monkeypatch.setattr(
        pipeline.evaluate,
        "run_evaluation",
        lambda: calls.append("evaluation"),
    )

    results = pipeline.run_full_pipeline(
        skip_data_processing=True,
        skip_feature_engineering=True,
        skip_training=True,
        skip_evaluation=True,
    )

    assert calls == []
    assert results == []


def test_run_stage_returns_stage_result():
    result = pipeline._run_stage(
        "Example stage",
        lambda: None,
    )

    assert isinstance(result, pipeline.StageResult)
    assert result.name == "Example stage"
    assert result.duration_seconds >= 0


def test_run_stage_propagates_failure():
    def failing_stage():
        raise RuntimeError("stage failed")

    with pytest.raises(
        RuntimeError,
        match="stage failed",
    ):
        pipeline._run_stage(
            "Failing stage",
            failing_stage,
        )


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


def test_main_returns_one_when_pipeline_fails(monkeypatch):
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

    def fail_pipeline(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        pipeline,
        "run_full_pipeline",
        fail_pipeline,
    )

    assert pipeline.main() == 1
