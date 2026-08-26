from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import evaluate


class DummyImputer:
    def transform(self, data):
        return data.to_numpy(copy=True)


class DummyModel:
    def __init__(self):
        self.feature_importances_ = np.linspace(
            0.0,
            1.0,
            len(evaluate.FEATURES),
        )

    def predict(self, data):
        assert list(data.columns) == evaluate.FEATURES
        assert len(data) == 2
        return np.array([9.0, 18.0])


def _prediction_frame():
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ]
            ),
            "Station": ["Station A", "Station A", "Station B"],
            "Actual": [10.0, 12.0, 20.0],
            "Predicted": [9.0, 14.0, 18.0],
            "Residual": [1.0, -2.0, 2.0],
            "Absolute_Error": [1.0, 2.0, 2.0],
        }
    )


def _patch_evaluation_inputs(monkeypatch):
    model = DummyModel()
    imputer = DummyImputer()

    source_df = pd.DataFrame({"placeholder": [1]})
    train_df = pd.DataFrame({"split": ["train"]})

    test_df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                ]
            ),
            "Station": ["Station A", "Station B"],
        }
    )

    x_test = pd.DataFrame(
        np.ones((2, len(evaluate.FEATURES))),
        columns=evaluate.FEATURES,
    )
    y_test = pd.Series([10.0, 20.0])

    monkeypatch.setattr(
        evaluate,
        "load_model_and_imputer",
        lambda: (model, imputer),
    )
    monkeypatch.setattr(
        evaluate,
        "load_engineered_dataset",
        lambda _path: source_df,
    )
    monkeypatch.setattr(
        evaluate,
        "train_test_split_by_year",
        lambda _df: (train_df, test_df),
    )
    monkeypatch.setattr(
        evaluate,
        "get_model_matrices",
        lambda _train, _test: (None, x_test, None, y_test),
    )

    return test_df


def test_load_model_and_imputer_round_trip(tmp_path):
    model_path = tmp_path / "model.pkl"
    imputer_path = tmp_path / "imputer.pkl"

    expected_model = {"kind": "test-model"}
    expected_imputer = {"kind": "test-imputer"}

    joblib.dump(expected_model, model_path)
    joblib.dump(expected_imputer, imputer_path)

    model, imputer = evaluate.load_model_and_imputer(
        model_path=model_path,
        imputer_path=imputer_path,
    )

    assert model == expected_model
    assert imputer == expected_imputer


def test_compute_feature_importance_sorts_descending():
    class FeatureModel:
        feature_importances_ = np.array([0.1, 0.7, 0.2])

    result = evaluate.compute_feature_importance(
        FeatureModel(),
        features=["feature_a", "feature_b", "feature_c"],
    )

    assert result["Feature"].tolist() == [
        "feature_b",
        "feature_c",
        "feature_a",
    ]
    assert result["Importance"].tolist() == pytest.approx(
        [0.7, 0.2, 0.1]
    )


def test_compute_feature_importance_uses_default_features():
    model = DummyModel()

    result = evaluate.compute_feature_importance(model)

    assert len(result) == len(evaluate.FEATURES)
    assert set(result["Feature"]) == set(evaluate.FEATURES)
    assert result["Importance"].is_monotonic_decreasing


def test_build_prediction_df_calculates_residuals_and_errors():
    test_df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                ]
            ),
            "Station": ["Station A", "Station B"],
        }
    )
    y_test = pd.Series([10.0, 20.0])
    predictions = np.array([8.0, 23.0])

    result = evaluate.build_prediction_df(
        test_df,
        y_test,
        predictions,
    )

    assert result["Actual"].tolist() == [10.0, 20.0]
    assert result["Predicted"].tolist() == [8.0, 23.0]
    assert result["Residual"].tolist() == [2.0, -3.0]
    assert result["Absolute_Error"].tolist() == [2.0, 3.0]


def test_error_summary_returns_expected_metrics():
    prediction_df = _prediction_frame()

    result = evaluate.error_summary(prediction_df)

    assert result["Mean_Absolute_Error"] == pytest.approx(5.0 / 3.0)
    assert result["Mean_Residual"] == pytest.approx(1.0 / 3.0)
    assert result["Median_Absolute_Error"] == pytest.approx(2.0)
    assert result["Max_Absolute_Error"] == pytest.approx(2.0)


def test_station_level_errors_calculates_and_sorts_metrics():
    prediction_df = _prediction_frame()

    result = evaluate.station_level_errors(prediction_df)

    assert result.index.tolist() == ["Station B", "Station A"]

    assert result.loc["Station A", "MAE"] == pytest.approx(1.5)
    assert result.loc["Station A", "RMSE"] == pytest.approx(
        np.sqrt(2.5)
    )
    assert result.loc["Station A", "Mean_Residual"] == pytest.approx(-0.5)
    assert result.loc["Station A", "Observations"] == 2

    assert result.loc["Station B", "MAE"] == pytest.approx(2.0)
    assert result.loc["Station B", "RMSE"] == pytest.approx(2.0)
    assert result.loc["Station B", "Mean_Residual"] == pytest.approx(2.0)
    assert result.loc["Station B", "Observations"] == 1


def test_plot_functions_create_nonempty_files(tmp_path):
    feature_importance = pd.DataFrame(
        {
            "Feature": ["feature_a", "feature_b", "feature_c"],
            "Importance": [0.2, 0.6, 0.2],
        }
    )
    prediction_df = _prediction_frame()
    station_errors = evaluate.station_level_errors(prediction_df)

    feature_path = tmp_path / "feature_importance.png"
    prediction_path = tmp_path / "actual_vs_predicted.png"
    residual_path = tmp_path / "residual_distribution.png"
    station_path = tmp_path / "station_errors.png"

    evaluate.plot_feature_importance(
        feature_importance,
        feature_path,
        top_n=2,
    )
    evaluate.plot_actual_vs_predicted(
        prediction_df,
        prediction_path,
        n_points=2,
    )
    evaluate.plot_residual_distribution(
        prediction_df,
        residual_path,
    )
    evaluate.plot_station_errors(
        station_errors,
        station_path,
    )

    for path in [
        feature_path,
        prediction_path,
        residual_path,
        station_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_run_evaluation_without_saving(monkeypatch, capsys):
    _patch_evaluation_inputs(monkeypatch)

    result = evaluate.run_evaluation(
        data_path=Path("unused.csv"),
        save=False,
    )

    captured = capsys.readouterr()

    assert "Generating predictions..." in captured.out
    assert "Computing feature importance..." in captured.out
    assert "Error analysis:" in captured.out
    assert "Station-level evaluation:" in captured.out

    assert set(result) == {
        "errors",
        "feature_importance",
        "station_errors",
    }

    assert result["errors"]["Mean_Absolute_Error"] == pytest.approx(1.5)
    assert len(result["feature_importance"]) == len(evaluate.FEATURES)
    assert len(result["station_errors"]) == 2


def test_run_evaluation_saves_reports_and_figures(
    tmp_path,
    monkeypatch,
):
    _patch_evaluation_inputs(monkeypatch)

    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"

    predictions_path = reports_dir / "final_predictions.csv"
    station_errors_path = reports_dir / "station_level_errors.csv"
    feature_importance_path = reports_dir / "feature_importance.csv"

    monkeypatch.setattr(evaluate, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(
        evaluate,
        "PREDICTIONS_PATH",
        predictions_path,
    )
    monkeypatch.setattr(
        evaluate,
        "STATION_ERRORS_PATH",
        station_errors_path,
    )
    monkeypatch.setattr(
        evaluate,
        "FEATURE_IMPORTANCE_PATH",
        feature_importance_path,
    )

    result = evaluate.run_evaluation(
        data_path=Path("unused.csv"),
        save=True,
    )

    expected_figures = [
        figures_dir / "final_feature_importance.png",
        figures_dir / "final_actual_vs_predicted.png",
        figures_dir / "final_residual_distribution.png",
        figures_dir / "final_station_rmse.png",
    ]

    for path in expected_figures:
        assert path.exists()
        assert path.stat().st_size > 0

    for path in [
        predictions_path,
        station_errors_path,
        feature_importance_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0

    saved_predictions = pd.read_csv(predictions_path)
    saved_station_errors = pd.read_csv(station_errors_path)
    saved_feature_importance = pd.read_csv(feature_importance_path)

    assert len(saved_predictions) == 2
    assert len(saved_station_errors) == 2
    assert len(saved_feature_importance) == len(evaluate.FEATURES)

    assert result["errors"]["Mean_Absolute_Error"] == pytest.approx(1.5)