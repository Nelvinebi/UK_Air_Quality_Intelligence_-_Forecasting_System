import json

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from src import train


def test_save_model_records_runtime_environment(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "models"
    model_path = model_dir / "model.pkl"
    imputer_path = model_dir / "imputer.pkl"
    metadata_path = model_dir / "metadata.json"

    monkeypatch.setattr(
        train,
        "MODEL_DIR",
        model_dir,
    )

    monkeypatch.setattr(
        train,
        "IMPUTER_PATH",
        imputer_path,
    )

    monkeypatch.setattr(
        train,
        "METADATA_PATH",
        metadata_path,
    )

    X = pd.DataFrame(
        {
            "PM2.5": [10.0, 20.0, 30.0],
            "Temperature": [5.0, 6.0, 7.0],
        }
    )

    y = pd.Series([11.0, 21.0, 31.0])

    imputer = SimpleImputer(
        strategy="median",
    )

    X_imputed = imputer.fit_transform(X)

    X_imputed = pd.DataFrame(
        X_imputed,
        columns=X.columns,
    )

    model = RandomForestRegressor(
        n_estimators=2,
        random_state=42,
    )

    model.fit(
        X_imputed,
        y,
    )

    train.save_model(
        model=model,
        imputer=imputer,
        features=list(X.columns),
        target="PM2.5_target",
        path=model_path,
    )

    assert model_path.exists()
    assert imputer_path.exists()
    assert metadata_path.exists()

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["features"] == [
        "PM2.5",
        "Temperature",
    ]

    assert metadata["target"] == "PM2.5_target"

    assert "environment" in metadata
    assert "python" in metadata["environment"]
    assert "scikit_learn" in metadata["environment"]
    assert "joblib" in metadata["environment"]

    assert metadata["environment"]["scikit_learn"] == train.sklearn.__version__

    assert metadata["environment"]["joblib"] == joblib.__version__


def test_load_engineered_dataset_parses_datetime(tmp_path):
    data_path = tmp_path / "engineered.csv"

    pd.DataFrame(
        {
            "Datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "PM2.5": [10.0, 12.0],
        }
    ).to_csv(data_path, index=False)

    result = train.load_engineered_dataset(data_path)

    assert pd.api.types.is_datetime64_any_dtype(result["Datetime"])
    assert result["PM2.5"].tolist() == [10.0, 12.0]


def test_impute_features_uses_training_medians_and_preserves_shape(monkeypatch):
    features = ["feature_a", "feature_b"]
    monkeypatch.setattr(train, "FEATURES", features)

    X_train = pd.DataFrame(
        {
            "feature_a": [1.0, None, 3.0],
            "feature_b": [10.0, 20.0, None],
        },
        index=[10, 11, 12],
    )
    X_test = pd.DataFrame(
        {
            "feature_a": [None],
            "feature_b": [30.0],
        },
        index=[20],
    )

    imputer, X_train_imputed, X_test_imputed = train.impute_features(
        X_train,
        X_test,
    )

    assert imputer.strategy == "median"
    assert X_train_imputed.columns.tolist() == features
    assert X_test_imputed.columns.tolist() == features
    assert X_train_imputed.index.tolist() == [10, 11, 12]
    assert X_test_imputed.index.tolist() == [20]
    assert X_train_imputed.loc[11, "feature_a"] == pytest.approx(2.0)
    assert X_train_imputed.loc[12, "feature_b"] == pytest.approx(15.0)
    assert X_test_imputed.loc[20, "feature_a"] == pytest.approx(2.0)


def test_train_random_forest_accepts_parameter_overrides():
    X_train = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    y_train = pd.Series([10.0, 20.0, 30.0, 40.0])

    model = train.train_random_forest(
        X_train,
        y_train,
        n_estimators=2,
        max_depth=2,
        n_jobs=1,
        random_state=7,
    )

    assert model.n_estimators == 2
    assert model.max_depth == 2
    assert model.n_jobs == 1
    assert model.random_state == 7
    assert len(model.estimators_) == 2


def test_evaluate_predictions_returns_expected_regression_metrics():
    metrics = train.evaluate_predictions(
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 2.0],
    )

    assert metrics["MAE"] == pytest.approx(2.0 / 3.0)
    assert metrics["RMSE"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert metrics["R2"] == pytest.approx(0.0)


class _DummyTrainingModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.predict_input = None

    def predict(self, X):
        self.predict_input = X
        return self.predictions


def _stub_training_dependencies(monkeypatch):
    source_df = pd.DataFrame({"row": [0, 1, 2]})
    train_df = source_df.iloc[:2].copy()
    test_df = source_df.iloc[2:].copy()

    X_train = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "feature_a": [5.0],
            "feature_b": [6.0],
        }
    )
    y_train = pd.Series([10.0, 20.0])
    y_test = pd.Series([30.0])

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    imputer = object()
    model = _DummyTrainingModel([29.5])
    metrics = {
        "MAE": 0.5,
        "RMSE": 0.5,
        "R2": 0.9,
    }

    monkeypatch.setattr(
        train,
        "load_engineered_dataset",
        lambda _path: source_df,
    )
    monkeypatch.setattr(
        train,
        "train_test_split_by_year",
        lambda _df: (train_df, test_df),
    )
    monkeypatch.setattr(
        train,
        "get_model_matrices",
        lambda _train_df, _test_df: (X_train, X_test, y_train, y_test),
    )
    monkeypatch.setattr(
        train,
        "impute_features",
        lambda _X_train, _X_test: (
            imputer,
            X_train_imputed,
            X_test_imputed,
        ),
    )
    monkeypatch.setattr(
        train,
        "train_random_forest",
        lambda _X_train, _y_train: model,
    )
    monkeypatch.setattr(
        train,
        "evaluate_predictions",
        lambda _y_true, _y_pred: metrics,
    )

    return {
        "model": model,
        "imputer": imputer,
        "metrics": metrics,
        "X_test_imputed": X_test_imputed,
    }


def test_run_training_without_saving_artifacts(tmp_path, monkeypatch):
    context = _stub_training_dependencies(monkeypatch)
    results_path = tmp_path / "outputs" / "results.csv"
    save_calls = []

    monkeypatch.setattr(train, "RESULTS_PATH", results_path)
    monkeypatch.setattr(
        train,
        "save_model",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    metrics = train.run_training(
        data_path=tmp_path / "unused.csv",
        save=False,
    )

    assert metrics == context["metrics"]
    assert save_calls == []
    assert not results_path.exists()
    assert context["model"].predict_input.equals(context["X_test_imputed"])


def test_run_training_saves_new_results_file(tmp_path, monkeypatch):
    context = _stub_training_dependencies(monkeypatch)
    results_path = tmp_path / "outputs" / "results.csv"
    save_calls = []

    monkeypatch.setattr(train, "RESULTS_PATH", results_path)

    def fake_save_model(model, imputer, features, target):
        save_calls.append(
            {
                "model": model,
                "imputer": imputer,
                "features": features,
                "target": target,
            }
        )

    monkeypatch.setattr(train, "save_model", fake_save_model)

    metrics = train.run_training(
        data_path=tmp_path / "unused.csv",
        save=True,
    )

    assert metrics == context["metrics"]
    assert len(save_calls) == 1
    assert save_calls[0]["model"] is context["model"]
    assert save_calls[0]["imputer"] is context["imputer"]
    assert save_calls[0]["features"] == train.FEATURES
    assert save_calls[0]["target"] == train.TARGET

    saved_results = pd.read_csv(results_path)

    assert saved_results["Model"].tolist() == ["Random Forest (final)"]
    assert saved_results.loc[0, "MAE"] == pytest.approx(0.5)
    assert saved_results.loc[0, "RMSE"] == pytest.approx(0.5)
    assert saved_results.loc[0, "R2"] == pytest.approx(0.9)


def test_run_training_replaces_existing_final_result(tmp_path, monkeypatch):
    context = _stub_training_dependencies(monkeypatch)
    results_path = tmp_path / "outputs" / "results.csv"
    results_path.parent.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "Model": "Persistence baseline",
                "MAE": 2.0,
                "RMSE": 3.0,
                "R2": 0.5,
            },
            {
                "Model": "Random Forest (final)",
                "MAE": 99.0,
                "RMSE": 99.0,
                "R2": -99.0,
            },
        ]
    ).to_csv(results_path, index=False)

    monkeypatch.setattr(train, "RESULTS_PATH", results_path)
    monkeypatch.setattr(train, "save_model", lambda *args, **kwargs: None)

    metrics = train.run_training(
        data_path=tmp_path / "unused.csv",
        save=True,
    )

    assert metrics == context["metrics"]

    saved_results = pd.read_csv(results_path)

    assert saved_results["Model"].tolist() == [
        "Persistence baseline",
        "Random Forest (final)",
    ]

    final_row = saved_results.loc[saved_results["Model"] == "Random Forest (final)"].iloc[0]

    assert final_row["MAE"] == pytest.approx(0.5)
    assert final_row["RMSE"] == pytest.approx(0.5)
    assert final_row["R2"] == pytest.approx(0.9)
