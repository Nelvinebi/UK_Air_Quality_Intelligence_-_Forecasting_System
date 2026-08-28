import json

import joblib
import pandas as pd
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
