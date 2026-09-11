from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "Datasets" / "UNSW-NB15" / "UNSW_NB15_training-set.csv"


def train(dataset_path: Path, sample_size: int, model_name: str) -> None:
    data = pd.read_csv(dataset_path)
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)

    label_column = next(
        (column for column in data.columns if column.strip().lower() in {"label", "class", "target"}),
        None,
    )
    if label_column is None:
        raise ValueError("Dataset must contain a label, class, or target column.")

    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=[label_column])
    features = data.drop(columns=[label_column])
    features = features.drop(columns=["attack_cat"], errors="ignore")
    target = data[label_column].astype(str)

    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    model_path = ROOT / "models" / f"{model_name}.joblib"
    metrics_path = ROOT / "models" / f"{model_name}_metrics.json"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    report_text = classification_report(y_test, predictions, zero_division=0)
    print(report_text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the network anomaly detector.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--model-name", default="unsw_nb15_model")
    args = parser.parse_args()
    train(args.dataset, args.sample_size, args.model_name)
