# src/data_ingestion.py

from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def load_adult_dataset():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    return X, y


def save_raw_data(output_path="data/raw"):
    X, y = load_adult_dataset()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(output_dir / "X_raw.csv", index=False)
    y.to_csv(output_dir / "y_raw.csv", index=False)

    print("Datos guardados en data/raw/")


if __name__ == "__main__":
    save_raw_data()
    