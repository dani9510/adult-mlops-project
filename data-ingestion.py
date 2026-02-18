from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def load_adult_dataset():
    """
    Descarga el dataset Adult desde UCI.
    Retorna:
        X (DataFrame): variables predictoras
        y (DataFrame): variable objetivo
    """
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets

    return X, y


def save_raw_data(X: pd.DataFrame, y: pd.DataFrame, output_path="data/raw"):
    """
    Guarda los datos en formato CSV para trazabilidad y reproducibilidad.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(output_dir / "X_raw.csv", index=False)
    y.to_csv(output_dir / "y_raw.csv", index=False)

    print("✅ Datos guardados correctamente en data/raw/")


if __name__ == "__main__":
    X, y = load_adult_dataset()
    save_raw_data(X, y)
