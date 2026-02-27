from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def load_adult_dataset():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    return X, y


def save_raw_data(X: pd.DataFrame, y: pd.DataFrame, output_path: str = "data/raw") -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Requerido: Parquet
    X.to_parquet(output_dir / "features.parquet", index=False)
    y.to_parquet(output_dir / "targets.parquet", index=False)

    # (Opcional) si quieren mantener CSV por debug
    # X.to_csv(output_dir / "X_raw.csv", index=False)
    # y.to_csv(output_dir / "y_raw.csv", index=False)

    print("✅ Datos guardados correctamente en data/raw/ (parquet)")


if __name__ == "__main__":
    X, y = load_adult_dataset()
    save_raw_data(X, y)