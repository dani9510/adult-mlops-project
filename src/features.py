from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


RAW_X = Path("data/raw/features.parquet")
RAW_Y = Path("data/raw/targets.parquet")

OUT_X = Path("data/processed/features.parquet")
OUT_Y = Path("data/processed/targets.parquet")

SCALER_PATH = Path("artifacts/scaler.joblib")
ENCODER_PATH = Path("artifacts/encoder.joblib")

TARGET_COL = "income"

NUM_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CAT_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def normalize_income(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.$", "", regex=True)
    return s


def build_preprocessor() -> ColumnTransformer:
    num_pipe = StandardScaler()
    cat_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


if __name__ == "__main__":
    if not RAW_X.exists():
        raise FileNotFoundError(RAW_X)
    if not RAW_Y.exists():
        raise FileNotFoundError(RAW_Y)

    X = pd.read_parquet(RAW_X)
    y = pd.read_parquet(RAW_Y)

    # Limpieza mínima permitida: '?' -> 'Unknown' en categóricas
    for c in CAT_COLS:
        X[c] = X[c].astype(str).str.strip()
        X[c] = X[c].replace("?", "Unknown")

    # Normaliza target (quita punto final)
    y[TARGET_COL] = normalize_income(y[TARGET_COL])

    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(Xt.shape[1])]

    OUT_X.parent.mkdir(parents=True, exist_ok=True)
    OUT_Y.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(Xt, columns=feature_names).to_parquet(OUT_X, index=False)
    y.to_parquet(OUT_Y, index=False)

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor.named_transformers_["num"], SCALER_PATH)
    joblib.dump(preprocessor.named_transformers_["cat"], ENCODER_PATH)

    print(f"✅ Saved processed: {OUT_X}, {OUT_Y}")
    print(f"✅ Saved transformers: {SCALER_PATH}, {ENCODER_PATH}")