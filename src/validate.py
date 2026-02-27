from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema


RAW_X = Path("data/raw/features.parquet")
RAW_Y = Path("data/raw/targets.parquet")
REPORT = Path("artifacts/validation_report.json")

EXPECTED_FEATURE_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
TARGET_COL = "income"


def build_schema() -> DataFrameSchema:
    """
    Schema mínimo con checks seguros.
    No inventa columnas: usa exactamente las que trae tu parquet.
    """
    return DataFrameSchema(
        {
            "age": Column(int, Check.between(16, 100), nullable=False),
            "fnlwgt": Column(int, Check.gt(0), nullable=False),
            "education-num": Column(int, Check.between(1, 20), nullable=False),
            "capital-gain": Column(int, Check.ge(0), nullable=False),
            "capital-loss": Column(int, Check.ge(0), nullable=False),
            "hours-per-week": Column(int, Check.between(1, 120), nullable=False),
            "workclass": Column(object, nullable=True),
            "education": Column(object, nullable=True),
            "marital-status": Column(object, nullable=True),
            "occupation": Column(object, nullable=True),
            "relationship": Column(object, nullable=True),
            "race": Column(object, nullable=True),
            "sex": Column(object, nullable=True),
            "native-country": Column(object, nullable=True),
        },
        strict=True,   # exige exactamente estas columnas
        coerce=True,   # intenta castear tipos
    )


def normalize_income(series: pd.Series) -> pd.Series:
    """
    Normaliza el target que viene con variantes:
    '<=50K', '>50K', '<=50K.', '>50K.' y a veces espacios.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.$", "", regex=True)  # quita punto final
    return s


def run(null_threshold_pct: float = 20.0) -> dict[str, Any]:
    report: dict[str, Any] = {
        "passed": False,
        "errors": [],
        "warnings": [],
        "stats": {},
        "paths": {"features": str(RAW_X), "targets": str(RAW_Y)},
    }

    if not RAW_X.exists():
        report["errors"].append(f"Missing file: {RAW_X}")
        return report
    if not RAW_Y.exists():
        report["errors"].append(f"Missing file: {RAW_Y}")
        return report

    X = pd.read_parquet(RAW_X)
    y = pd.read_parquet(RAW_Y)

    report["stats"]["X_shape"] = [int(X.shape[0]), int(X.shape[1])]
    report["stats"]["y_shape"] = [int(y.shape[0]), int(y.shape[1])]
    report["stats"]["X_cols"] = X.columns.tolist()
    report["stats"]["y_cols"] = y.columns.tolist()

    # columnas exactas esperadas
    missing = [c for c in EXPECTED_FEATURE_COLS if c not in X.columns]
    extra = [c for c in X.columns if c not in EXPECTED_FEATURE_COLS]
    if missing:
        report["errors"].append(f"Missing feature columns: {missing}")
    if extra:
        report["errors"].append(f"Unexpected feature columns: {extra}")

    if TARGET_COL not in y.columns:
        report["errors"].append(f"Missing target column: '{TARGET_COL}'")
    else:
        y_norm = normalize_income(y[TARGET_COL])
        uniq_raw = sorted(y[TARGET_COL].astype(str).str.strip().unique().tolist())
        uniq_norm = sorted(y_norm.unique().tolist())
        report["stats"]["target_unique_raw"] = uniq_raw
        report["stats"]["target_unique_normalized"] = uniq_norm

        allowed = {"<=50K", ">50K"}
        bad = sorted(set(uniq_norm) - allowed)
        if bad:
            report["errors"].append(f"Unexpected target values after normalization: {bad}")

    # nulos
    nulls_pct = (X.isna().mean() * 100).round(4).to_dict()
    report["stats"]["nulls_pct_features"] = nulls_pct
    high_nulls = {k: v for k, v in nulls_pct.items() if v > null_threshold_pct}
    if high_nulls:
        report["warnings"].append(
            f"Columns above null threshold ({null_threshold_pct}%): {high_nulls}"
        )

    # duplicados
    dup = int(X.duplicated().sum())
    report["stats"]["duplicates_features"] = dup
    if dup > 0:
        report["warnings"].append(f"Found {dup} duplicated rows in features.")

    # valores "?" típicos del Adult (no son NaN, pero sí "missing semántico")
    qmarks = {}
    for col in X.select_dtypes(include=["object"]).columns:
        cnt = int((X[col].astype(str).str.strip() == "?").sum())
        if cnt > 0:
            qmarks[col] = cnt
    if qmarks:
        report["warnings"].append(f"Found '?' markers in categorical columns: {qmarks}")
        report["stats"]["question_mark_counts"] = qmarks

    # pandera schema
    schema = build_schema()
    try:
        schema.validate(X, lazy=True)
    except pa.errors.SchemaErrors as e:
        report["errors"].append("Pandera schema validation failed.")
        report["stats"]["pandera_failure_cases_head"] = (
            e.failure_cases.head(20).to_dict(orient="records")
        )

    report["passed"] = len(report["errors"]) == 0
    return report


def save(report: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    rep = run(null_threshold_pct=20.0)
    save(rep)

    if not rep["passed"]:
        raise SystemExit(f"Validation failed. See: {REPORT} | Errors: {rep['errors']}")
    print(f"✅ Validation report saved at {REPORT}")