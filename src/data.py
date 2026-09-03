"""Data loading, EDA helpers, and stratified train/holdout split."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    DATA_PROCESSED,
    DATASET_PATH,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV under data/raw/."
        )
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")
    # Normalize whitespace; keep dataset class strings for consistency
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    unknown = set(df[TARGET_COL].unique()) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unexpected abuse_type values: {sorted(unknown)}")
    return df


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "class_balance": df[TARGET_COL].value_counts().to_dict(),
        "class_balance_pct": (
            (df[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict()
        ),
        "null_pct": (df.isna().mean() * 100).round(3).to_dict(),
        "product_category_counts": df["product_category"].value_counts().head(20).to_dict()
        if "product_category" in df.columns
        else {},
        "abuse_by_category": (
            df.groupby("product_category")[TARGET_COL]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .round(3)
            .to_dict()
            if "product_category" in df.columns
            else {}
        ),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)
    meta = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": test_size,
        "random_state": random_state,
        "train_balance": train_df[TARGET_COL].value_counts().to_dict(),
        "test_balance": test_df[TARGET_COL].value_counts().to_dict(),
    }
    (DATA_PROCESSED / "split_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return train_df, test_df


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    df = load_raw()
    return stratified_split(df)
