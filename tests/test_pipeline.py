"""Pipeline / policy / leakage tests for Return-Risk Scorer."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bounds import compute_manual_field_bounds, is_out_of_bounds
from src.config import (
    DATASET_PATH,
    LABEL_ID_COL,
    LEAKAGE_COLS,
    MODELS_DIR,
    REFUND_COST_CAP_USD,
    REFUND_COST_FLOOR_USD,
    TARGET_COL,
)
from src.data import load_split
from src.detector.model import ReturnRiskDetector
from src.features import feature_columns
from src.pipeline import score_return
from src.verifier.policy import expected_action_cost, refund_exposure_usd


def _dataset_available() -> bool:
    return DATASET_PATH.exists()


def _model_available() -> bool:
    return (MODELS_DIR / "detector.joblib").exists()


pytestmark = pytest.mark.skipif(
    not _dataset_available(),
    reason=(
        f"Raw dataset missing at {DATASET_PATH}. "
        "Place ecommerce_return_abuse_dataset.csv under data/raw/ to run tests."
    ),
)


@pytest.fixture(scope="session")
def split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_split()


@pytest.fixture(scope="session")
def detector(split_frames) -> ReturnRiskDetector:
    model_path = MODELS_DIR / "detector.joblib"
    if model_path.exists():
        return ReturnRiskDetector.load(model_path)
    train_df, _ = split_frames
    det = ReturnRiskDetector().fit(train_df)
    det.save(model_path)
    return det


def test_no_label_leakage(detector: ReturnRiskDetector) -> None:
    """Leakage columns must not appear in prepared features or fitted names."""
    prepared = set(feature_columns())
    leakage = set(LEAKAGE_COLS) | {TARGET_COL, LABEL_ID_COL}
    leaked_prepared = prepared & leakage
    assert not leaked_prepared, f"Leakage cols in feature_columns(): {leaked_prepared}"

    fitted = list(detector.feature_names_ or [])
    assert fitted, "Fitted detector has empty feature_names_"
    leaked_fitted = [
        name
        for name in fitted
        if name in leakage
        or any(name == col or name.startswith(f"{col}_") for col in leakage)
    ]
    assert not leaked_fitted, f"Leakage cols in fitted feature names: {leaked_fitted}"

    # Explicit target/label must never be model inputs
    assert TARGET_COL not in prepared and TARGET_COL not in fitted
    assert LABEL_ID_COL not in prepared and LABEL_ID_COL not in fitted


def test_no_train_test_overlap(split_frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = split_frames
    drop_cols = [c for c in (TARGET_COL, LABEL_ID_COL) if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    train_keys = train_df[feature_cols].astype(str).agg("|".join, axis=1)
    test_keys = test_df[feature_cols].astype(str).agg("|".join, axis=1)
    overlap = set(train_keys) & set(test_keys)
    assert len(overlap) == 0, f"Found {len(overlap)} duplicate rows across train/test"


def test_cost_scales_with_refund_amount() -> None:
    risk = 0.8
    low = expected_action_cost(risk, "auto_approve", 50.0)
    high = expected_action_cost(risk, "auto_approve", 500.0)
    assert high > low
    assert high / low == pytest.approx(500.0 / 50.0, rel=1e-6)

    hold_low = expected_action_cost(risk, "hold_for_evidence", 50.0)
    hold_high = expected_action_cost(risk, "hold_for_evidence", 500.0)
    assert hold_high > hold_low

    # Floor: tiny refunds still priced at REFUND_COST_FLOOR_USD
    below_floor = refund_exposure_usd(1.0)
    assert below_floor == REFUND_COST_FLOOR_USD
    cost_below = expected_action_cost(1.0, "auto_approve", 1.0)
    cost_at_floor = expected_action_cost(1.0, "auto_approve", REFUND_COST_FLOOR_USD)
    assert cost_below == cost_at_floor == REFUND_COST_FLOOR_USD

    # Cap: huge refunds stop growing past REFUND_COST_CAP_USD
    above_cap = refund_exposure_usd(REFUND_COST_CAP_USD * 10)
    assert above_cap == REFUND_COST_CAP_USD
    cost_above = expected_action_cost(1.0, "auto_approve", REFUND_COST_CAP_USD * 10)
    cost_at_cap = expected_action_cost(1.0, "auto_approve", REFUND_COST_CAP_USD)
    assert cost_above == cost_at_cap == REFUND_COST_CAP_USD


def test_insufficient_signal_triggers(
    split_frames: tuple[pd.DataFrame, pd.DataFrame],
    detector: ReturnRiskDetector,
) -> None:
    _, test_df = split_frames
    row = test_df.iloc[0].to_dict()
    row["order_id"] = "TEST-INSUFFICIENT-001"
    row["product_category"] = ""
    row["total_returns_lifetime"] = int(row.get("total_orders_lifetime", 0) or 0) + 5

    result = score_return(row, detector, audit=None, with_shap=False, use_llm=False)
    assert result["action"] == "insufficient_signal"
    assert result["risk_score"] is None
    assert result.get("insufficient_signal") is True
    assert result.get("issues")


def test_manual_form_bounds_reject_out_of_range(
    split_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _ = split_frames
    bounds = compute_manual_field_bounds(train_df)
    refund_max = float(bounds["refund_amount_requested_usd"]["max"])
    assert refund_max > 0

    # Same check the UI relies on: value above 99th-percentile max is OOD
    assert is_out_of_bounds(
        "refund_amount_requested_usd",
        refund_max + 1.0,
        bounds,
    )
    assert is_out_of_bounds(
        "refund_amount_requested_usd",
        999_999.0,
        bounds,
    )
    assert not is_out_of_bounds(
        "refund_amount_requested_usd",
        min(refund_max, float(train_df["refund_amount_requested_usd"].median())),
        bounds,
    )
