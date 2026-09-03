"""Shared paths and constants for Return-Risk Scorer."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
METRICS_DIR = OUTPUTS / "metrics"
AUDIT_DIR = OUTPUTS / "audit"
PLOTS_DIR = OUTPUTS / "plots"

DATASET_PATH = DATA_RAW / "ecommerce_return_abuse_dataset.csv"
TARGET_COL = "abuse_type"
LABEL_ID_COL = "abuse_label"

# Dataset class names (held fixed for metrics)
CLASS_NAMES = [
    "Legitimate",
    "Policy Abuser",
    "Fraudulent Return",
    "Wardrobing",
]

# Columns that must never be used as model features
LEAKAGE_COLS = {
    TARGET_COL,
    LABEL_ID_COL,
    "order_id",
    "customer_id",
    "order_date",
    "return_date",
}

# Fields required for a scorable return; missing/contradictory → insufficient_signal
REQUIRED_FIELDS = [
    "account_age_days",
    "product_category",
    "return_reason",
    "refund_amount_requested_usd",
    "total_orders_lifetime",
    "total_returns_lifetime",
    "days_to_return",
]

# Cost rates are applied to refund_amount_requested_usd (capped), not flat dollars.
# Example: FN at rate 1.0 on a $500 refund → $500 expected loss if abuse is certain.
COST_FN_ABUSE_RATE = 1.00          # missed abuse ≈ full refund loss
COST_FP_HOLD_RATE = 0.05           # support + mild CX friction (~5% of refund)
COST_FP_ESCALATE_RATE = 0.12       # denser review + higher churn (~12% of refund)
REFUND_COST_FLOOR_USD = 10.0       # avoid near-zero costs on tiny refunds
REFUND_COST_CAP_USD = 2000.0       # cap extreme outliers

RANDOM_STATE = 42
TEST_SIZE = 0.20
