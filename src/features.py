"""Feature engineering for return-abuse risk scoring."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.config import LEAKAGE_COLS, TARGET_COL


CATEGORICAL_COLS = [
    "customer_segment",
    "country",
    "platform",
    "device_type",
    "payment_method",
    "product_category",
    "return_reason",
    "shipping_carrier",
    "account_age_bucket",
]

NUMERIC_BASE = [
    "age",
    "account_age_days",
    "avg_order_value_usd",
    "refund_amount_requested_usd",
    "is_high_value_item",
    "discount_used",
    "days_to_return",
    "total_orders_lifetime",
    "total_returns_lifetime",
    "return_rate_pct",
    "item_returned_opened",
    "return_packaging_intact",
    "photo_evidence_provided",
    "tracking_number_valid",
    "address_change_before_delivery",
    "refund_to_different_account",
    "multiple_accounts_flag",
    "customer_support_contacts",
    "previous_dispute_count",
    "wishlist_to_cart_time_hrs",
    "review_left_after_return",
]

ENGINEERED_NUMERIC = [
    "refund_to_aov_ratio",
    "returns_per_order",
    "category_abuse_rate",
    "fast_return_flag",
    "high_return_rate_flag",
    "dispute_intensity",
]


def _account_age_bucket(days: pd.Series) -> pd.Series:
    bins = [-np.inf, 30, 90, 365, 730, np.inf]
    labels = ["0-30d", "31-90d", "91-365d", "1-2y", "2y+"]
    return pd.cut(days, bins=bins, labels=labels).astype(str)


def engineer_features(
    df: pd.DataFrame,
    category_abuse_rate: dict[str, float] | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add engineered columns. When fit=True, compute category abuse rates from labels."""
    out = df.copy()
    out["account_age_bucket"] = _account_age_bucket(out["account_age_days"])

    aov = out["avg_order_value_usd"].replace(0, np.nan)
    out["refund_to_aov_ratio"] = (
        out["refund_amount_requested_usd"] / aov
    ).fillna(0.0).clip(0, 10)

    orders = out["total_orders_lifetime"].replace(0, np.nan)
    out["returns_per_order"] = (
        out["total_returns_lifetime"] / orders
    ).fillna(0.0).clip(0, 5)

    out["fast_return_flag"] = (out["days_to_return"] <= 3).astype(int)
    out["high_return_rate_flag"] = (out["return_rate_pct"] >= 30).astype(int)
    out["dispute_intensity"] = (
        out["previous_dispute_count"] + out["customer_support_contacts"]
    ).astype(float)

    if fit:
        if TARGET_COL not in out.columns:
            raise ValueError("fit=True requires abuse_type labels")
        abuse = out[TARGET_COL] != "Legitimate"
        rates = (
            out.assign(_abuse=abuse.astype(float))
            .groupby("product_category")["_abuse"]
            .mean()
            .to_dict()
        )
        category_abuse_rate = {str(k): float(v) for k, v in rates.items()}

    if category_abuse_rate is None:
        category_abuse_rate = {}

    global_rate = float(np.mean(list(category_abuse_rate.values()))) if category_abuse_rate else 0.0
    out["category_abuse_rate"] = (
        out["product_category"].astype(str).map(category_abuse_rate).fillna(global_rate)
    )

    return out, category_abuse_rate


def feature_columns() -> list[str]:
    return NUMERIC_BASE + ENGINEERED_NUMERIC + CATEGORICAL_COLS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_BASE + ENGINEERED_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


def prepare_xy(
    df: pd.DataFrame,
    category_abuse_rate: dict[str, float] | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None, dict[str, Any]]:
    engineered, rates = engineer_features(df, category_abuse_rate=category_abuse_rate, fit=fit)
    cols = feature_columns()
    missing = [c for c in cols if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing feature columns after engineering: {missing}")
    X = engineered[cols].copy()
    y = engineered[TARGET_COL] if TARGET_COL in engineered.columns else None
    meta = {
        "category_abuse_rate": rates,
        "feature_columns": cols,
        "dropped_leakage": sorted(LEAKAGE_COLS & set(df.columns)),
    }
    return X, y, meta
