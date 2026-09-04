"""Manual-entry numeric bounds derived from the training split."""
from __future__ import annotations

from typing import Any

import pandas as pd

MANUAL_BOUND_FIELDS = (
    "refund_amount_requested_usd",
    "days_to_return",
    "return_rate_pct",
    "account_age_days",
)


def compute_manual_field_bounds(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """1st–99th percentile bounds for manual numeric inputs.

    Using percentiles (not raw max) avoids a single outlier stretching the UI
    into out-of-distribution territory.
    """
    bounds: dict[str, dict[str, float]] = {}
    for col in MANUAL_BOUND_FIELDS:
        if col not in train_df.columns:
            raise KeyError(f"Missing column for manual bounds: {col}")
        series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        lo = float(series.quantile(0.01))
        hi = float(series.quantile(0.99))
        if hi < lo:
            lo, hi = hi, lo
        bounds[col] = {"p01": lo, "p99": hi, "max": hi}
    return bounds


def is_out_of_bounds(
    field: str,
    value: float,
    bounds: dict[str, dict[str, float]],
    *,
    min_value: float = 0.0,
) -> bool:
    """True when value is outside [min_value, bounds[field]['max']]."""
    if field not in bounds:
        raise KeyError(field)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    if v != v:  # NaN
        return True
    return v < min_value or v > float(bounds[field]["max"])
