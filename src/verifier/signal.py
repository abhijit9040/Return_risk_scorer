"""Insufficient-signal gate — never force a classification on bad inputs."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import REQUIRED_FIELDS


def check_insufficient_signal(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Return whether the record should skip scoring and go to manual review."""
    data = row if isinstance(row, dict) else row.to_dict()
    issues: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in data or data[field] is None or (isinstance(data[field], float) and pd.isna(data[field])):
            issues.append(f"missing:{field}")
        elif isinstance(data[field], str) and not str(data[field]).strip():
            issues.append(f"empty:{field}")

    # Contradictions
    try:
        orders = float(data.get("total_orders_lifetime", 0) or 0)
        returns = float(data.get("total_returns_lifetime", 0) or 0)
        if returns > orders > 0:
            issues.append("contradiction:returns_exceed_orders")
        refund = float(data.get("refund_amount_requested_usd", 0) or 0)
        if refund < 0:
            issues.append("contradiction:negative_refund")
        days = float(data.get("days_to_return", 0) or 0)
        if days < 0:
            issues.append("contradiction:negative_days_to_return")
        account_age = float(data.get("account_age_days", 0) or 0)
        if account_age < 0:
            issues.append("contradiction:negative_account_age")
    except (TypeError, ValueError):
        issues.append("contradiction:non_numeric_core_fields")

    return {
        "insufficient_signal": len(issues) > 0,
        "issues": issues,
    }
