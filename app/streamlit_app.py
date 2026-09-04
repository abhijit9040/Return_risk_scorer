"""Streamlit dashboard — batch results, metrics, audit samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.logger import AuditLogger
from src.bounds import compute_manual_field_bounds
from src.config import METRICS_DIR, MODELS_DIR, PLOTS_DIR, TARGET_COL, LABEL_ID_COL
from src.data import load_split
from src.detector.model import ReturnRiskDetector
from src.pipeline import score_return
from src.verifier.policy import POLICY_TABLE


st.set_page_config(page_title="Return-Risk Scorer", layout="wide")
st.title("Return-Risk Scorer & Auto-Responder")
st.caption("Defense-only AI Risk Manager — Detector → Verifier → Auto-Responder → Audit")

ACTION_LABELS = {
    "auto_approve": "Auto-approve",
    "hold_for_evidence": "Hold for evidence",
    "escalate_to_human": "Escalate to human",
    "insufficient_signal": "Insufficient signal → manual review",
}


@st.cache_resource
def get_detector() -> ReturnRiskDetector:
    return ReturnRiskDetector.load()


@st.cache_data
def get_split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_split()


@st.cache_data
def get_row_defaults() -> dict[str, Any]:
    """Median / mode defaults from the training split for columns not on the form."""
    train_df, _ = get_split_frames()
    defaults: dict[str, Any] = {}
    skip = {TARGET_COL, LABEL_ID_COL}
    for col in train_df.columns:
        if col in skip:
            continue
        series = train_df[col]
        if pd.api.types.is_numeric_dtype(series):
            val = series.median()
            # Preserve int-like columns as int when possible
            if pd.api.types.is_integer_dtype(series):
                defaults[col] = int(val)
            else:
                defaults[col] = float(val)
        else:
            mode = series.mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if len(mode) else ""
    defaults["order_id"] = "MANUAL-001"
    defaults["customer_id"] = "CUST-MANUAL"
    return defaults


@st.cache_data
def get_category_options() -> dict[str, list[str]]:
    train_df, _ = get_split_frames()
    options: dict[str, list[str]] = {}
    for col in ("product_category", "payment_method", "return_reason"):
        options[col] = sorted(train_df[col].astype(str).unique().tolist())
    return options


@st.cache_data
def get_manual_field_bounds() -> dict[str, dict[str, float]]:
    """1st–99th percentile bounds from train_df for manual numeric inputs."""
    train_df, _ = get_split_frames()
    return compute_manual_field_bounds(train_df)

def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def render_action_badge(action: str | None) -> None:
    label = ACTION_LABELS.get(action or "", action or "unknown")
    if action == "auto_approve":
        st.success(f"**Decision:** {label}")
    elif action == "hold_for_evidence":
        st.warning(f"**Decision:** {label}")
    elif action == "escalate_to_human":
        st.error(f"**Decision:** {label}")
    elif action == "insufficient_signal":
        st.info(f"**Decision:** {label}")
    else:
        st.write(f"**Decision:** {label}")


def render_score_result(result: dict[str, Any]) -> None:
    """Readable score view; full JSON stays available in an expander."""
    render_action_badge(result.get("action"))

    m1, m2, m3 = st.columns(3)
    risk = result.get("risk_score")
    m1.metric("Risk score", "—" if risk is None else f"{float(risk):.4f}")
    pred = result.get("predicted_class")
    m2.metric("Predicted class", pred if pred else "—")
    conf = result.get("confidence")
    m3.metric("Confidence", "—" if conf is None else f"{float(conf):.4f}")

    reasons = result.get("reason_codes") or []
    if reasons:
        st.markdown("**Top reason codes**")
        for r in reasons:
            feature = r.get("feature", "?")
            direction = r.get("direction", "")
            direction_txt = direction.replace("_", " ") if direction else "n/a"
            st.markdown(f"- `{feature}` — {direction_txt}")
    elif result.get("insufficient_signal"):
        issues = result.get("issues") or []
        st.markdown("**Signal issues**")
        for issue in issues:
            st.markdown(f"- `{issue}`")

    case_note = result.get("case_note")
    if case_note:
        st.markdown("**Internal case note**")
        st.write(case_note)

    customer_message = result.get("customer_message")
    if customer_message:
        st.markdown("**Customer message**")
        st.write(customer_message)

    with st.expander("Raw output (JSON)"):
        st.json(result)


tabs = st.tabs(["Score a return", "Holdout metrics", "Policy", "Audit trail"])

with tabs[0]:
    st.subheader("Live score")
    st.caption(
        "Score a single return end-to-end: Detector → Verifier → Auto-Responder → Audit. "
        "Pick a held-out row or enter key fields manually."
    )
    if not (MODELS_DIR / "detector.joblib").exists():
        st.warning("Model not trained yet. Run: python scripts/train.py")
    else:
        _, test_df = get_split_frames()
        defaults = get_row_defaults()
        cat_options = get_category_options()

        mode = st.radio(
            "Input mode",
            ["Holdout row", "Manual entry"],
            horizontal=True,
        )

        if mode == "Holdout row":
            idx = st.number_input(
                "Holdout row index",
                min_value=0,
                max_value=len(test_df) - 1,
                value=0,
            )
            force_bad = st.checkbox(
                "Force insufficient_signal (blank category + contradictory returns)"
            )
            if st.button("Score", type="primary", key="score_holdout"):
                row = test_df.iloc[int(idx)].to_dict()
                if force_bad:
                    row["order_id"] = f"UI-INSUFFICIENT-{idx}"
                    row["product_category"] = ""
                    row["total_returns_lifetime"] = (
                        int(row.get("total_orders_lifetime", 0) or 0) + 3
                    )
                det = get_detector()
                audit = AuditLogger()
                result = score_return(row, det, audit=audit)
                render_score_result(result)
        else:
            bounds = get_manual_field_bounds()
            refund_max = float(bounds["refund_amount_requested_usd"]["max"])
            days_max = int(round(bounds["days_to_return"]["max"]))
            rate_max = float(bounds["return_rate_pct"]["max"])
            age_max = int(round(bounds["account_age_days"]["max"]))
            orders_max = int(round(bounds["total_orders_lifetime"]["max"]))
            returns_max = int(round(bounds["total_returns_lifetime"]["max"]))
            disputes_max = int(round(bounds["previous_dispute_count"]["max"]))
            support_max = int(round(bounds["customer_support_contacts"]["max"]))

            refund_p01 = float(bounds["refund_amount_requested_usd"]["p01"])
            days_p01 = float(bounds["days_to_return"]["p01"])
            rate_p01 = float(bounds["return_rate_pct"]["p01"])
            age_p01 = float(bounds["account_age_days"]["p01"])

            refund_default = min(float(defaults["refund_amount_requested_usd"]), refund_max)
            days_default = min(int(defaults["days_to_return"]), days_max)
            rate_default = min(float(defaults["return_rate_pct"]), rate_max)
            age_default = min(int(defaults["account_age_days"]), age_max)
            orders_default = min(int(defaults["total_orders_lifetime"]), orders_max)
            returns_default = min(int(defaults["total_returns_lifetime"]), returns_max)
            disputes_default = min(int(defaults["previous_dispute_count"]), disputes_max)
            support_default = min(int(defaults["customer_support_contacts"]), support_max)

            st.info(
                "Other columns default to training medians (typical legitimate shopper). "
                "Changing refund/category alone rarely flips the decision — raise "
                "**return rate %**, **lifetime returns**, and **disputes** to see hold/escalate."
            )

            preset = st.selectbox(
                "Quick preset (fills the form fields below)",
                [
                    "Custom (use fields below)",
                    "Legitimate (low risk)",
                    "Policy abuser (hold)",
                    "Fraudulent return (escalate)",
                ],
                index=0,
            )
            if preset == "Legitimate (low risk)":
                refund_default, days_default, rate_default = 80.0, 14, 5.0
                age_default, orders_default, returns_default = 800, 40, 2
                disputes_default, support_default = 0, 0
            elif preset == "Policy abuser (hold)":
                refund_default, days_default, rate_default = 120.0, 10, 65.0
                age_default, orders_default, returns_default = 600, 20, 12
                disputes_default, support_default = 3, 4
            elif preset == "Fraudulent return (escalate)":
                refund_default, days_default, rate_default = 450.0, 3, 55.0
                age_default, orders_default, returns_default = 90, 8, 5
                disputes_default, support_default = 4, 5

            # Clamp presets into bounds
            refund_default = min(max(0.0, refund_default), refund_max)
            days_default = min(max(0, days_default), days_max)
            rate_default = min(max(0.0, rate_default), rate_max)
            age_default = min(max(0, age_default), age_max)
            orders_default = min(max(0, orders_default), orders_max)
            returns_default = min(max(0, returns_default), min(returns_max, orders_default))
            disputes_default = min(max(0, disputes_default), disputes_max)
            support_default = min(max(0, support_default), support_max)

            with st.form("manual_score_form"):
                st.markdown("**Key return fields**")
                c1, c2 = st.columns(2)
                with c1:
                    refund = st.number_input(
                        "Refund amount (USD)",
                        min_value=0.0,
                        max_value=refund_max,
                        value=refund_default,
                        step=1.0,
                    )
                    days_to_return = st.number_input(
                        "Days to return",
                        min_value=0,
                        max_value=days_max,
                        value=days_default,
                        step=1,
                    )
                    return_rate = st.number_input(
                        "Return rate (%) — strong risk driver",
                        min_value=0.0,
                        max_value=rate_max,
                        value=rate_default,
                        step=0.1,
                    )
                    account_age = st.number_input(
                        "Account age (days)",
                        min_value=0,
                        max_value=age_max,
                        value=age_default,
                        step=1,
                    )
                    total_orders = st.number_input(
                        "Total orders (lifetime)",
                        min_value=0,
                        max_value=orders_max,
                        value=orders_default,
                        step=1,
                    )
                    total_returns = st.number_input(
                        "Total returns (lifetime)",
                        min_value=0,
                        max_value=returns_max,
                        value=returns_default,
                        step=1,
                    )
                with c2:
                    previous_disputes = st.number_input(
                        "Previous dispute count",
                        min_value=0,
                        max_value=disputes_max,
                        value=disputes_default,
                        step=1,
                    )
                    support_contacts = st.number_input(
                        "Customer support contacts",
                        min_value=0,
                        max_value=support_max,
                        value=support_default,
                        step=1,
                    )
                    high_value = st.checkbox(
                        "High-value item",
                        value=bool(int(defaults.get("is_high_value_item", 0))),
                    )
                    opened = st.checkbox(
                        "Item returned opened",
                        value=bool(int(defaults.get("item_returned_opened", 0))),
                    )
                    product_category = st.selectbox(
                        "Product category",
                        options=cat_options["product_category"],
                        index=cat_options["product_category"].index(
                            defaults["product_category"]
                        )
                        if defaults["product_category"] in cat_options["product_category"]
                        else 0,
                    )
                    payment_method = st.selectbox(
                        "Payment method",
                        options=cat_options["payment_method"],
                        index=cat_options["payment_method"].index(
                            defaults["payment_method"]
                        )
                        if defaults["payment_method"] in cat_options["payment_method"]
                        else 0,
                    )
                    return_reason = st.selectbox(
                        "Return reason",
                        options=cat_options["return_reason"],
                        index=cat_options["return_reason"].index(
                            defaults["return_reason"]
                        )
                        if defaults["return_reason"] in cat_options["return_reason"]
                        else 0,
                    )
                submitted = st.form_submit_button("Score manual return", type="primary")

            st.caption(
                "Numeric inputs are capped at the training data's 1st–99th percentile "
                "range to avoid out-of-distribution predictions: "
                f"refund ${refund_p01:,.0f}–${refund_max:,.0f}, "
                f"days to return {days_p01:.0f}–{days_max}, "
                f"return rate {rate_p01:.1f}–{rate_max:.1f}%, "
                f"account age {age_p01:.0f}–{age_max} days. "
                "Try preset **Policy abuser** or **Fraudulent return** if you want a "
                "non-approve decision quickly."
            )

            if submitted:
                row = dict(defaults)
                orders_val = int(total_orders)
                returns_val = int(total_returns)
                # Keep returns <= orders to avoid insufficient_signal contradiction
                if returns_val > orders_val > 0:
                    returns_val = orders_val
                row.update(
                    {
                        "order_id": "MANUAL-UI-001",
                        "refund_amount_requested_usd": float(refund),
                        "days_to_return": int(days_to_return),
                        "return_rate_pct": float(return_rate),
                        "account_age_days": int(account_age),
                        "total_orders_lifetime": orders_val,
                        "total_returns_lifetime": returns_val,
                        "previous_dispute_count": int(previous_disputes),
                        "customer_support_contacts": int(support_contacts),
                        "is_high_value_item": int(high_value),
                        "item_returned_opened": int(opened),
                        "product_category": product_category,
                        "payment_method": payment_method,
                        "return_reason": return_reason,
                    }
                )

                det = get_detector()
                audit = AuditLogger()
                result = score_return(row, det, audit=audit)
                render_score_result(result)

with tabs[1]:
    st.subheader("Holdout metrics")
    st.caption(
        "Held-out precision, recall, F1, action mix, and refund-scaled false-positive cost — "
        "from a true test split never used in training."
    )
    report = load_json(METRICS_DIR / "holdout_report.json")
    clf = load_json(METRICS_DIR / "classification_metrics.json")
    if report:
        c1, c2, c3 = st.columns(3)
        c1.metric("Batch size", report.get("batch_size"))
        c2.metric("Macro F1", round(report.get("classifier_macro_f1", 0), 3))
        c3.metric("Exception rate %", report.get("exception_rate_pct"))
        st.caption(
            "This dataset is synthetically generated and highly separable — "
            "we verified no label leakage or train/test overlap; see README for the full check. "
            "Differentiation is the policy, refund-scaled FP cost, audit trail, and failure path — "
            "not inflated accuracy claims."
        )
        st.write("Action counts", report.get("action_counts"))
        st.write("False-positive cost", report.get("false_positive_cost"))
        st.write("F1 per class", report.get("classifier_f1_per_class"))
    elif clf:
        st.write("Classification metrics", clf)
    else:
        st.info("Run: python scripts/evaluate.py")
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path))

with tabs[2]:
    st.subheader("Decision policy")
    st.caption(
        "Risk score bands mapped to bounded actions — thresholds tuned against "
        "false-positive cost (scaled by refund amount), not accuracy alone."
    )
    st.dataframe(pd.DataFrame(POLICY_TABLE), width="stretch")
    policy = load_json(METRICS_DIR / "policy_table.json")
    if policy:
        st.json(policy)

with tabs[3]:
    st.subheader("Audit trail")
    st.caption(
        "Recent logged decisions: input summary → score → action → rationale → timestamp. "
        "Every automated decision is human-reversible."
    )
    audit = AuditLogger()
    n = st.slider("Entries", 5, 50, 10)
    entries = audit.tail(n)
    if entries:
        st.json(entries)
    else:
        st.info("No audit entries yet. Score a return or run scripts/run_demo.py")
