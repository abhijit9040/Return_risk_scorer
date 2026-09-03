"""Streamlit dashboard — batch results, metrics, audit samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.logger import AuditLogger
from src.config import METRICS_DIR, MODELS_DIR, PLOTS_DIR
from src.data import load_split
from src.detector.model import ReturnRiskDetector
from src.pipeline import score_return
from src.verifier.policy import POLICY_TABLE


st.set_page_config(page_title="Return-Risk Scorer", layout="wide")
st.title("Return-Risk Scorer & Auto-Responder")
st.caption("Defense-only AI Risk Manager — Detector → Verifier → Auto-Responder → Audit")


@st.cache_resource
def get_detector() -> ReturnRiskDetector:
    return ReturnRiskDetector.load()


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


tabs = st.tabs(["Score a return", "Holdout metrics", "Policy", "Audit trail"])

with tabs[0]:
    st.subheader("Live score")
    if not (MODELS_DIR / "detector.joblib").exists():
        st.warning("Model not trained yet. Run: python scripts/train.py")
    else:
        _, test_df = load_split()
        idx = st.number_input("Holdout row index", min_value=0, max_value=len(test_df) - 1, value=0)
        force_bad = st.checkbox("Force insufficient_signal (blank category + contradictory returns)")
        if st.button("Score", type="primary"):
            row = test_df.iloc[int(idx)].to_dict()
            if force_bad:
                row["order_id"] = f"UI-INSUFFICIENT-{idx}"
                row["product_category"] = ""
                row["total_returns_lifetime"] = int(row.get("total_orders_lifetime", 0) or 0) + 3
            det = get_detector()
            audit = AuditLogger()
            result = score_return(row, det, audit=audit)
            st.json(result)

with tabs[1]:
    report = load_json(METRICS_DIR / "holdout_report.json")
    clf = load_json(METRICS_DIR / "classification_metrics.json")
    if report:
        c1, c2, c3 = st.columns(3)
        c1.metric("Batch size", report.get("batch_size"))
        c2.metric("Macro F1", round(report.get("classifier_macro_f1", 0), 3))
        c3.metric("Exception rate %", report.get("exception_rate_pct"))
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
    st.dataframe(pd.DataFrame(POLICY_TABLE), width="stretch")
    policy = load_json(METRICS_DIR / "policy_table.json")
    if policy:
        st.json(policy)

with tabs[3]:
    audit = AuditLogger()
    n = st.slider("Entries", 5, 50, 10)
    entries = audit.tail(n)
    if entries:
        st.json(entries)
    else:
        st.info("No audit entries yet. Score a return or run scripts/run_demo.py")
