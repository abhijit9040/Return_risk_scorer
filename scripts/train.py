"""Day 1â€“2: load data, EDA, split, train detector, write policy table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR
from src.data import load_raw, run_eda, stratified_split
from src.detector.model import ReturnRiskDetector
from src.verifier.policy import DEFAULT_COSTS, DEFAULT_POLICY, POLICY_TABLE


def main() -> None:
    print("Loading dataset...")
    df = load_raw()
    print(f"Rows={len(df):,} cols={df.shape[1]}")

    print("Running EDA...")
    eda = run_eda(df)
    print("Class balance:", eda["class_balance"])

    print("Stratified split (hold out 20%)...")
    train_df, test_df = stratified_split(df)
    print(f"Train={len(train_df):,} Test={len(test_df):,}")

    print("Training LightGBM detector...")
    detector = ReturnRiskDetector().fit(train_df)
    path = detector.save()
    print(f"Saved model -> {path}")

    print("Quick eval on holdout (for training feedback only)...")
    metrics = detector.evaluate(test_df)
    print("Macro F1:", round(metrics["macro_f1"], 4))
    print("Per-class F1:", {k: round(v, 4) for k, v in metrics["f1_per_class"].items()})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    policy_doc = {
        "thresholds": DEFAULT_POLICY.to_dict(),
        "cost_assumptions": DEFAULT_COSTS.to_dict(),
        "cost_formula": (
            "expected_cost = rate * clamp(refund_amount_requested_usd, floor, cap) "
            "* probability term; not flat dollars"
        ),
        "policy_table": POLICY_TABLE,
        "risk_score_definition": "risk_score = 1 - P(Legitimate)",
    }
    (METRICS_DIR / "policy_table.json").write_text(
        json.dumps(policy_doc, indent=2), encoding="utf-8"
    )
    print("Wrote policy table -> outputs/metrics/policy_table.json")
    print("Done.")


if __name__ == "__main__":
    main()
