"""Day 4: held-out metrics, FP cost, exception rate, sample audits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.logger import AuditLogger
from src.config import METRICS_DIR, PLOTS_DIR
from src.data import load_split
from src.detector.model import ReturnRiskDetector
from src.pipeline import score_batch
from src.verifier.policy import estimate_batch_false_positive_cost


def main(batch_size: int = 80, with_shap: bool = True, use_llm: bool = False) -> None:
    _, test_df = load_split()
    detector = ReturnRiskDetector.load()

    print("Classifier metrics on full holdout...")
    clf_metrics = detector.evaluate(test_df)

    # Confusion matrix plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = clf_metrics["labels"]
    cm = clf_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    mode = "LLM" if use_llm else "template"
    print(
        f"Scoring end-to-end batch (n={batch_size}, shap={with_shap}, responder={mode})..."
    )
    audit = AuditLogger(jsonl_path=METRICS_DIR.parent / "audit" / "holdout_batch.jsonl")
    # Fresh file for this run
    if audit.jsonl_path.exists():
        audit.jsonl_path.unlink()
    sample = test_df.sample(n=min(batch_size, len(test_df)), random_state=42)
    records = score_batch(
        sample,
        detector,
        audit=audit,
        with_shap=with_shap,
        use_llm=use_llm,
    )

    actions = [r["action"] for r in records]
    y_true = sample["abuse_type"].tolist()
    refunds = sample["refund_amount_requested_usd"].tolist()
    fp = estimate_batch_false_positive_cost(y_true, actions, refund_amounts=refunds)

    exception_n = sum(1 for r in records if r["action"] == "insufficient_signal")
    summary = {
        "batch_size": len(records),
        "action_counts": pd.Series(actions).value_counts().to_dict(),
        "exception_rate_pct": round(100.0 * exception_n / max(len(records), 1), 3),
        "false_positive_cost": fp,
        "classifier_macro_f1": clf_metrics["macro_f1"],
        "classifier_f1_per_class": clf_metrics["f1_per_class"],
        "precision_per_class": clf_metrics["precision_per_class"],
        "recall_per_class": clf_metrics["recall_per_class"],
        "sample_audit_entries": records[:5],
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "holdout_report.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "sample_audit_entries"}, indent=2))
    print(f"Wrote {out}")
    print(f"Audit log -> {audit.jsonl_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--no-shap", action="store_true")
    p.add_argument(
        "--llm",
        action="store_true",
        help="Call OpenAI for every row (slow; default uses templates)",
    )
    args = p.parse_args()
    main(batch_size=args.batch_size, with_shap=not args.no_shap, use_llm=args.llm)
