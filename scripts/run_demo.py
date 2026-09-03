"""Demo: score a few returns including an insufficient-signal case."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.logger import AuditLogger
from src.data import load_split
from src.detector.model import ReturnRiskDetector
from src.pipeline import score_return


def main() -> None:
    _, test_df = load_split()
    detector = ReturnRiskDetector.load()
    audit = AuditLogger()

    print("=== Example A: random holdout return ===")
    row = test_df.sample(1, random_state=7).iloc[0]
    a = score_return(row, detector, audit=audit, use_llm=True)
    print(json.dumps(a, indent=2)[:2000])

    print("\n=== Example B: high-risk looking slice ===")
    risky = test_df[
        (test_df["abuse_type"] != "Legitimate")
        & (test_df["return_rate_pct"] > 20)
    ]
    if len(risky):
        b = score_return(
            risky.sample(1, random_state=3).iloc[0],
            detector,
            audit=audit,
            use_llm=True,
        )
        print(json.dumps(b, indent=2)[:2000])

    print("\n=== Example C: insufficient_signal (deliberate incomplete record) ===")
    bad = row.to_dict()
    bad["order_id"] = "DEMO-INSUFFICIENT-001"
    bad["product_category"] = ""
    bad["total_returns_lifetime"] = int(bad.get("total_orders_lifetime", 0) or 0) + 5
    c = score_return(bad, detector, audit=audit, use_llm=True)
    print(json.dumps(c, indent=2)[:2000])
    print("\nDemo complete. Audit tail written to outputs/audit/decisions.jsonl")


if __name__ == "__main__":
    main()
