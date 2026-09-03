"""End-to-end: Detector → Verifier → Auto-Responder → Audit."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.responder import draft_response
from src.audit.logger import AuditLogger
from src.detector.model import ReturnRiskDetector
from src.verifier.policy import choose_action
from src.verifier.signal import check_insufficient_signal


def score_return(
    row: pd.Series | dict[str, Any],
    detector: ReturnRiskDetector,
    audit: AuditLogger | None = None,
    with_shap: bool = True,
    use_llm: bool = True,
) -> dict[str, Any]:
    data = row if isinstance(row, dict) else row.to_dict()
    return_id = str(data.get("order_id", data.get("return_id", "UNKNOWN")))
    signal = check_insufficient_signal(data)

    if signal["insufficient_signal"]:
        refund = float(data.get("refund_amount_requested_usd") or 0.0)
        decision = choose_action(
            risk_score=0.0,
            confidence=0.0,
            refund_amount_usd=refund,
            insufficient_signal=True,
        )
        agent_payload = {
            "return_id": return_id,
            "risk_score": None,
            "predicted_class": None,
            "action": decision["action"],
            "band": decision["band"],
            "reason_codes": [],
            "rationale": decision["rationale"],
            "issues": signal["issues"],
        }
        messages = draft_response(agent_payload, use_llm=use_llm)
        record = {
            "return_id": return_id,
            "input_summary": {
                k: data.get(k)
                for k in [
                    "product_category",
                    "return_reason",
                    "refund_amount_requested_usd",
                    "account_age_days",
                    "return_rate_pct",
                ]
                if k in data
            },
            "insufficient_signal": True,
            "issues": signal["issues"],
            "risk_score": None,
            "predicted_class": None,
            "confidence": None,
            "reason_codes": [],
            "action": decision["action"],
            "band": decision["band"],
            "rationale": decision["rationale"],
            "case_note": messages["case_note"],
            "customer_message": messages["customer_message"],
            "generator": messages.get("generator"),
        }
        if audit:
            audit.log(record)
        return record

    frame = pd.DataFrame([data])
    scored = detector.score_frame(frame).iloc[0]
    reason_codes = detector.shap_top_reasons(frame, top_k=3)[0] if with_shap else []
    refund = float(data.get("refund_amount_requested_usd") or 0.0)
    decision = choose_action(
        risk_score=float(scored["risk_score"]),
        confidence=float(scored["confidence"]),
        predicted_class=str(scored["predicted_class"]),
        refund_amount_usd=refund,
        insufficient_signal=False,
    )
    agent_payload = {
        "return_id": return_id,
        "risk_score": round(float(scored["risk_score"]), 4),
        "predicted_class": scored["predicted_class"],
        "action": decision["action"],
        "band": decision["band"],
        "reason_codes": reason_codes,
        "rationale": decision["rationale"],
    }
    messages = draft_response(agent_payload, use_llm=use_llm)
    record = {
        "return_id": return_id,
        "input_summary": {
            k: data.get(k)
            for k in [
                "product_category",
                "return_reason",
                "refund_amount_requested_usd",
                "account_age_days",
                "return_rate_pct",
                "payment_method",
            ]
            if k in data
        },
        "insufficient_signal": False,
        "issues": [],
        "risk_score": round(float(scored["risk_score"]), 4),
        "predicted_class": scored["predicted_class"],
        "confidence": round(float(scored["confidence"]), 4),
        "class_probabilities": {
            c.replace("p_", ""): round(float(scored[c]), 4)
            for c in scored.index
            if str(c).startswith("p_")
        },
        "reason_codes": reason_codes,
        "action": decision["action"],
        "band": decision["band"],
        "rationale": decision["rationale"],
        "expected_cost_usd": decision.get("expected_cost_usd"),
        "candidate_costs_usd": decision.get("candidate_costs_usd"),
        "refund_amount_usd": decision.get("refund_amount_usd"),
        "refund_exposure_usd": decision.get("refund_exposure_usd"),
        "case_note": messages["case_note"],
        "customer_message": messages["customer_message"],
        "generator": messages.get("generator"),
    }
    if audit:
        audit.log(record)
    return record


def score_batch(
    df: pd.DataFrame,
    detector: ReturnRiskDetector,
    audit: AuditLogger | None = None,
    with_shap: bool = True,
    use_llm: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    subset = df if limit is None else df.head(limit)
    records: list[dict[str, Any]] = []
    total = len(subset)
    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        records.append(
            score_return(
                row,
                detector,
                audit=audit,
                with_shap=with_shap,
                use_llm=use_llm,
            )
        )
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  scored {i}/{total}", flush=True)
    return records
