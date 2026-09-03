"""Policy verifier: refund-scaled, cost-weighted thresholds + class-aware actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from src.config import (
    COST_FN_ABUSE_RATE,
    COST_FP_ESCALATE_RATE,
    COST_FP_HOLD_RATE,
    REFUND_COST_CAP_USD,
    REFUND_COST_FLOOR_USD,
)

Action = Literal["auto_approve", "hold_for_evidence", "escalate_to_human", "insufficient_signal"]

# On highly separable data, risk_score is often ~0 or ~1. Class-aware routing
# keeps the medium "hold for evidence" path meaningful for abuse subtypes
# where photos/packaging proof is the right next step.
HOLD_CLASSES = {"Wardrobing", "Policy Abuser"}
ESCALATE_CLASSES = {"Fraudulent Return"}


@dataclass
class PolicyThresholds:
    """Risk-score bands (risk_score = 1 - P(Legitimate))."""

    approve_max: float = 0.25
    hold_max: float = 0.55
    min_confidence: float = 0.40

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class CostAssumptions:
    """Rates multiplied by capped refund exposure (not flat dollars)."""

    fn_abuse_rate: float = COST_FN_ABUSE_RATE
    fp_hold_rate: float = COST_FP_HOLD_RATE
    fp_escalate_rate: float = COST_FP_ESCALATE_RATE
    refund_floor_usd: float = REFUND_COST_FLOOR_USD
    refund_cap_usd: float = REFUND_COST_CAP_USD

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_POLICY = PolicyThresholds()
DEFAULT_COSTS = CostAssumptions()


POLICY_TABLE = [
    {
        "band": "Low",
        "rule": f"risk_score < {DEFAULT_POLICY.approve_max}",
        "action": "auto_approve",
        "rationale": "Low cost of being wrong; fast approval preserves CX.",
    },
    {
        "band": "Medium",
        "rule": (
            f"{DEFAULT_POLICY.approve_max} <= risk_score <= {DEFAULT_POLICY.hold_max} "
            "OR predicted Wardrobing/Policy Abuser at high risk"
        ),
        "action": "hold_for_evidence",
        "rationale": "Ambiguous or evidence-resolvable abuse pattern; request proof before refund.",
    },
    {
        "band": "High",
        "rule": (
            f"risk_score > {DEFAULT_POLICY.hold_max} and Fraudulent Return "
            "(or low-confidence prediction)"
        ),
        "action": "escalate_to_human",
        "rationale": "High-confidence fraud signal or low confidence; never auto-deny — human reviews.",
    },
    {
        "band": "Insufficient signal",
        "rule": "missing/contradictory required fields",
        "action": "insufficient_signal",
        "rationale": "Missing/contradictory fields — do not force a classification.",
    },
]


def refund_exposure_usd(
    refund_amount_usd: float,
    costs: CostAssumptions = DEFAULT_COSTS,
) -> float:
    """Clamp refund into [floor, cap] so costs scale with $ at stake."""
    try:
        refund = float(refund_amount_usd)
    except (TypeError, ValueError):
        refund = 0.0
    if refund != refund:  # NaN
        refund = 0.0
    return min(max(refund, costs.refund_floor_usd), costs.refund_cap_usd)


def expected_action_cost(
    risk_score: float,
    action: Action,
    refund_amount_usd: float,
    costs: CostAssumptions = DEFAULT_COSTS,
) -> float:
    """Expected $ cost for an action, scaled by refund exposure.

    auto_approve:    P(abuse) * fn_rate * refund
    hold:            P(legit) * fp_hold_rate * refund
                     + 0.15 * P(abuse) * fn_rate * refund  (residual miss risk while held)
    escalate:        P(legit) * fp_escalate_rate * refund
    """
    exposure = refund_exposure_usd(refund_amount_usd, costs)
    p_abuse = float(risk_score)
    p_legit = 1.0 - p_abuse
    if action == "auto_approve":
        return p_abuse * costs.fn_abuse_rate * exposure
    if action == "hold_for_evidence":
        return (
            p_legit * costs.fp_hold_rate * exposure
            + 0.15 * p_abuse * costs.fn_abuse_rate * exposure
        )
    if action == "escalate_to_human":
        return p_legit * costs.fp_escalate_rate * exposure
    return 0.0


def choose_action(
    risk_score: float,
    confidence: float,
    predicted_class: str | None = None,
    refund_amount_usd: float = 0.0,
    insufficient_signal: bool = False,
    policy: PolicyThresholds = DEFAULT_POLICY,
    costs: CostAssumptions = DEFAULT_COSTS,
) -> dict[str, Any]:
    exposure = refund_exposure_usd(refund_amount_usd, costs)
    if insufficient_signal:
        return {
            "action": "insufficient_signal",
            "band": "Insufficient signal",
            "rationale": "Key fields missing or contradictory; routed to manual review.",
            "expected_cost_usd": 0.0,
            "refund_amount_usd": round(float(refund_amount_usd or 0.0), 2),
            "refund_exposure_usd": round(exposure, 2),
            "policy": policy.to_dict(),
            "costs": costs.to_dict(),
        }

    candidates: list[Action] = ["auto_approve", "hold_for_evidence", "escalate_to_human"]
    cost_map = {
        a: expected_action_cost(risk_score, a, refund_amount_usd, costs) for a in candidates
    }
    pred = predicted_class or ""

    if risk_score < policy.approve_max and confidence >= policy.min_confidence:
        action: Action = "auto_approve"
        band = "Low"
    elif confidence < policy.min_confidence:
        action = "escalate_to_human"
        band = "High"
    elif policy.approve_max <= risk_score <= policy.hold_max:
        action = "hold_for_evidence"
        band = "Medium"
    elif pred in HOLD_CLASSES:
        # Evidence-resolvable subtype → hold (photos/packaging), still never auto-deny.
        action = "hold_for_evidence"
        band = "Medium"
    elif pred in ESCALATE_CLASSES or risk_score > policy.hold_max:
        action = "escalate_to_human"
        band = "High"
    else:
        action = min(
            ["hold_for_evidence", "escalate_to_human"],
            key=lambda a: cost_map[a],
        )
        band = "Medium" if action == "hold_for_evidence" else "High"

    return {
        "action": action,
        "band": band,
        "rationale": next(r["rationale"] for r in POLICY_TABLE if r["action"] == action),
        "expected_cost_usd": round(cost_map.get(action, 0.0), 2),
        "candidate_costs_usd": {k: round(v, 2) for k, v in cost_map.items()},
        "refund_amount_usd": round(float(refund_amount_usd or 0.0), 2),
        "refund_exposure_usd": round(exposure, 2),
        "predicted_class": predicted_class,
        "policy": policy.to_dict(),
        "costs": costs.to_dict(),
    }


def estimate_batch_false_positive_cost(
    y_true: Sequence[str],
    actions: Sequence[str],
    refund_amounts: Sequence[float] | None = None,
    costs: CostAssumptions = DEFAULT_COSTS,
) -> dict[str, Any]:
    """Honest FP/FN $ cost on held-out, scaled by each return's refund amount."""
    fp_hold = 0
    fp_esc = 0
    fn_approve = 0
    total_fp_usd = 0.0
    total_fn_usd = 0.0

    n = len(y_true)
    if refund_amounts is None:
        refund_amounts = [costs.refund_floor_usd] * n

    for yt, act, refund in zip(y_true, actions, refund_amounts):
        exposure = refund_exposure_usd(refund, costs)
        is_legit = yt == "Legitimate"
        is_abuse = not is_legit
        if is_legit and act == "hold_for_evidence":
            fp_hold += 1
            total_fp_usd += costs.fp_hold_rate * exposure
        elif is_legit and act == "escalate_to_human":
            fp_esc += 1
            total_fp_usd += costs.fp_escalate_rate * exposure
        elif is_abuse and act == "auto_approve":
            fn_approve += 1
            total_fn_usd += costs.fn_abuse_rate * exposure

    n = max(n, 1)
    return {
        "fp_hold_count": fp_hold,
        "fp_escalate_count": fp_esc,
        "fn_auto_approve_count": fn_approve,
        "fp_cost_usd": round(total_fp_usd, 2),
        "fn_cost_usd": round(total_fn_usd, 2),
        "total_decision_cost_usd": round(total_fp_usd + total_fn_usd, 2),
        "avg_cost_per_return_usd": round((total_fp_usd + total_fn_usd) / n, 4),
        "assumptions": costs.to_dict(),
        "note": (
            "FP/FN dollars = rate * capped refund_amount_requested_usd per case. "
            "FP = legitimate customer held/escalated; FN = abusive return auto-approved."
        ),
    }
