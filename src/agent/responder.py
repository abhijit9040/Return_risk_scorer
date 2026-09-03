"""LLM auto-responder: internal case note + customer-facing message."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a defense-only returns risk assistant for an e-commerce merchant.
You never recommend account bans, payment blocks, or permanent auto-denials.
You write (1) a concise internal case note for reviewers and (2) a polite customer message
only when the decision is hold_for_evidence or escalate_to_human.
If action is auto_approve or insufficient_signal, set customer_message to null.
Respond with JSON only matching the schema provided.
JSON keys: case_note (string), customer_message (string|null)."""


def _fallback_response(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = payload.get("reason_codes") or []
    reason_txt = ", ".join(
        f"{r.get('feature')} ({r.get('direction')})" for r in reasons[:3]
    ) or "model signals"
    action = payload.get("action")
    case_note = (
        f"Return {payload.get('return_id')}: action={action}, "
        f"risk_score={payload.get('risk_score')}, predicted={payload.get('predicted_class')}. "
        f"Top drivers: {reason_txt}. Policy band: {payload.get('band')}."
    )
    customer_message = None
    if action == "hold_for_evidence":
        customer_message = (
            "Thanks for your return request. To complete processing, please reply with "
            "clear photos of the item and original packaging. We will review promptly."
        )
    elif action == "escalate_to_human":
        customer_message = (
            "Thanks for contacting us about your return. A specialist is reviewing your "
            "request and will follow up shortly with next steps."
        )
    elif action == "insufficient_signal":
        case_note += " Routed to manual review due to insufficient signal."
    return {
        "case_note": case_note,
        "customer_message": customer_message,
        "generator": "fallback_template",
    }


def _user_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_id": payload.get("return_id"),
        "risk_score": payload.get("risk_score"),
        "predicted_class": payload.get("predicted_class"),
        "decision": payload.get("action"),
        "band": payload.get("band"),
        "reason_codes": payload.get("reason_codes"),
        "rationale": payload.get("rationale"),
        "issues": payload.get("issues"),
    }


def _parse_json_text(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _draft_with_gemini(payload: dict[str, Any]) -> dict[str, Any] | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            "Produce JSON with keys: case_note (string), customer_message (string|null).\n"
            + json.dumps(_user_payload(payload))
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        data = _parse_json_text(response.text or "{}")
        return {
            "case_note": data.get("case_note") or _fallback_response(payload)["case_note"],
            "customer_message": data.get("customer_message"),
            "generator": f"gemini:{model_name}",
        }
    except Exception as exc:  # noqa: BLE001
        fb = _fallback_response(payload)
        fb["generator"] = f"fallback_after_gemini_error:{type(exc).__name__}"
        fb["error"] = str(exc)
        return fb


def _draft_with_openai(payload: dict[str, Any]) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or api_key.startswith("sk-your-key"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=20.0, max_retries=0)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Produce JSON with keys: case_note (string), "
                        "customer_message (string|null).\n"
                        + json.dumps(_user_payload(payload))
                    ),
                },
            ],
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        return {
            "case_note": data.get("case_note") or _fallback_response(payload)["case_note"],
            "customer_message": data.get("customer_message"),
            "generator": f"openai:{model}",
        }
    except Exception as exc:  # noqa: BLE001
        fb = _fallback_response(payload)
        fb["generator"] = f"fallback_after_openai_error:{type(exc).__name__}"
        fb["error"] = str(exc)
        return fb


def draft_response(
    payload: dict[str, Any],
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Draft via Gemini (preferred), then OpenAI, then template fallback."""
    if not use_llm:
        return _fallback_response(payload)

    provider = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()

    if provider in {"auto", "gemini"}:
        result = _draft_with_gemini(payload)
        if result is not None and not str(result.get("generator", "")).startswith(
            "fallback_after_gemini_error"
        ):
            return result
        if provider == "gemini":
            return result or _fallback_response(payload)
        # auto: if gemini hard-failed, try openai next
        if result is not None and str(result.get("generator", "")).startswith(
            "fallback_after_gemini_error"
        ):
            openai_result = _draft_with_openai(payload)
            if openai_result is not None and not str(
                openai_result.get("generator", "")
            ).startswith("fallback_after_openai_error"):
                return openai_result
            return result

    if provider in {"auto", "openai"}:
        result = _draft_with_openai(payload)
        if result is not None:
            return result

    return _fallback_response(payload)
