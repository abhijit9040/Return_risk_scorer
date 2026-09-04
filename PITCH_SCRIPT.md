# 5-minute pitch script — Return-Risk Scorer & Auto-Responder

**Track:** Razorpay AI Buildathon — AI Risk Manager  
**Length:** 5:00  
**Repo:** https://github.com/abhijit9040/Return_risk_scorer  

---

## Before you hit record (2 minutes)

1. Open Streamlit: `.\.venv\Scripts\streamlit.exe run app\streamlit_app.py` → `http://localhost:8501`
2. Leave **Score a return** tab open. Zoom browser to ~125%.
3. Have **Holdout metrics**, **Policy**, and **Audit trail** ready.
4. Hide `.env` / terminal secrets. Do not scroll past API keys.
5. Speak at a calm pace. If you freeze, skip a sentence — do not restart unless you must.

**On-screen plan**

| Time | Screen |
|---|---|
| 0:00–0:45 | Face / title, then Streamlit home |
| 0:45–1:30 | Architecture (README or Policy tab) |
| 1:30–3:45 | Live demo — 3 scores |
| 3:45–4:25 | Holdout metrics tab |
| 4:25–4:50 | Failure case (checkbox) + audit |
| 4:50–5:00 | Close |

---

## Script (read this)

### 0:00–0:45 — Problem

Hi, I’m [Your Name]. This is **Return-Risk Scorer & Auto-Responder**, built for the Razorpay AI Risk Manager track.

Merchants lose margin to return abuse: **wardrobing**, **policy abuse**, and **fraudulent returns**. Reviewing every return does not scale. Blunt rules like “block after N returns” punish good customers and create churn.

The brief asked for a working **detector, verifier, or auto-responder**, with **honest precision and recall**, including **false-positive cost**, and **strictly defense-only**. We built all three, chained together, plus an audit trail.

---

### 0:45–1:30 — Architecture *(show Policy tab or README diagram)*

**[Click Policy tab]**

Four stages.

**Detector** — LightGBM multi-class model. Four labels: Legitimate, Policy Abuser, Wardrobing, Fraudulent Return. SHAP gives the top three reason codes — not a black-box score.

**Verifier** — a policy engine. It does **not** auto-deny. Actions are only **auto-approve**, **hold for evidence**, or **escalate to a human**.

Costs scale with **refund amount**. Missing a $488 fraud is not the same as missing a $73 case. FN is treated as roughly the refund at stake. FP hold and escalate are a small rate times that refund.

**Auto-responder** — Gemini drafts an internal case note, and a customer message only when we hold or escalate.

**Audit** — every decision is logged: input, score, action, rationale, timestamp.

---

### 1:30–2:20 — Demo A: escalate *(Score a return)*

**[Score a return tab. Set holdout row index to a known high-risk example if you have one; otherwise pick any index and Score.]**

Live scoring. One return in.

Watch four things: **predicted class**, **risk score**, **action**, and **candidate costs**.

On a high-confidence **fraudulent return**, the action is **escalate to human** — never a silent deny. SHAP typically points at things like days-to-return, wishlist-to-cart time, refund-to-AOV.

The customer message is polite: a specialist is reviewing. A human can still reverse this.

If this refund is around **$488**, auto-approve expected cost is about **$488**. Hold is much smaller. Escalate on certain fraud is cheap on the legit-customer side because we are not wrongly holding a good customer. The dollars follow the refund, not a flat constant.

---

### 2:20–3:10 — Demo B: hold for evidence

**[Change the row index. Score again. Aim for Policy Abuser / Wardrobing — or narrate from a previous run if the next row is legit.]**

Second case: **policy abuser** or **wardrobing** — high return rate, evidence can still resolve it.

Action: **hold for evidence**. Gemini asks for photos and packaging. That is cheaper than a full specialist queue when the pattern is “show proof,” not “this looks like empty-box fraud.”

If the refund is only about **$73**, the candidate costs shrink with it. Same policy, different dollars at stake. That is the verifier a judge can read in the code.

---

### 3:10–3:45 — Demo C: auto-approve *(optional if time)*

**[Pick a low-risk / Legitimate row. Score.]**

Low risk: **auto-approve**. Fast path. No customer interrogation. That is how we avoid punishing legitimate shoppers.

---

### 3:45–4:25 — Metrics *(Holdout metrics tab)*

**[Holdout metrics]**

Held-out test split. Never used to train.

Macro F1 is about **0.999**. Per-class F1 is similarly high. **I want to be honest:** this Kaggle set is highly separable. We are **not** claiming 99.9% on messy live Razorpay traffic. Differentiation is the **policy, refund-scaled cost, audit, and the failure path** — not a magic accuracy number.

On an 80-row end-to-end batch we saw roughly **51 auto-approve, 22 hold, 7 escalate**. Confusion matrix is on screen. False-positive cost is explicit: rate times capped refund. On this clean holdout, FP and FN counts were zero — we still show the formula so a judge can audit it.

---

### 4:25–4:50 — Failure case *(checkbox + Audit trail)*

The brief asked for **one failure handled gracefully**.

**[Score a return → check “Force insufficient_signal” → Score]**

Blank category, returns exceeding orders. The model **does not guess**. Action is **insufficient_signal**. Routed to manual review. Gemini writes a case note; customer message is null — we do not invent a story.

**[Audit trail tab]**

Every click is in the log. Input, action, rationale, timestamp. Defensible if a customer disputes.

Defense-only: no account bans, no payment blocks, no auto-deny. A human can reverse every automated action.

---

### 4:50–5:00 — Close

Public GitHub, pinned requirements, one command to train and evaluate.

Next: calibrate on real merchant returns, and tune hold vs escalate with live refund mix.

Return-Risk Scorer: score, explain, bounded action, full audit. Thank you.

---

## Backup lines (if a score is slow or Gemini lags)

- “While this loads: the same pipeline writes JSONL and SQLite, so the demo is the production path, not a fake UI.”
- If Gemini fails: “Template fallback still produces a case note — the decision never depends on the LLM.”

## Do not say

- That we auto-block customers or Razorpay payments  
- That 0.999 F1 will hold in production without caveats  
- Any API keys or `.env` contents  

## Click checklist while recording

1. Policy tab — 20 seconds  
2. Score (fraud / escalate)  
3. Score (hold)  
4. Score (approve) if time  
5. Metrics + confusion matrix  
6. Force insufficient_signal + Audit trail  
7. End on title / GitHub URL on screen if you have it  
