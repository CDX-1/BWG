# PARALLAX

**Preserve Anomalies, Retain Alternate Explanations**

> Existing systems transmit what they understand. PARALLAX preserves the evidence they do not.

PARALLAX is an onboard evidence-preservation assistant for deep-space missions. When an anomaly-detection system encounters an event it cannot confidently explain, PARALLAX uses Gemma to maintain competing hypotheses and determine which raw records must survive a limited transmission window — so investigators on Earth can still distinguish between explanations after the data is sent.

---

## The Problem

Deep-space spacecraft operate with:
- Communication delays of 38–90 minutes
- Transmission budgets of 25–50 KB per anomaly report
- Autonomous compression that permanently discards data before mission control can review it

A conventional system flags that something unusual happened. It cannot determine whether the event was a genuine scientific observation, an instrument fault, or corrupted data. A standard compression policy selects the smallest files — often discarding the primary instrument record in favour of thumbnail images.

**Information lost during transmission cannot be recovered.**

---

## The Solution

PARALLAX adds a reasoning layer after conventional anomaly detection:

1. **Detect** — Deterministic threshold triggers flag the anomaly
2. **Retrieve** — TF-IDF search pulls relevant passages from instrument manuals, incident procedures, and known event descriptions
3. **Reason** — Gemma generates competing hypotheses, rates each piece of evidence by preservation priority, and warns what compression would destroy
4. **Preserve** — A 0/1 knapsack algorithm selects the highest-value evidence combination that fits within the byte budget
5. **Compare** — The dashboard shows what a naive compression baseline loses vs. what PARALLAX keeps

---

## Demo Scenarios

| Scenario | Event | Key Finding |
|---|---|---|
| Spectrometer Transient — Europa Flyby | 8.7σ spectral spike at T+04:17:32 | Naive compression drops `raw_spectrum` (14 KB primary data) in favour of a camera thumbnail |
| Solar Array Power Drop — Jupiter Approach | 35% output loss at T+09:42:15 | Naive compression drops `solar_iv_curves` (segment-level panel data) in favour of star tracker frames |

In both cases, PARALLAX preserves the primary instrument record. The naive baseline loses it.

---

## Architecture

```
Synthetic Mission Simulator
        ↓
Deterministic Anomaly Detector
        ↓
TF-IDF Context Retriever (5 knowledge documents)
        ↓
Gemma Reasoning Layer (hosted inference)
  → Competing hypotheses
  → Evidence preservation priorities
  → Compression warnings
        ↓
Pydantic Validation
        ↓
0/1 Knapsack Evidence-Capsule Builder
        ↓
Mission-Control Dashboard (Streamlit)
```

**Gemma handles:** hypothesis generation, evidence ranking, uncertainty preservation, compression warnings

**Deterministic Python handles:** anomaly detection, byte-budget arithmetic, evidence selection, schema validation

---

## Setup

```bash
git clone <repo>
cd BWG
pip install -r requirements.txt
cp .env.example .env
# Edit .env if your endpoint requires an API key
streamlit run parallax/app.py
```

Without an API key or live endpoint, the app falls back to pre-generated cached analyses automatically.

---

## Environment Variables

```
GEMMA_ENDPOINT=https://ai.spuric.com/v1/chat/completions
API_KEY=                        # optional Bearer token
USE_LIVE_GEMMA=true
MODEL_NAME=gemma-3-27b-it
```

---

## Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| Model | Gemma via hosted OpenAI-compatible endpoint |
| Retrieval | TF-IDF (scikit-learn) |
| Validation | Pydantic v2 |
| Charts | Plotly |
| Data | Pandas / NumPy |

---

## Hackathon Track

**Track 2: Trajectory & Orbit (Deep Space Navigation)**

PARALLAX demonstrates that an LLM reasoning layer can meaningfully improve evidence preservation decisions under bandwidth constraints — without replacing deterministic safety systems or performing unreliable numerical calculations.
