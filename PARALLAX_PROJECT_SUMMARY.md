# PARALLAX

## Preserve Anomalies, Retain Alternate Explanations

PARALLAX is an onboard evidence-preservation assistant for deep-space missions. It activates when existing anomaly-detection systems encounter an event they cannot confidently explain.

Instead of forcing a diagnosis, PARALLAX uses Gemma to preserve uncertainty, maintain competing hypotheses, determine which records matter most, and construct a bandwidth-limited evidence capsule for transmission to Earth.

> Existing systems transmit what they understand. PARALLAX preserves the evidence they do not.

---

## 1. Project Goal

Deep-space spacecraft often operate with:

- Long communication delays
- Limited transmission bandwidth
- Limited onboard storage
- Autonomous filtering and compression
- Incomplete or ambiguous observations

An unexplained event may be:

- A genuine scientific observation
- A hardware fault
- A calibration error
- A software issue
- A packet corruption event
- An environmental disturbance

A conventional system may compress, average, summarize, or discard data before mission control can investigate it. PARALLAX prevents irreversible information loss.

The system answers:

> Given a strict transmission budget, which evidence must be preserved so investigators on Earth can still distinguish between competing explanations?

---

## 2. Hackathon Positioning

PARALLAX is not intended to replace:

- Flight-qualified fault-detection systems
- Numerical anomaly detectors
- Command validators
- Orbital mechanics software
- Deterministic spacecraft control systems

Instead, it adds a reasoning layer after conventional anomaly detection.

The deterministic detector identifies that something unusual occurred. Gemma then reasons across telemetry, logs, operating procedures, system context, and available records to determine what information must be preserved.

This makes Gemma central without asking it to perform unreliable low-level numerical control or safety-critical calculations.

---

## 3. Core Demo Scenario

### Mission Context

- Spacecraft: Asteria-7
- Mission phase: Europa flyby science pass
- Earth communication delay: 43 minutes
- Available transmission budget: 25 KB
- Event: Unclassified spectrometer transient

### Event Details

At mission time `T+04:17:32`:

- Spectrometer output spikes for approximately 1.8 seconds
- Radiation sensor readings also rise
- Instrument temperature changes slightly
- Packet checksum errors occur near the event
- Power telemetry remains mostly nominal
- The spacecraft cannot transmit all available evidence

### Plausible Hypotheses

1. Genuine external radiation event
2. Spectrometer electronics fault
3. Packet corruption or data-transmission error

No single hypothesis completely explains the evidence.

### Evidence Candidates

| Evidence | Approximate Size | Why It Matters |
|---|---:|---|
| Raw spectrum | 14 KB | Preserves temporal and spectral shape |
| Radiation window | 5 KB | Tests external-event hypothesis |
| Thermal telemetry | 4 KB | Tests electronics-fault hypothesis |
| Packet integrity log | 2.5 KB | Tests corruption hypothesis |
| Context image | 8 KB | Adds environmental context |
| Power telemetry | 3.5 KB | Tests power-related instability |

PARALLAX must select a subset that fits inside the 25 KB transmission budget.

---

## 4. User Experience

The live demo should follow this sequence:

1. The dashboard displays normal spacecraft telemetry.
2. The anomaly occurs.
3. A deterministic detector flags the event.
4. Relevant operating procedures and manuals are retrieved.
5. Gemma generates competing hypotheses.
6. Gemma identifies evidence supporting and contradicting each hypothesis.
7. Gemma ranks available evidence by preservation importance.
8. Deterministic code constructs the evidence capsule within the byte budget.
9. The dashboard compares PARALLAX with a naive compression baseline.
10. Mission control sees which explanations remain testable after transmission.

---

## 5. Recommended Technology Stack

### Primary Stack

| Layer | Technology |
|---|---|
| User interface | Streamlit |
| Application logic | Python |
| Model | Gemma through hosted inference |
| Data processing | Pandas and NumPy |
| Charts | Plotly |
| Output validation | Pydantic |
| Retrieval | TF-IDF with scikit-learn |
| Scenario storage | JSON |
| Knowledge storage | Markdown |
| Deployment | Local machine or Streamlit Community Cloud |
| Version control | GitHub |

### Why Streamlit

Streamlit is preferred because it provides:

- Fast dashboard development
- Charts and progress bars
- Buttons and session state
- JSON and structured-data rendering
- Minimal frontend/backend integration work
- A single-process Python application

Do not use React unless a team member can independently own the frontend and already has a working dashboard template.

### Model Access

Preferred approach:

- Use a hosted Gemma endpoint available through the hackathon environment or Google tooling.
- Wrap all model access behind one Python function.
- Cache successful responses.
- Support a transparent offline fallback for the live demo.

Do not fine-tune during the hackathon.

---

## 6. System Architecture

```text
Synthetic Mission Simulator
        |
        |-- Telemetry time series
        |-- Instrument observations
        |-- Command history
        |-- Spacecraft operating state
        |
        v
Deterministic Anomaly Detector
        |
        v
Context Retriever
        |
        |-- Instrument manual
        |-- Known event descriptions
        |-- Packet-integrity rules
        |-- Compression risks
        |-- Incident-response procedures
        |
        v
Gemma Reasoning Layer
        |
        |-- Competing hypotheses
        |-- Supporting evidence
        |-- Contradicting evidence
        |-- Missing evidence
        |-- Preservation priorities
        |-- Recommended follow-up
        |
        v
Pydantic Validation
        |
        v
Deterministic Evidence-Capsule Builder
        |
        |-- Selected evidence
        |-- Rejected evidence
        |-- Byte-budget calculation
        |-- Hypothesis coverage
        |
        v
Mission-Control Dashboard
```

---

## 7. Responsibility Boundaries

### Deterministic Components

Use normal Python logic for:

- Telemetry generation
- Numerical anomaly triggers
- File sizes
- Byte-budget calculations
- Evidence selection
- Schema validation
- Retry logic
- Baseline compression
- Chart generation

### Gemma Responsibilities

Use Gemma for:

- Interpreting heterogeneous evidence
- Generating plausible competing explanations
- Linking evidence to hypotheses
- Identifying contradictions
- Determining which records are critical
- Warning about information destroyed by compression
- Producing a human-readable incident handoff
- Preserving uncertainty

Gemma should not:

- Perform orbital mechanics
- Directly control the spacecraft
- Calculate exact storage budgets
- Declare an ambiguous event as confirmed
- Invent missing telemetry
- Replace deterministic safety checks

---

## 8. Suggested Repository Structure

```text
parallax/
├── app.py
├── config.py
├── gemma.py
├── retrieval.py
├── simulator.py
├── detector.py
├── capsule.py
├── baseline.py
├── models.py
├── utils.py
├── requirements.txt
├── README.md
├── scenarios/
│   └── spectrometer_001.json
├── knowledge/
│   ├── spectrometer_manual.md
│   ├── radiation_events.md
│   ├── packet_integrity.md
│   ├── compression_risks.md
│   └── incident_response.md
├── cached_outputs/
│   └── spectrometer_001.json
└── assets/
    └── architecture.png
```

---

## 9. Scenario Data Contract

Example scenario file:

```json
{
  "scenario_id": "spectrometer_001",
  "mission": {
    "spacecraft": "Asteria-7",
    "phase": "Europa flyby science pass",
    "earth_delay_minutes": 43,
    "transmission_budget_bytes": 25000
  },
  "event": {
    "timestamp": "T+04:17:32",
    "trigger": "spectrometer transient",
    "description": "Unclassified 1.8-second spectral transient"
  },
  "telemetry_summary": {
    "spectrometer_peak_sigma": 8.7,
    "radiation_counter_change_percent": 41,
    "instrument_temperature_change_c": 0.8,
    "bus_voltage_change_percent": 0.3,
    "checksum_errors": 2
  },
  "recent_commands": [
    "04:14:00 START_SCIENCE_CAPTURE",
    "04:16:50 SET_COMPRESSION_MODE HIGH"
  ],
  "available_evidence": [
    {
      "id": "raw_spectrum",
      "size_bytes": 14000,
      "description": "Raw spectral frames from T-3s to T+3s"
    },
    {
      "id": "radiation_window",
      "size_bytes": 5000,
      "description": "Radiation counter data from T-30s to T+30s"
    },
    {
      "id": "thermal_window",
      "size_bytes": 4000,
      "description": "Instrument thermal telemetry"
    },
    {
      "id": "packet_checksums",
      "size_bytes": 2500,
      "description": "Packet integrity and sequencing log"
    },
    {
      "id": "camera_thumbnail",
      "size_bytes": 8000,
      "description": "Context camera thumbnail"
    },
    {
      "id": "power_window",
      "size_bytes": 3500,
      "description": "Power subsystem telemetry"
    }
  ]
}
```

---

## 10. Gemma Output Schema

Use Pydantic to validate all model responses.

```python
from typing import Literal
from pydantic import BaseModel, Field


class EvidenceAssessment(BaseModel):
    evidence_id: str
    priority: int = Field(ge=1, le=5)
    reason: str
    supports_hypotheses: list[str]
    consequence_if_lost: str


class Hypothesis(BaseModel):
    name: str
    confidence: Literal["low", "medium", "high"]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    evidence_needed: list[str]


class ParallaxAnalysis(BaseModel):
    event_summary: str
    resolution_status: Literal[
        "known_event",
        "likely_fault",
        "likely_scientific_event",
        "unresolved"
    ]
    hypotheses: list[Hypothesis]
    preservation_priorities: list[EvidenceAssessment]
    recommended_follow_up: str
    compression_warning: str
    uncertainty_statement: str
```

Optional additions:

```python
class SourceReference(BaseModel):
    source_id: str
    relevance: str


class ParallaxAnalysis(BaseModel):
    ...
    source_references: list[SourceReference]
```

---

## 11. Prompt Template

```text
You are PARALLAX, an evidence-preservation assistant for a deep-space
spacecraft operating under communication delay.

Your role is NOT to conclusively diagnose an unexplained event.

Your role is to:
1. Generate plausible competing explanations.
2. State evidence supporting and contradicting each explanation.
3. Identify which available records must be preserved so investigators
   can distinguish between the explanations later.
4. Warn when compression, averaging, interpolation, or summarization
   could destroy relevant evidence.
5. Recommend one safe follow-up observation.
6. Preserve uncertainty explicitly.

Rules:
- Do not invent telemetry.
- Refer only to evidence included in the input or retrieved context.
- Never report an ambiguous event as confirmed.
- Prefer raw observations when derived summaries would remove temporal,
  spectral, sequencing, or cross-sensor information.
- Consider hardware fault, environmental event, software error,
  calibration error, and data corruption when supported.
- The total transmission capacity is limited.
- Do not perform byte-budget arithmetic.
- Return output matching the required JSON structure.
```

Append to the prompt:

- Scenario JSON
- Retrieved knowledge passages
- Available evidence list
- Required output schema

---

## 12. Retrieval-Augmented Generation

Do not install a vector database.

Create a small knowledge base:

```text
knowledge/
├── spectrometer_manual.md
├── radiation_events.md
├── packet_integrity.md
├── compression_risks.md
└── incident_response.md
```

Each file should contain short, clearly labeled sections.

Example:

```text
[SPECTROMETER-MANUAL §3.2]
Short transient signals must not be temporally averaged before review.
Averaging may remove asymmetry, peak structure, or secondary pulses.

[PACKET-INTEGRITY §2.4]
Checksum errors occurring near an anomalous event require preservation
of packet sequence numbers, duplication flags, and raw payload boundaries.
```

Use TF-IDF retrieval:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query: str, chunks: list[str], top_k: int = 4) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in indices]
```

The UI should display the retrieved passages in a collapsible panel.

---

## 13. Evidence-Capsule Builder

Gemma assigns each item:

- Priority from 1 to 5
- Supported hypotheses
- Consequence if lost
- Human-readable reason

Deterministic code performs the final selection.

### Basic Scoring

```python
def evidence_score(item: dict, assessment: EvidenceAssessment) -> float:
    hypothesis_coverage = len(set(assessment.supports_hypotheses))

    return (
        assessment.priority * 10
        + hypothesis_coverage * 8
    ) / max(item["size_bytes"], 1)
```

### Greedy Selection

```python
def build_capsule(
    available_evidence: list[dict],
    assessments: list[EvidenceAssessment],
    budget: int,
) -> list[dict]:
    assessment_map = {
        assessment.evidence_id: assessment
        for assessment in assessments
    }

    ranked = sorted(
        available_evidence,
        key=lambda item: evidence_score(
            item,
            assessment_map[item["id"]],
        ),
        reverse=True,
    )

    selected = []
    used = 0

    for item in ranked:
        if used + item["size_bytes"] <= budget:
            selected.append(item)
            used += item["size_bytes"]

    return selected
```

A 0/1 knapsack implementation can replace the greedy algorithm later, but it is not required for the MVP.

---

## 14. Baseline Comparison

PARALLAX needs a visible comparison against a simple non-reasoning baseline.

Possible baseline policies:

- Select the smallest files first
- Prefer pre-compressed summaries
- Preserve only the primary instrument stream
- Preserve records with the highest anomaly score
- Discard contextual records

The baseline should be reasonable, not intentionally broken.

Example:

```python
def naive_capsule(items: list[dict], budget: int) -> list[dict]:
    ranked = sorted(items, key=lambda item: item["size_bytes"])

    selected = []
    used = 0

    for item in ranked:
        if used + item["size_bytes"] <= budget:
            selected.append(item)
            used += item["size_bytes"]

    return selected
```

The dashboard should explain which hypotheses remain investigable after each policy.

---

## 15. Telemetry Simulation

Use deterministic synthetic data.

```python
import numpy as np
import pandas as pd


def generate_spectrometer_scenario(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, 120)

    spectrum = rng.normal(10, 0.4, len(t))
    radiation = rng.normal(30, 1.2, len(t))
    temperature = rng.normal(18, 0.05, len(t))
    checksum_errors = np.zeros(len(t))

    spectrum[60:63] += [4, 9, 3]
    radiation[59:64] += [2, 8, 13, 7, 2]
    temperature[61:68] += np.linspace(0.1, 0.8, 7)
    checksum_errors[62] = 2

    return pd.DataFrame({
        "time": t,
        "spectrum": spectrum,
        "radiation": radiation,
        "temperature": temperature,
        "checksum_errors": checksum_errors,
    })
```

The anomaly detector can be simple and deterministic:

```python
def detect_event(df: pd.DataFrame) -> bool:
    spectrum_alert = df["spectrum"].max() > 15
    radiation_alert = df["radiation"].max() > 38
    checksum_alert = df["checksum_errors"].sum() > 0

    return spectrum_alert and (radiation_alert or checksum_alert)
```

---

## 16. Dashboard Layout

### Header

```text
PARALLAX
Evidence-preserving autonomy for delayed deep-space missions

Mission: Asteria-7
Earth delay: 43 min
Transmission window: 25 KB
Status: UNRESOLVED EVENT
```

### Left Column: Telemetry

Display:

- Spectrometer intensity
- Radiation count
- Instrument temperature
- Checksum errors

Mark the anomaly timestamp.

### Middle Column: Gemma Analysis

Show:

- Event summary
- Resolution status
- Hypothesis cards
- Supporting evidence
- Contradicting evidence
- Missing evidence
- Uncertainty statement
- Recommended follow-up

### Right Column: Evidence Capsule

Show:

- Selected evidence
- Rejected evidence
- File sizes
- Total budget used
- Number of hypotheses still testable
- Why each selected item matters

### Bottom Panel: Comparison

Compare:

#### Naive Compression

- What it preserves
- What it loses
- Which hypotheses can no longer be tested

#### PARALLAX

- What it preserves
- Why it was selected
- Which hypotheses remain testable

Suggested progress indicator:

```text
1 Detect ✓
2 Retrieve ✓
3 Reason ✓
4 Preserve ✓
5 Transmit ✓
```

---

## 17. Model Wrapper and Fallback

All model access should pass through one function:

```python
def run_parallax_analysis(
    scenario: dict,
    retrieved_context: list[str],
) -> ParallaxAnalysis:
    ...
```

Recommended behavior:

1. Call hosted Gemma
2. Parse structured JSON
3. Validate with Pydantic
4. Retry once if parsing fails
5. Save successful output to `cached_outputs/`
6. Fall back to cached output on failure

Example:

```python
try:
    analysis = call_gemma(scenario, retrieved_context)
except Exception:
    analysis = load_cached_result(scenario["scenario_id"])
    st.warning(
        "Demo fallback active: displaying a previously generated Gemma analysis."
    )
```

Do not hide fallback mode.

---

## 18. Minimum Viable Product

The project is demo-ready when it can:

1. Load the main scenario
2. Generate and display telemetry charts
3. Detect the anomaly
4. Retrieve relevant knowledge passages
5. Call Gemma
6. Return three validated hypotheses
7. Rank available evidence
8. Build a capsule inside the byte budget
9. Compare against a baseline
10. Explain what information would otherwise be lost

Do not add extra features before all ten steps work.

---

## 19. Seven-Hour Build Plan

### Hour 0:00–0:30 — Lock Scope

Complete:

- One-sentence pitch
- Repository setup
- Main scenario
- Output schema
- UI sketch
- Team responsibilities

### Hour 0:30–1:30 — Scenario and Simulator

Build:

- Synthetic telemetry
- Anomaly timestamp
- Scenario JSON
- Evidence list
- Charts

Goal: telemetry is visible and reproducible.

### Hour 1:30–2:30 — Gemma Integration

Build:

- Model wrapper
- Prompt
- Structured output
- Pydantic validation
- Retry logic
- Cached fallback

Goal: one successful end-to-end analysis.

### Hour 2:30–3:15 — Retrieval

Build:

- Five knowledge files
- Chunking
- TF-IDF retrieval
- Source display

Goal: Gemma analysis visibly uses retrieved context.

### Hour 3:15–4:00 — Capsule Logic

Build:

- Evidence scoring
- Byte-budget selection
- Baseline policy
- Hypothesis-coverage comparison

Goal: selected and rejected records are displayed.

### Hour 4:00–5:00 — Dashboard Integration

Connect:

- Telemetry
- Detection
- Retrieval
- Gemma
- Capsule builder
- Baseline
- Status indicators

Goal: one-button demo flow.

### Hour 5:00–5:45 — Reliability

Test:

- Missing API key
- Invalid JSON
- Model timeout
- Empty retrieval
- Tiny byte budget
- Streamlit rerun behavior
- Wi-Fi failure
- Machine restart

Goal: demo cannot catastrophically fail.

### Hour 5:45–6:20 — Visual Polish

Improve only:

- Status hierarchy
- Typography
- Card layout
- Byte progress bar
- Comparison panel
- Reduced text
- Consistent terminology

Do not add major features.

### Hour 6:20–7:00 — Presentation

Prepare:

- 90-second demo recording
- Three-minute pitch
- Architecture diagram
- Kaggle writeup
- README
- Backup screen recording

Record the demo before making final risky code changes.

---

## 20. Team Split

### Two-Person Team

#### Person A: AI and Backend

- Gemma integration
- Prompting
- Retrieval
- Pydantic models
- Capsule selection
- Error handling

#### Person B: Product and Demo

- Simulator
- Streamlit interface
- Charts
- Baseline comparison
- Pitch
- Writeup
- Recording

### Three-Person Team

#### Person A

- Gemma
- Prompting
- Structured output
- Retrieval

#### Person B

- Simulator
- Detector
- Capsule algorithm
- Validation

#### Person C

- Streamlit interface
- Visual design
- Demo script
- Kaggle writeup
- Video

The UI should initially use hard-coded placeholder data so frontend work can proceed before Gemma integration is complete.

---

## 21. What Not to Build

Do not spend time on:

- Fine-tuning
- Multiple AI agents
- Full orbital simulation
- Real spacecraft control
- Real-time NASA telemetry ingestion
- Computer vision
- Authentication
- User accounts
- Databases
- WebSockets
- Mobile applications
- Ten anomaly scenarios
- 3D solar-system visualization
- Fully realistic spacecraft physics

One polished scenario is better than several incomplete ones.

---

## 22. Presentation Narrative

### Opening

Asteria-7 is exploring Europa, 43 light-minutes from Earth. During a science pass, it records an unexplained spectrometer transient.

The spacecraft has only a 25 KB transmission window.

### Problem

A conventional anomaly detector can flag the event, but it cannot determine whether the event is:

- A scientific observation
- An instrument failure
- Corrupted data

A standard compression policy may permanently remove the evidence needed to find out.

### Demonstration

PARALLAX:

1. Retrieves relevant procedures
2. Uses Gemma to maintain competing explanations
3. Identifies evidence supporting and contradicting each explanation
4. Determines which raw records must survive compression
5. Creates a byte-limited evidence capsule
6. Preserves all three explanations for later investigation

### Final Line

> Existing systems transmit what they understand. PARALLAX preserves the evidence they do not.

---

## 23. Gemma Integration Justification

Without Gemma, the system can:

- Detect a statistical anomaly
- Calculate file sizes
- Select small records
- Plot telemetry

Without Gemma, it cannot reliably:

- Interpret heterogeneous context
- Maintain competing hypotheses
- Connect evidence to explanations
- Identify contradictions across manuals and logs
- Explain what compression would irreversibly destroy
- Produce an uncertainty-preserving incident handoff

Gemma is therefore the central reasoning component, while deterministic systems remain responsible for calculations and validation.

---

## 24. Suggested Requirements

```text
streamlit
pandas
numpy
plotly
pydantic
scikit-learn
python-dotenv
google-genai
```

Pin versions after confirming compatibility on the development machine.

---

## 25. Environment Variables

Example `.env`:

```text
GOOGLE_API_KEY=replace_with_hackathon_key
USE_LIVE_GEMMA=true
MODEL_NAME=replace_with_available_gemma_model
```

Do not commit `.env`.

Add to `.gitignore`:

```text
.env
__pycache__/
*.pyc
.streamlit/secrets.toml
```

---

## 26. First Development Tasks

Start with these tasks in order:

1. Create the repository structure
2. Add `spectrometer_001.json`
3. Implement telemetry generation
4. Display four Plotly charts in Streamlit
5. Add the deterministic event trigger
6. Create Pydantic output models
7. Implement the Gemma wrapper
8. Add cached fallback output
9. Create five knowledge documents
10. Implement TF-IDF retrieval
11. Implement capsule selection
12. Implement baseline comparison
13. Connect the complete UI flow
14. Test failure modes
15. Record the demo

---

## 27. Definition of Done

PARALLAX is complete when a judge can watch the demo and immediately understand:

- Why the event is ambiguous
- Why existing anomaly detection is insufficient
- Why limited bandwidth creates an irreversible risk
- What Gemma contributes
- Which evidence PARALLAX preserves
- What the baseline loses
- Why the system does not overclaim certainty
- How the prototype could extend to real mission operations

The final prototype should prioritize reliability, clarity, and a strong live demonstration over broad functionality.
