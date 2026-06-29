# Provenance Guard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask API that classifies text content as human or AI-written, returns a confidence score and plain-language transparency label, and handles creator appeals with a full structured audit trail.

**Architecture:** Two independent detection signals (LLM semantic + stylometric structural) run in parallel, combine via a weighted confidence scorer, and produce a transparency label. Every decision and appeal is logged to SQLite.

**Tech Stack:** Flask, Flask-Limiter, Groq Python SDK (`llama-3.3-70b-versatile`), SQLite (built-in), pure Python for stylometrics.

---

## Milestone 1: Architecture

### Architecture Narrative

A piece of text submitted to `POST /submit` flows through the system as follows:

**1. Rate Limiter** — checks the submitter's IP against a sliding window. Returns HTTP 429 if the limit is exceeded, preventing API abuse and Groq quota exhaustion.

**2. Request Validator** — ensures `content` and `creator_id` are present, content is non-empty, and content is within the size limit (10,000 characters, minimum 10 words). Returns HTTP 400 on failure.

**3. Detection Pipeline** — runs two independent signals:

- **LLM Classifier** (Groq API, `llama-3.3-70b-versatile`): sends the text with a structured prompt asking for an AI probability score 0–1. Captures holistic semantic and stylistic coherence — whether the text "reads" as AI-generated, including sentence flow, phrasing naturalness, and presence of a personal voice.
- **Stylometric Analyzer** (pure Python, no external libraries): computes sentence length variance, type-token ratio, and expressive punctuation density. Returns a structural AI-likelihood score 0–1. Captures statistical patterns invisible to semantic models.

**4. Confidence Scorer** — combines the two signal scores with a weighted average (LLM 60%, stylometric 40%). Applies asymmetric thresholds biased against false positives: combined score must exceed **0.70** to label AI-generated, but need only fall below **0.35** to label human-written. The 0.35–0.70 range is the "uncertain" zone — when in doubt, the system defaults toward the less-damaging label.

**5. Transparency Label Generator** — converts the confidence score and attribution into plain-language text the platform displays to readers. Three variants (see below).

**6. Audit Logger** — writes a structured SQLite record: `content_id`, timestamp, all individual signal scores, combined score, label, and status.

**7. Response** — returns JSON with `content_id`, `attribution_result`, `confidence_score`, and `transparency_label`.

For appeals: `POST /appeal` → Request Validator → Appeal Handler (looks up original decision, validates creator, captures reasoning) → Audit Logger (appends appeal, updates status to `under_review`) → Response.

---

### Detection Signals

**Signal 1: LLM Classification (Groq)**

| Property | Detail |
|---|---|
| What it measures | Holistic semantic and stylistic coherence — whether the text "feels" AI-generated based on sentence flow, phrasing choices, structural patterns, and presence/absence of a personal voice |
| Why it differs | AI writing tends toward polished, coherent, well-structured text without rough edges. Human creative writing has natural imperfections, idiosyncratic choices, personal references, and stylistic quirks |
| Blind spots | Can be fooled by highly polished human writers (editors, academics). Cannot detect AI writing that deliberately introduces errors. Subject to the LLM's own biases about what "AI writing" looks like. Non-deterministic — same text may return slightly different scores across calls. Cannot assess non-linguistic properties |

**Signal 2: Stylometric Heuristics (Pure Python)**

| Property | Detail |
|---|---|
| What it measures | Three structural statistics: (1) sentence length variance — std dev of word counts per sentence; (2) type-token ratio — unique words / total words = vocabulary diversity; (3) expressive punctuation density — frequency of `! ? … — ; :` relative to total characters |
| Why it differs | AI text tends toward uniform sentence lengths and moderate vocabulary diversity. Human creative writing varies more: sentence fragments alongside complex constructions, and expressive punctuation is more common in personal voice |
| Blind spots | Cannot capture meaning or narrative quality. Genre skews results (poetry vs. blog post). A concise human writer may score similarly to AI on length variance. Fails on very short texts (< 10 words → returns 0.5 = uncertain). Does not adapt to code or structured lists embedded in prose |

These two signals are genuinely independent: one is **semantic** (what the words mean and how they fit together), the other is **structural** (measurable statistics about word and punctuation distribution). Combining them is more informative than either alone.

---

### False Positive Analysis

**Scenario:** A human writer submits a polished, professionally edited essay. Both signals return elevated but not conclusive scores — say LLM=0.55, stylometric=0.48. Combined: (0.6 × 0.55) + (0.4 × 0.48) = 0.33 + 0.19 = 0.52.

**What happens:**
- 0.52 falls in the uncertain zone (0.35–0.70) → label is **"Authorship Uncertain"**, not "Likely AI-Generated"
- The label explicitly says: *"This content has not been flagged as AI-generated"* and invites an appeal
- If the LLM had returned 0.75 instead, combined would be (0.6×0.75)+(0.4×0.48)=0.45+0.19=0.64 — still in the uncertain zone
- To reach the AI label threshold of 0.70, the LLM would need to return ~0.90+ while the stylometric score remains at 0.48

This asymmetry is deliberate. A false positive (human labeled as AI) on a creative writing platform damages reputation; a false negative (AI labeled as uncertain) is far less harmful. The threshold gap forces confidence before we make the more damaging call.

If the writer disagrees with any label: they `POST /appeal`, provide reasoning, and the system immediately marks the content `under_review`. No automated re-classification occurs — that's a human decision.

---

### API Surface

| Endpoint | Method | Request Body | Success Response |
|---|---|---|---|
| `/submit` | POST | `{content, creator_id}` | `{content_id, attribution_result, confidence_score, transparency_label}` |
| `/appeal` | POST | `{content_id, creator_id, reason}` | `{appeal_id, content_id, status, message}` |
| `/log` | GET | — | `{entries: [...]}` |
| `/health` | GET | — | `{status, version}` |

**POST /submit — example:**
```json
Request:
{
  "content": "The starling flew past the window at dusk...",
  "creator_id": "user_abc123"
}

Response 200:
{
  "content_id": "cnt_4a9f2e1b8c3d",
  "attribution_result": "human",
  "confidence_score": 0.23,
  "transparency_label": {
    "verdict": "Likely Human-Written",
    "confidence_display": "High",
    "detail": "Our system analyzed this content and found strong indicators of human authorship. This is an automated assessment and is not guaranteed to be correct. If you are the creator and believe this label is incorrect, you may submit an appeal."
  }
}

Response 429:
{"error": "Rate limit exceeded. Try again in 1 hour."}
```

**POST /appeal — example:**
```json
Request:
{
  "content_id": "cnt_4a9f2e1b8c3d",
  "creator_id": "user_abc123",
  "reason": "I wrote this poem about my grandmother — it reflects a memory from my childhood."
}

Response 200:
{
  "appeal_id": "app_7b3e1f9d2c4a",
  "content_id": "cnt_4a9f2e1b8c3d",
  "status": "under_review",
  "message": "Your appeal has been received. The classification has been marked as under review."
}
```

**GET /log — example:**
```json
{
  "entries": [
    {
      "content_id": "cnt_4a9f2e1b8c3d",
      "creator_id": "user_abc123",
      "content_snippet": "The starling flew past the window at dusk...",
      "timestamp": "2026-06-29T14:32:00Z",
      "llm_score": 0.18,
      "stylometric_score": 0.31,
      "confidence_score": 0.23,
      "attribution": "human",
      "transparency_label": "Likely Human-Written",
      "status": "under_review",
      "appeals": [
        {
          "appeal_id": "app_7b3e1f9d2c4a",
          "creator_id": "user_abc123",
          "reason": "I wrote this poem about my grandmother...",
          "timestamp": "2026-06-29T14:45:00Z"
        }
      ]
    }
  ]
}
```

---

### Architecture Diagram

#### Submission Flow

```
POST /submit
{content, creator_id}
        |
        v
  ┌─────────────┐
  │ Rate Limiter │ ──────────────────► 429 Too Many Requests
  └──────┬──────┘
         |
         v
  ┌──────────────────┐
  │ Request Validator │ ─────────────► 400 Bad Request
  └────────┬─────────┘
           |
     ┌─────┴──────┐
     │             │
     v             v
┌──────────┐  ┌───────────────┐
│   LLM    │  │ Stylometric   │
│Classifier│  │  Analyzer     │
│(Groq API)│  │ (pure Python) │
│          │  │               │
│ llm_score│  │stylo_score    │
│  (0–1)   │  │  (0–1)        │
└────┬─────┘  └──────┬────────┘
     │                │
     │  llm_score     │  stylo_score
     └────────┬───────┘
              │
              v
    ┌──────────────────────┐
    │   Confidence Scorer   │
    │ 0.60×LLM + 0.40×stylo│
    │ → confidence_score    │
    │   (0–1)               │
    └──────────┬────────────┘
               │
               │  confidence_score + attribution
               v
    ┌──────────────────────┐
    │   Label Generator     │
    │ < 0.35 → human        │
    │ 0.35–0.70 → uncertain │
    │ > 0.70 → ai           │
    │ → transparency_label  │
    └──────────┬────────────┘
               │
         ┌─────┴──────┐
         │             │
         v             v
    ┌──────────┐  ┌─────────────────────────────┐
    │  Audit   │  │          Response             │
    │  Logger  │  │ {content_id,                  │
    │ (SQLite) │  │  attribution_result,          │
    │          │  │  confidence_score,            │
    └──────────┘  │  transparency_label}          │
                  └─────────────────────────────┘

Arrows between components carry:
  POST /submit body ──► Rate Limiter: {content, creator_id}
  Rate Limiter ──► Validator: {content, creator_id}
  Validator ──► LLM Classifier: raw text
  Validator ──► Stylometric Analyzer: raw text
  LLM Classifier ──► Confidence Scorer: llm_score (float 0–1)
  Stylometric Analyzer ──► Confidence Scorer: stylo_score (float 0–1)
  Confidence Scorer ──► Label Generator: confidence_score + attribution string
  Label Generator ──► Audit Logger: full decision record
  Label Generator ──► Response: transparency_label dict
```

#### Appeal Flow

```
POST /appeal
{content_id, creator_id, reason}
        |
        v
  ┌─────────────┐
  │ Rate Limiter │ ──────────────────► 429 Too Many Requests
  └──────┬──────┘
         |
         v
  ┌──────────────────┐
  │ Request Validator │ ─────────────► 400 Bad Request
  └────────┬─────────┘
           │
           v
  ┌──────────────────────┐
  │    Appeal Handler     │
  │  lookup original      │ ──────────► 404 content_id not found
  │  validate creator_id  │ ──────────► 400 creator mismatch
  │  validate reason      │ ──────────► 400 empty reason
  └──────────┬────────────┘
             │
       ┌─────┴──────┐
       │             │
       v             v
  ┌──────────┐  ┌────────────────────┐
  │  Audit   │  │   Status Updater   │
  │  Logger  │  │ status →           │
  │ (append  │  │   "under_review"   │
  │  appeal) │  └────────────────────┘
  └──────────┘
       │
       v
  ┌─────────────────────────────────┐
  │            Response              │
  │ {appeal_id, content_id,          │
  │  status: "under_review",         │
  │  message: "..."}                 │
  └─────────────────────────────────┘

Arrows between components carry:
  POST /appeal body ──► Rate Limiter: {content_id, creator_id, reason}
  Rate Limiter ──► Validator: {content_id, creator_id, reason}
  Validator ──► Appeal Handler: validated fields
  Appeal Handler ──► Audit Logger (read): content_id → existing entry
  Appeal Handler ──► Audit Logger (write): appeal record + status update
  Audit Logger ──► Response: appeal_id
```

---

## Transparency Label Variants

These are the exact strings returned by the API and displayed to platform users.

**Variant 1 — High-confidence human (confidence_score < 0.35):**
```
Verdict: Likely Human-Written
Confidence: High

Our system analyzed this content and found strong indicators of human authorship.
This is an automated assessment and is not guaranteed to be correct.
If you are the creator and believe this label is incorrect, you may submit an appeal.
```

**Variant 2 — Uncertain (0.35 ≤ confidence_score ≤ 0.70):**
```
Verdict: Authorship Uncertain
Confidence: Low — this content has not been labeled as AI-generated.

Our system detected mixed signals and cannot determine authorship with confidence.
This content has not been flagged as AI-generated.
If you are the creator and disagree with this assessment, you may submit an appeal.
```

**Variant 3 — High-confidence AI (confidence_score > 0.70):**
```
Verdict: Likely AI-Generated
Confidence: High

Our system found strong indicators that this content may have been AI-generated.
This is an automated assessment and may be incorrect.
If you are the creator and believe this label is incorrect, you may submit an appeal.
```

---

## Rate Limiting

**Limit:** 10 submissions per hour per IP address.

**Reasoning:**
- A typical active writer submits 1–5 pieces per day. 10/hour gives legitimate users comfortable headroom for testing, iterating, and resubmission without ever hitting a wall.
- An adversary trying to flood the classifier or probe thresholds would need to spread requests across many hours or IPs — 10/hour makes brute-force threshold probing expensive and slow.
- The free Groq tier has its own rate limits (~30 req/min). Our 10/hour ceiling keeps us well under that and prevents a single bad actor from exhausting the API quota for all users.
- Appeals also fall under the same limit; a creator contesting multiple pieces within an hour is unusual and should not exceed 10 interactions.

---

## File Structure

```
provenance-guard/
├── app.py                    # Flask app factory + all routes
├── config.py                 # Thresholds, weights, rate limit, Groq model
├── requirements.txt
├── .env.example
├── planning.md               # This file
├── README.md
├── audit.db                  # Created at runtime by AuditLogger
├── detection/
│   ├── __init__.py
│   ├── pipeline.py           # Orchestrates both signals → PipelineResult
│   ├── llm_signal.py         # Groq API call + score extraction
│   ├── stylometric.py        # Pure Python heuristics → StylemetricResult
│   └── confidence.py         # Weighted scoring + label generation
├── audit/
│   ├── __init__.py
│   └── logger.py             # SQLite schema init + read/write
├── appeals/
│   ├── __init__.py
│   └── handler.py            # Appeal validation + delegation to logger
└── tests/
    ├── test_stylometric.py
    ├── test_confidence.py
    ├── test_llm_signal.py
    ├── test_pipeline.py
    ├── test_audit.py
    ├── test_appeals.py
    └── test_routes.py
```

---

## Milestone 2: Implementation Tasks

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `.env.example`
- Create: `detection/__init__.py`, `audit/__init__.py`, `appeals/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
flask==3.1.0
flask-limiter==3.9.4
groq==0.29.0
python-dotenv==1.0.1
pytest==8.4.0
pytest-mock==3.14.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Create `config.py`**

```python
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Confidence score thresholds (asymmetric — protects against false positives)
HUMAN_THRESHOLD = 0.35   # below this → high-confidence human
AI_THRESHOLD = 0.70      # above this → high-confidence AI

# Signal weights (must sum to 1.0)
LLM_WEIGHT = 0.60
STYLOMETRIC_WEIGHT = 0.40

# Rate limiting
RATE_LIMIT = "10 per hour"

# Content limits
MAX_CONTENT_CHARS = 10_000
MIN_CONTENT_WORDS = 10
```

- [ ] **Step 4: Create `.env.example`**

```
GROQ_API_KEY=your_groq_api_key_here
```

- [ ] **Step 5: Create empty `__init__.py` files**

Create: `detection/__init__.py`, `audit/__init__.py`, `appeals/__init__.py` (all empty).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt config.py .env.example detection/__init__.py audit/__init__.py appeals/__init__.py
git commit -m "feat: project setup — requirements, config, package structure"
```

---

### Task 2: Stylometric Analyzer

**Files:**
- Create: `detection/stylometric.py`
- Create: `tests/test_stylometric.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stylometric.py`:
```python
import pytest
from detection.stylometric import compute_stylometric_score, StylemetricResult

def test_uniform_sentences_score_higher_ai():
    uniform = "The cat sat on the mat. The dog ran to the park. The bird flew past quickly."
    varied = "Short. This sentence is considerably longer and more complex in its structure. Brief. But this one is very very very very long indeed and goes on and on."
    score_uniform = compute_stylometric_score(uniform)
    score_varied = compute_stylometric_score(varied)
    assert score_uniform.score > score_varied.score

def test_returns_stylometric_result():
    result = compute_stylometric_score("Hello world. This is a test sentence with enough words.")
    assert isinstance(result, StylemetricResult)
    assert 0.0 <= result.score <= 1.0

def test_result_contains_component_scores():
    result = compute_stylometric_score(
        "The cat sat on the mat. The dog ran to the park. "
        "The bird flew high in the sky and circled twice above the old oak tree."
    )
    assert 0.0 <= result.variance_score <= 1.0
    assert 0.0 <= result.ttr_score <= 1.0
    assert 0.0 <= result.punctuation_score <= 1.0

def test_insufficient_text_returns_uncertain():
    result = compute_stylometric_score("Short text.")
    assert result.score == 0.5

def test_empty_string_returns_uncertain():
    result = compute_stylometric_score("")
    assert result.score == 0.5

def test_expressive_punctuation_lowers_ai_score():
    no_punct = "The system processed the data. It returned a result. The result was correct."
    with_punct = "The system processed the data! It returned a result — unexpectedly. Was it correct?"
    score_no = compute_stylometric_score(no_punct)
    score_with = compute_stylometric_score(with_punct)
    assert score_no.punctuation_score > score_with.punctuation_score
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stylometric.py -v
```

Expected: `ModuleNotFoundError: No module named 'detection.stylometric'`

- [ ] **Step 3: Implement `detection/stylometric.py`**

```python
import re
import math
from dataclasses import dataclass

@dataclass
class StylemetricResult:
    score: float           # 0–1, where 1 = likely AI
    variance_score: float  # sentence length uniformity (high = AI)
    ttr_score: float       # low vocabulary diversity (high = AI)
    punctuation_score: float  # low expressive punctuation (high = AI)

def compute_stylometric_score(text: str) -> StylemetricResult:
    """Compute AI-likelihood from structural text properties. Returns 0–1 (1 = AI)."""
    words = text.split()
    if len(words) < 10:
        return StylemetricResult(score=0.5, variance_score=0.5, ttr_score=0.5, punctuation_score=0.5)

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return StylemetricResult(score=0.5, variance_score=0.5, ttr_score=0.5, punctuation_score=0.5)

    variance_score = _sentence_variance_score(sentences)
    ttr_score = _type_token_ratio_score(words)
    punctuation_score = _punctuation_score(text)

    score = (0.40 * variance_score) + (0.35 * ttr_score) + (0.25 * punctuation_score)
    return StylemetricResult(
        score=round(score, 4),
        variance_score=round(variance_score, 4),
        ttr_score=round(ttr_score, 4),
        punctuation_score=round(punctuation_score, 4),
    )

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

def _sentence_variance_score(sentences: list[str]) -> float:
    """Low variance → high AI score. AI produces more uniform sentence lengths."""
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    # AI typically std_dev < 4; human creative writing often > 10
    score = max(0.0, 1.0 - (std_dev / 12.0))
    return min(1.0, score)

def _type_token_ratio_score(words: list[str]) -> float:
    """Low TTR → high AI score. AI reuses vocabulary patterns more."""
    lower_words = [w.lower().strip(".,!?;:\"'") for w in words]
    ttr = len(set(lower_words)) / len(lower_words)
    # High TTR = diverse vocabulary = human; score inverts TTR to penalize diversity
    score = max(0.0, 1.0 - ttr)
    return min(1.0, score)

def _punctuation_score(text: str) -> float:
    """Low expressive punctuation density → high AI score."""
    expressive = sum(1 for c in text if c in '!?;:—–…()')
    total = len(text)
    if total == 0:
        return 0.5
    density = expressive / total
    # density=0 → score=1.0 (AI: no expressive punct); density>=0.05 → score≈0.0
    score = max(0.0, 1.0 - (density * 20.0))
    return min(1.0, score)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stylometric.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add detection/stylometric.py tests/test_stylometric.py
git commit -m "feat: stylometric analyzer — sentence variance, TTR, punctuation density"
```

---

### Task 3: Confidence Scorer and Label Generator

**Files:**
- Create: `detection/confidence.py`
- Create: `tests/test_confidence.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_confidence.py`:
```python
import pytest
from detection.confidence import combine_scores, generate_label, AttributionResult

def test_low_score_returns_human():
    result = combine_scores(llm_score=0.20, stylometric_score=0.25)
    assert result.attribution == "human"
    assert result.confidence_score < 0.35

def test_high_score_returns_ai():
    result = combine_scores(llm_score=0.85, stylometric_score=0.80)
    assert result.attribution == "ai"
    assert result.confidence_score > 0.70

def test_mid_score_returns_uncertain():
    result = combine_scores(llm_score=0.55, stylometric_score=0.50)
    assert result.attribution == "uncertain"

def test_weights_applied_correctly():
    # LLM=0.60 weight, stylometric=0.40 weight
    result = combine_scores(llm_score=1.0, stylometric_score=0.0)
    assert abs(result.confidence_score - 0.60) < 0.01

def test_label_high_confidence_human():
    label = generate_label(0.20, "human")
    assert "Human-Written" in label["verdict"]
    assert "High" in label["confidence_display"]
    assert "appeal" in label["detail"].lower()

def test_label_uncertain():
    label = generate_label(0.52, "uncertain")
    assert "Uncertain" in label["verdict"]
    assert "not been labeled as AI-generated" in label["confidence_display"]
    assert "appeal" in label["detail"].lower()

def test_label_high_confidence_ai():
    label = generate_label(0.85, "ai")
    assert "AI-Generated" in label["verdict"]
    assert "appeal" in label["detail"].lower()

def test_attribution_result_stores_signal_scores():
    result = combine_scores(llm_score=0.30, stylometric_score=0.40)
    assert result.llm_score == 0.30
    assert result.stylometric_score == 0.40
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_confidence.py -v
```

Expected: `ModuleNotFoundError: No module named 'detection.confidence'`

- [ ] **Step 3: Implement `detection/confidence.py`**

```python
from dataclasses import dataclass
from config import HUMAN_THRESHOLD, AI_THRESHOLD, LLM_WEIGHT, STYLOMETRIC_WEIGHT

@dataclass
class AttributionResult:
    attribution: str        # "human" | "ai" | "uncertain"
    confidence_score: float # 0–1 (higher = stronger AI signal)
    llm_score: float
    stylometric_score: float

def combine_scores(llm_score: float, stylometric_score: float) -> AttributionResult:
    """Combine signal scores into a single confidence score and attribution."""
    combined = (LLM_WEIGHT * llm_score) + (STYLOMETRIC_WEIGHT * stylometric_score)
    combined = round(combined, 4)

    if combined < HUMAN_THRESHOLD:
        attribution = "human"
    elif combined > AI_THRESHOLD:
        attribution = "ai"
    else:
        attribution = "uncertain"

    return AttributionResult(
        attribution=attribution,
        confidence_score=combined,
        llm_score=round(llm_score, 4),
        stylometric_score=round(stylometric_score, 4),
    )

def generate_label(confidence_score: float, attribution: str) -> dict:
    """Generate the transparency label text shown to platform users."""
    if attribution == "human":
        return {
            "verdict": "Likely Human-Written",
            "confidence_display": "High",
            "detail": (
                "Our system analyzed this content and found strong indicators of human authorship. "
                "This is an automated assessment and is not guaranteed to be correct. "
                "If you are the creator and believe this label is incorrect, you may submit an appeal."
            ),
        }
    elif attribution == "ai":
        return {
            "verdict": "Likely AI-Generated",
            "confidence_display": "High",
            "detail": (
                "Our system found strong indicators that this content may have been AI-generated. "
                "This is an automated assessment and may be incorrect. "
                "If you are the creator and believe this label is incorrect, you may submit an appeal."
            ),
        }
    else:
        return {
            "verdict": "Authorship Uncertain",
            "confidence_display": "Low — this content has not been labeled as AI-generated",
            "detail": (
                "Our system detected mixed signals and cannot determine authorship with confidence. "
                "This content has not been flagged as AI-generated. "
                "If you are the creator and disagree with this assessment, you may submit an appeal."
            ),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_confidence.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add detection/confidence.py tests/test_confidence.py
git commit -m "feat: confidence scorer and transparency label generator"
```

---

### Task 4: LLM Signal (Groq)

**Files:**
- Create: `detection/llm_signal.py`
- Create: `tests/test_llm_signal.py`

- [ ] **Step 1: Write failing tests (with mocks — do not hit real Groq API in tests)**

Create `tests/test_llm_signal.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from detection.llm_signal import classify_with_llm, LLMResult

def _make_mock_response(content: str):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock

def test_returns_llm_result_type():
    with patch("detection.llm_signal.client.chat.completions.create",
               return_value=_make_mock_response("0.15")):
        result = classify_with_llm("Once upon a time in a small village...")
        assert isinstance(result, LLMResult)

def test_parses_score_correctly():
    with patch("detection.llm_signal.client.chat.completions.create",
               return_value=_make_mock_response("0.87")):
        result = classify_with_llm("This content was generated by an AI system.")
        assert abs(result.score - 0.87) < 0.01

def test_clamps_score_above_1():
    with patch("detection.llm_signal.client.chat.completions.create",
               return_value=_make_mock_response("1.5")):
        result = classify_with_llm("some text")
        assert result.score == 1.0

def test_clamps_score_below_0():
    with patch("detection.llm_signal.client.chat.completions.create",
               return_value=_make_mock_response("-0.3")):
        result = classify_with_llm("some text")
        assert result.score == 0.0

def test_unparseable_response_returns_uncertain():
    with patch("detection.llm_signal.client.chat.completions.create",
               return_value=_make_mock_response("I cannot determine this.")):
        result = classify_with_llm("some text")
        assert result.score == 0.5
        assert result.parse_error is True

def test_api_error_returns_uncertain():
    with patch("detection.llm_signal.client.chat.completions.create",
               side_effect=Exception("API error")):
        result = classify_with_llm("some text")
        assert result.score == 0.5
        assert result.api_error is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_signal.py -v
```

Expected: `ModuleNotFoundError: No module named 'detection.llm_signal'`

- [ ] **Step 3: Implement `detection/llm_signal.py`**

```python
from dataclasses import dataclass, field
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

_SYSTEM_PROMPT = """You are an expert at detecting whether text was written by a human or generated by AI.
Analyze the submitted text and return ONLY a single decimal number between 0.0 and 1.0, where:
- 0.0 = you are certain the text is human-written
- 1.0 = you are certain the text is AI-generated
- 0.5 = you cannot determine

Return only the number. No explanation. Example: 0.12"""

@dataclass
class LLMResult:
    score: float
    raw_response: str = ""
    parse_error: bool = False
    api_error: bool = False

def classify_with_llm(text: str) -> LLMResult:
    """Send text to Groq LLM and return AI-likelihood score 0–1."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{text}"},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_score(raw)
    except Exception:
        return LLMResult(score=0.5, api_error=True)

def _parse_score(raw: str) -> LLMResult:
    try:
        score = float(raw)
        score = max(0.0, min(1.0, score))
        return LLMResult(score=round(score, 4), raw_response=raw)
    except ValueError:
        return LLMResult(score=0.5, raw_response=raw, parse_error=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_signal.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add detection/llm_signal.py tests/test_llm_signal.py
git commit -m "feat: LLM classifier signal via Groq API with error handling"
```

---

### Task 5: Detection Pipeline

**Files:**
- Create: `detection/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:
```python
import pytest
from unittest.mock import patch
from detection.pipeline import run_detection_pipeline, PipelineResult
from detection.llm_signal import LLMResult
from detection.stylometric import StylemetricResult

def _mock_llm(score):
    return LLMResult(score=score)

def _mock_stylo(score):
    return StylemetricResult(score=score, variance_score=score, ttr_score=score, punctuation_score=score)

def test_returns_pipeline_result():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.2)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.3)):
        result = run_detection_pipeline("Some human text here with enough words.")
        assert isinstance(result, PipelineResult)

def test_human_signals_produce_human_attribution():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.1)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.1)):
        result = run_detection_pipeline("Human text.")
        assert result.attribution == "human"
        assert result.confidence_score < 0.35

def test_ai_signals_produce_ai_attribution():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.9)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.9)):
        result = run_detection_pipeline("AI text.")
        assert result.attribution == "ai"
        assert result.confidence_score > 0.70

def test_result_includes_signal_breakdown():
    with patch("detection.pipeline.classify_with_llm", return_value=_mock_llm(0.1)), \
         patch("detection.pipeline.compute_stylometric_score", return_value=_mock_stylo(0.2)):
        result = run_detection_pipeline("Human text.")
        assert hasattr(result, "llm_score")
        assert hasattr(result, "stylometric_score")
        assert hasattr(result, "transparency_label")
        assert isinstance(result.transparency_label, dict)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'detection.pipeline'`

- [ ] **Step 3: Implement `detection/pipeline.py`**

```python
from dataclasses import dataclass
from detection.llm_signal import classify_with_llm
from detection.stylometric import compute_stylometric_score
from detection.confidence import combine_scores, generate_label

@dataclass
class PipelineResult:
    attribution: str           # "human" | "ai" | "uncertain"
    confidence_score: float    # 0–1
    llm_score: float
    stylometric_score: float
    transparency_label: dict   # {verdict, confidence_display, detail}

def run_detection_pipeline(text: str) -> PipelineResult:
    """Run both detection signals and return a combined result."""
    llm_result = classify_with_llm(text)
    stylo_result = compute_stylometric_score(text)

    attribution_result = combine_scores(
        llm_score=llm_result.score,
        stylometric_score=stylo_result.score,
    )
    label = generate_label(attribution_result.confidence_score, attribution_result.attribution)

    return PipelineResult(
        attribution=attribution_result.attribution,
        confidence_score=attribution_result.confidence_score,
        llm_score=llm_result.score,
        stylometric_score=stylo_result.score,
        transparency_label=label,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add detection/pipeline.py tests/test_pipeline.py
git commit -m "feat: detection pipeline — orchestrates LLM + stylometric signals"
```

---

### Task 6: Audit Logger (SQLite)

**Files:**
- Create: `audit/logger.py`
- Create: `tests/test_audit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_audit.py`:
```python
import pytest
from audit.logger import AuditLogger

@pytest.fixture
def logger(tmp_path):
    log = AuditLogger(db_path=str(tmp_path / "test_audit.db"))
    log.init_db()
    return log

def test_log_decision_returns_content_id(logger):
    content_id = logger.log_decision(
        creator_id="user_1",
        content_snippet="The cat sat on the mat.",
        llm_score=0.15,
        stylometric_score=0.25,
        confidence_score=0.19,
        attribution="human",
        transparency_label="Likely Human-Written",
    )
    assert content_id.startswith("cnt_")

def test_get_entry_returns_logged_decision(logger):
    content_id = logger.log_decision(
        creator_id="user_1",
        content_snippet="Test content.",
        llm_score=0.80,
        stylometric_score=0.75,
        confidence_score=0.78,
        attribution="ai",
        transparency_label="Likely AI-Generated",
    )
    entry = logger.get_entry(content_id)
    assert entry is not None
    assert entry["attribution"] == "ai"
    assert entry["confidence_score"] == 0.78
    assert entry["status"] == "classified"

def test_log_appeal_updates_status_to_under_review(logger):
    content_id = logger.log_decision(
        creator_id="user_2", content_snippet="Human text.",
        llm_score=0.20, stylometric_score=0.30,
        confidence_score=0.24, attribution="human",
        transparency_label="Likely Human-Written",
    )
    appeal_id = logger.log_appeal(content_id=content_id, creator_id="user_2",
                                   reason="I wrote this myself.")
    assert appeal_id.startswith("app_")
    entry = logger.get_entry(content_id)
    assert entry["status"] == "under_review"
    assert len(entry["appeals"]) == 1
    assert entry["appeals"][0]["reason"] == "I wrote this myself."

def test_get_all_entries_returns_list_of_decisions(logger):
    logger.log_decision("u1", "text1", 0.1, 0.2, 0.14, "human", "Likely Human-Written")
    logger.log_decision("u2", "text2", 0.8, 0.9, 0.84, "ai", "Likely AI-Generated")
    entries = logger.get_all_entries()
    assert len(entries) >= 2

def test_get_entry_returns_none_for_unknown_id(logger):
    assert logger.get_entry("cnt_doesnotexist") is None

def test_log_appeal_raises_on_unknown_content_id(logger):
    with pytest.raises(ValueError, match="not found"):
        logger.log_appeal("cnt_unknown", "user_1", "My work.")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_audit.py -v
```

Expected: `ModuleNotFoundError: No module named 'audit.logger'`

- [ ] **Step 3: Implement `audit/logger.py`**

```python
import sqlite3
import uuid
import json
from datetime import datetime, timezone

class AuditLogger:
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    content_id TEXT PRIMARY KEY,
                    creator_id TEXT NOT NULL,
                    content_snippet TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    llm_score REAL NOT NULL,
                    stylometric_score REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    attribution TEXT NOT NULL,
                    transparency_label TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'classified',
                    appeals TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.commit()

    def log_decision(self, creator_id: str, content_snippet: str,
                     llm_score: float, stylometric_score: float,
                     confidence_score: float, attribution: str,
                     transparency_label: str) -> str:
        content_id = f"cnt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO decisions
                   (content_id, creator_id, content_snippet, timestamp,
                    llm_score, stylometric_score, confidence_score,
                    attribution, transparency_label, status, appeals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'classified', '[]')""",
                (content_id, creator_id, content_snippet[:500],
                 timestamp, llm_score, stylometric_score,
                 confidence_score, attribution, transparency_label),
            )
            conn.commit()
        return content_id

    def log_appeal(self, content_id: str, creator_id: str, reason: str) -> str:
        appeal_id = f"app_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        appeal = {"appeal_id": appeal_id, "creator_id": creator_id,
                  "reason": reason, "timestamp": timestamp}
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT appeals FROM decisions WHERE content_id = ?", (content_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"content_id {content_id} not found")
            appeals = json.loads(row[0])
            appeals.append(appeal)
            conn.execute(
                "UPDATE decisions SET appeals = ?, status = 'under_review' WHERE content_id = ?",
                (json.dumps(appeals), content_id),
            )
            conn.commit()
        return appeal_id

    def get_entry(self, content_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM decisions WHERE content_id = ?", (content_id,)
            ).fetchone()
        if row is None:
            return None
        entry = dict(row)
        entry["appeals"] = json.loads(entry["appeals"])
        return entry

    def get_all_entries(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        entries = []
        for row in rows:
            entry = dict(row)
            entry["appeals"] = json.loads(entry["appeals"])
            entries.append(entry)
        return entries
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_audit.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add audit/logger.py tests/test_audit.py
git commit -m "feat: SQLite audit logger — decisions, appeals, status updates"
```

---

### Task 7: Appeals Handler

**Files:**
- Create: `appeals/handler.py`
- Create: `tests/test_appeals.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_appeals.py`:
```python
import pytest
from unittest.mock import MagicMock
from appeals.handler import process_appeal, AppealError

def test_returns_appeal_id_and_under_review_status():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_1"}
    mock_logger.log_appeal.return_value = "app_xyz"

    result = process_appeal(content_id="cnt_abc", creator_id="user_1",
                             reason="I wrote this myself.", logger=mock_logger)
    assert result["appeal_id"] == "app_xyz"
    assert result["status"] == "under_review"
    assert "message" in result

def test_raises_on_unknown_content_id():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = None

    with pytest.raises(AppealError, match="not found"):
        process_appeal(content_id="cnt_unknown", creator_id="user_1",
                       reason="My work.", logger=mock_logger)

def test_raises_on_creator_mismatch():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_correct"}

    with pytest.raises(AppealError, match="creator_id"):
        process_appeal(content_id="cnt_abc", creator_id="user_wrong",
                       reason="My work.", logger=mock_logger)

def test_raises_on_empty_reason():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_1"}

    with pytest.raises(AppealError, match="reason"):
        process_appeal(content_id="cnt_abc", creator_id="user_1",
                       reason="   ", logger=mock_logger)

def test_delegates_to_logger_with_correct_args():
    mock_logger = MagicMock()
    mock_logger.get_entry.return_value = {"content_id": "cnt_abc", "creator_id": "user_1"}
    mock_logger.log_appeal.return_value = "app_xyz"

    process_appeal(content_id="cnt_abc", creator_id="user_1",
                   reason="I wrote this.", logger=mock_logger)
    mock_logger.log_appeal.assert_called_once_with(
        content_id="cnt_abc", creator_id="user_1", reason="I wrote this."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_appeals.py -v
```

Expected: `ModuleNotFoundError: No module named 'appeals.handler'`

- [ ] **Step 3: Implement `appeals/handler.py`**

```python
class AppealError(Exception):
    pass

def process_appeal(content_id: str, creator_id: str, reason: str, logger) -> dict:
    """Validate and process a creator appeal for a content classification."""
    if not reason or not reason.strip():
        raise AppealError("reason must not be empty")

    entry = logger.get_entry(content_id)
    if entry is None:
        raise AppealError(f"content_id {content_id} not found")

    if entry["creator_id"] != creator_id:
        raise AppealError("creator_id does not match the original submission")

    appeal_id = logger.log_appeal(
        content_id=content_id,
        creator_id=creator_id,
        reason=reason.strip(),
    )

    return {
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "under_review",
        "message": (
            "Your appeal has been received. "
            "The classification has been marked as under review."
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_appeals.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add appeals/handler.py tests/test_appeals.py
git commit -m "feat: appeals handler — validation, creator verification, delegation to logger"
```

---

### Task 8: Flask Routes and Rate Limiting

**Files:**
- Create: `app.py`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_routes.py`:
```python
import pytest
import json
import os
from unittest.mock import patch
from detection.pipeline import PipelineResult

def _mock_result(attribution="human", confidence=0.20):
    return PipelineResult(
        attribution=attribution,
        confidence_score=confidence,
        llm_score=0.15,
        stylometric_score=0.28,
        transparency_label={
            "verdict": "Likely Human-Written",
            "confidence_display": "High",
            "detail": "Our system found strong indicators of human authorship. If you are the creator and believe this label is incorrect, you may submit an appeal.",
        },
    )

@pytest.fixture
def client(tmp_path):
    os.environ["AUDIT_DB_PATH"] = str(tmp_path / "test.db")
    # Reset singleton logger between tests
    import app as app_module
    app_module._logger_instance = None
    flask_app = app_module.create_app(testing=True)
    with flask_app.test_client() as c:
        yield c

_VALID_CONTENT = "The starling flew past the window at dusk. It circled twice, then vanished."

def test_submit_returns_200_with_all_fields(client):
    with patch("app.run_detection_pipeline", return_value=_mock_result()):
        resp = client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "user_1"})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "content_id" in data
        assert "attribution_result" in data
        assert "confidence_score" in data
        assert "transparency_label" in data

def test_submit_returns_400_on_missing_content(client):
    resp = client.post("/submit", json={"creator_id": "user_1"})
    assert resp.status_code == 400

def test_submit_returns_400_on_missing_creator_id(client):
    resp = client.post("/submit", json={"content": _VALID_CONTENT})
    assert resp.status_code == 400

def test_submit_returns_400_on_too_short_content(client):
    resp = client.post("/submit", json={"content": "Short.", "creator_id": "user_1"})
    assert resp.status_code == 400

def test_appeal_returns_200_and_under_review(client):
    with patch("app.run_detection_pipeline", return_value=_mock_result()):
        submit_resp = client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "user_1"})
        content_id = json.loads(submit_resp.data)["content_id"]

    resp = client.post("/appeal", json={
        "content_id": content_id, "creator_id": "user_1",
        "reason": "I wrote this myself, it reflects my personal experience.",
    })
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "under_review"
    assert "appeal_id" in data

def test_appeal_returns_404_on_unknown_content_id(client):
    resp = client.post("/appeal", json={
        "content_id": "cnt_nonexistent", "creator_id": "user_1",
        "reason": "I wrote this.",
    })
    assert resp.status_code == 404

def test_log_returns_entries_list(client):
    with patch("app.run_detection_pipeline", return_value=_mock_result()):
        client.post("/submit", json={"content": _VALID_CONTENT, "creator_id": "user_1"})
    resp = client.get("/log")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "entries" in data
    assert len(data["entries"]) >= 1

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert json.loads(resp.data)["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Implement `app.py`**

```python
import os
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from detection.pipeline import run_detection_pipeline
from audit.logger import AuditLogger
from appeals.handler import process_appeal, AppealError
from config import RATE_LIMIT, MAX_CONTENT_CHARS, MIN_CONTENT_WORDS

_logger_instance = None

def get_logger() -> AuditLogger:
    global _logger_instance
    if _logger_instance is None:
        db_path = os.getenv("AUDIT_DB_PATH", "audit.db")
        _logger_instance = AuditLogger(db_path=db_path)
        _logger_instance.init_db()
    return _logger_instance

def create_app(testing: bool = False) -> Flask:
    global _logger_instance
    if testing:
        _logger_instance = None

    app = Flask(__name__)
    app.config["TESTING"] = testing

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[RATE_LIMIT],
        storage_uri="memory://",
    )

    @app.route("/health", methods=["GET"])
    @limiter.exempt
    def health():
        return jsonify({"status": "ok", "version": "1.0.0"})

    @app.route("/submit", methods=["POST"])
    @limiter.limit(RATE_LIMIT)
    def submit():
        body = request.get_json(silent=True) or {}
        content = body.get("content", "").strip()
        creator_id = body.get("creator_id", "").strip()

        if not content:
            return jsonify({"error": "content is required"}), 400
        if not creator_id:
            return jsonify({"error": "creator_id is required"}), 400
        if len(content) > MAX_CONTENT_CHARS:
            return jsonify({"error": f"content exceeds {MAX_CONTENT_CHARS} characters"}), 400
        if len(content.split()) < MIN_CONTENT_WORDS:
            return jsonify({"error": f"content must have at least {MIN_CONTENT_WORDS} words"}), 400

        result = run_detection_pipeline(content)
        content_id = get_logger().log_decision(
            creator_id=creator_id,
            content_snippet=content[:500],
            llm_score=result.llm_score,
            stylometric_score=result.stylometric_score,
            confidence_score=result.confidence_score,
            attribution=result.attribution,
            transparency_label=result.transparency_label["verdict"],
        )

        return jsonify({
            "content_id": content_id,
            "attribution_result": result.attribution,
            "confidence_score": result.confidence_score,
            "transparency_label": result.transparency_label,
        })

    @app.route("/appeal", methods=["POST"])
    @limiter.limit(RATE_LIMIT)
    def appeal():
        body = request.get_json(silent=True) or {}
        content_id = body.get("content_id", "").strip()
        creator_id = body.get("creator_id", "").strip()
        reason = body.get("reason", "").strip()

        if not content_id or not creator_id or not reason:
            return jsonify({"error": "content_id, creator_id, and reason are required"}), 400

        try:
            result = process_appeal(
                content_id=content_id,
                creator_id=creator_id,
                reason=reason,
                logger=get_logger(),
            )
            return jsonify(result)
        except AppealError as e:
            msg = str(e)
            status_code = 404 if "not found" in msg else 400
            return jsonify({"error": msg}), status_code

    @app.route("/log", methods=["GET"])
    @limiter.exempt
    def log():
        entries = get_logger().get_all_entries()
        return jsonify({"entries": entries})

    return app

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    flask_app = create_app()
    flask_app.run(debug=True, port=5000)
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/ -v
```

Expected: all tests PASSED (23+ tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_routes.py
git commit -m "feat: Flask routes — submit, appeal, log, health with rate limiting"
```

---

### Task 9: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with all required grader documentation**

The README must include: setup instructions, all three transparency label variants (verbatim text), rate limiting rationale, and a sample audit log showing ≥3 entries with appeals visible. Write it after running the app locally to capture real output.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README — setup, label variants, rate limit reasoning, audit log sample"
```

---

## Spec Coverage Check

| Requirement | Task | Notes |
|---|---|---|
| Content Submission Endpoint (POST /submit, returns result + score + label) | Task 8 | ✅ |
| Multi-Signal Pipeline (≥2 distinct signals) | Tasks 2, 4, 5 | ✅ LLM (semantic) + stylometric (structural) |
| Confidence Scoring with Uncertainty | Task 3 | ✅ 0.51 vs 0.95 produce different labels |
| Transparency Label (3 variants verbatim in README) | Tasks 3, 9 | ✅ |
| Appeals Workflow (capture reason, log, update status) | Tasks 6, 7, 8 | ✅ |
| Rate Limiting with documented reasoning | Tasks 8, 9 | ✅ 10/hr with rationale |
| Audit Log structured, ≥3 entries visible | Tasks 6, 9 | ✅ GET /log + README sample |
| Architecture diagram in planning.md under ## Architecture | This file | ✅ |
| Verbatim label text in README (not just screenshot) | Task 9 | ✅ must verify |
| planning.md explains each signal + blind spots | This file | ✅ |
