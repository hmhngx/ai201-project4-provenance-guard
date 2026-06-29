# Provenance Guard — Planning & Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask API that classifies text content as human or AI-written, returns a confidence score and plain-language transparency label, and handles creator appeals with a full structured audit trail.

**Tech Stack:** Flask, Flask-Limiter, Groq Python SDK (`llama-3.3-70b-versatile`), SQLite (built-in), pure Python for stylometrics.

---

## Architecture

### Narrative

**Submission flow:** A creator POSTs text to `/submit`. The Rate Limiter checks the IP's request count (10/hour); if exceeded, returns 429. The Request Validator ensures `content` and `creator_id` are present and within size bounds; if not, returns 400. The Detection Pipeline runs two independent signals in sequence — first the LLM Classifier (Groq API), which returns a float 0–1, then the Stylometric Analyzer (pure Python), which also returns a float 0–1. The Confidence Scorer combines those two scores into a single weighted score and determines attribution (`human` / `uncertain` / `ai`) using asymmetric thresholds. The Label Generator converts that into three fields of plain-language text. The Audit Logger writes the full decision record to SQLite. The response returns `content_id`, `attribution_result`, `confidence_score`, and `transparency_label` as JSON.

**Appeal flow:** A creator POSTs to `/appeal` with `content_id`, `creator_id`, and `reason`. The Appeal Handler looks up the original decision in SQLite, validates that `creator_id` matches the original submission, and validates that `reason` is non-empty. If valid, it appends an appeal record to the entry's appeals list and updates its `status` to `under_review`. The response returns `appeal_id` and the new status. No automated re-classification occurs — the status change signals to a human reviewer that this entry needs attention.

### Diagram

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
│llm_score │  │stylo_score    │
│  (0–1)   │  │  (0–1)        │
└────┬─────┘  └──────┬────────┘
     │                │
     └────────┬───────┘
              │
              v
    ┌──────────────────────┐
    │   Confidence Scorer   │
    │ 0.60×LLM+0.40×stylo  │
    │ → confidence_score    │
    │         (0–1)         │
    └──────────┬────────────┘
               │
               v
    ┌──────────────────────┐
    │   Label Generator     │
    │ score < 0.35 → human  │
    │ 0.35–0.70 → uncertain │
    │ score > 0.70 → ai     │
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

Arrows carry:
  POST body ──► Rate Limiter: {content, creator_id}
  Rate Limiter ──► Validator: {content, creator_id}
  Validator ──► LLM Classifier: raw text string
  Validator ──► Stylometric Analyzer: raw text string
  LLM Classifier ──► Confidence Scorer: llm_score (float 0–1)
  Stylometric Analyzer ──► Confidence Scorer: stylo_score (float 0–1)
  Confidence Scorer ──► Label Generator: confidence_score (float) + attribution (string)
  Label Generator ──► Audit Logger: full decision record dict
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
  │  lookup content_id    │ ─────────► 404 not found
  │  validate creator_id  │ ─────────► 400 creator mismatch
  │  validate reason≠""   │ ─────────► 400 empty reason
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

Arrows carry:
  POST body ──► Rate Limiter: {content_id, creator_id, reason}
  Rate Limiter ──► Validator: {content_id, creator_id, reason}
  Validator ──► Appeal Handler: validated fields
  Appeal Handler ──► Audit Logger (read): content_id → existing entry dict
  Appeal Handler ──► Audit Logger (write): appeal record + status update
  Audit Logger ──► Response: appeal_id string
```

---

## Specification

### 1. Detection Signals

#### Signal 1: LLM Classification (Groq API)

**What it measures:** Holistic semantic and stylistic coherence — whether the text "feels" AI-generated based on sentence flow, phrasing naturalness, structural patterns, and presence/absence of a personal voice. The model evaluates whether word choices, transitions, and overall register suggest a human perspective or a generated synthesis.

**Output format:** A single float between 0.0 and 1.0, where 0.0 = certain human, 1.0 = certain AI. The LLM is prompted to return only this number; if the response is unparseable or the API call fails, the signal falls back to 0.5 (uncertain).

**Why it differs between human and AI text:** AI writing tends toward polished, coherent, well-structured text without rough edges — over-smooth transitions, generic phrasing, perfect grammar without personality. Human creative writing has natural imperfections: idiosyncratic word choices, personal references, tonal shifts, deliberate rule-breaking.

**Blind spots:** Can be fooled by highly polished human writers (professional editors, academics). Cannot detect AI writing that deliberately introduces stylistic errors. The model's assessment reflects its own training biases about what "AI writing looks like" — which may not match actual AI output. Non-deterministic: the same text may return slightly different scores across calls (mitigated by `temperature=0.1`).

**Implementation detail:** `POST` to Groq with system prompt asking for a single decimal 0–1. Parse `response.choices[0].message.content.strip()` as float. Clamp to [0.0, 1.0]. On any exception → return `LLMResult(score=0.5, api_error=True)`.

---

#### Signal 2: Stylometric Heuristics (Pure Python)

**What it measures:** Three independent structural statistics:

| Metric | What low values mean | What high values mean |
|---|---|---|
| Sentence length variance (std dev of word counts per sentence) | Uniform lengths → AI signal | High variation → human signal |
| Type-token ratio (unique words / total words) | Low vocabulary diversity → AI signal | High diversity → human signal |
| Expressive punctuation density (`! ? ; : — – … ()` / total chars) | Clean, minimal punct → AI signal | Rich expressive punct → human signal |

**Output format:** A single float 0–1 (1 = likely AI), computed as a weighted average of the three normalized sub-scores: `(0.40 × variance_score) + (0.35 × ttr_score) + (0.25 × punctuation_score)`. Each sub-score is independently normalized to [0, 1]. Also returns the three sub-scores individually for audit logging.

**Why it differs:** AI text generation produces more statistically uniform output — consistent sentence cadence, moderate vocabulary reuse, minimal expressive punctuation. Human creative writing varies more: fragments alongside complex constructions, vocabulary used precisely (high TTR in short texts), and emotional punctuation.

**Minimum text requirement:** Fewer than 10 words → return `StylemetricResult(score=0.5, ...)` (all sub-scores 0.5 = uncertain). Fewer than 2 sentences → same fallback.

**Blind spots:** Cannot capture meaning or narrative quality. Genre heavily skews results — see Edge Cases. Does not handle structured content (lists, code, templates) embedded in prose. Not calibrated against labeled data; thresholds are empirically estimated and would need tuning in a production deployment.

---

#### Combining the Signals

```
confidence_score = (0.60 × llm_score) + (0.40 × stylometric_score)
```

LLM receives 60% weight because it captures semantic and stylistic properties that the structural heuristics cannot see. Stylometric receives 40% because it is deterministic and transparent, providing an independent check on the LLM's holistic judgment.

Both signals return the same scale (0–1 = AI likelihood), so the weighted average is directly interpretable.

---

### 2. Uncertainty Representation

#### What does a score of 0.6 mean?

A confidence score of 0.6 means: the signals collectively lean toward AI-generated content, but not with enough certainty to make that call publicly. Specifically, it means the LLM and stylometric signals returned a weighted average that puts the content inside the uncertain zone (0.35–0.70). The system treats this as **"we don't know"**, not as "probably AI."

The user sees the uncertain label — which explicitly states "this content has NOT been flagged as AI-generated." This is a deliberate design choice: a 0.6 confidence is not enough to damage a creator's reputation. The asymmetric thresholds ensure that uncertain signals default toward the less-harmful outcome.

A score of 0.95 produces a materially different experience: it clears the 0.70 threshold, so the user sees the high-confidence AI label. The difference between 0.51 and 0.95 is not just a number — it determines which of three distinct label texts appears.

#### Threshold design

| Score range | Attribution | Reasoning |
|---|---|---|
| 0.00 – 0.34 | `human` | Strong evidence both signals point human |
| 0.35 – 0.70 | `uncertain` | Mixed signals — system withholds judgment |
| 0.71 – 1.00 | `ai` | Strong evidence both signals point AI |

The gap between 0.35 and 0.70 is intentionally wide. To reach `ai`, a piece would need something like LLM=0.85, stylometric=0.55: `(0.6×0.85)+(0.4×0.55) = 0.51+0.22 = 0.73`. A polished human essay scoring LLM=0.65, stylometric=0.50 gives `(0.6×0.65)+(0.4×0.50) = 0.39+0.20 = 0.59` — uncertain, not AI.

#### Calibration note

The system does not apply a secondary calibration layer (Platt scaling, isotonic regression). The raw weighted average is the score. This is honest about our limitations: we are not claiming to produce calibrated probabilities, just a risk-weighted signal. The thresholds do the work of converting scores to decisions. In a production deployment with labeled data, a calibration layer would improve reliability.

---

### 3. Transparency Label Variants

These are the exact strings the API returns in `transparency_label` and the platform displays to readers. All three variants include an appeal invitation.

**Variant 1 — High-confidence human (`confidence_score < 0.35`)**

```
Verdict: Likely Human-Written
Confidence: High

Our system analyzed this content and found strong indicators of human authorship.
This is an automated assessment and is not guaranteed to be correct.
If you are the creator and believe this label is incorrect, you may submit an appeal.
```

**Variant 2 — Uncertain (`0.35 ≤ confidence_score ≤ 0.70`)**

```
Verdict: Authorship Uncertain
Confidence: Low — this content has not been labeled as AI-generated.

Our system detected mixed signals and cannot determine authorship with confidence.
This content has not been flagged as AI-generated.
If you are the creator and disagree with this assessment, you may submit an appeal.
```

**Variant 3 — High-confidence AI (`confidence_score > 0.70`)**

```
Verdict: Likely AI-Generated
Confidence: High

Our system found strong indicators that this content may have been AI-generated.
This is an automated assessment and may be incorrect.
If you are the creator and believe this label is incorrect, you may submit an appeal.
```

**Design rationale:** All three variants use "may" or "is not guaranteed" language — we do not claim certainty even at high confidence. All three include the appeal path. The uncertain variant leads with what the system did NOT do ("has not been flagged as AI-generated") rather than what it found, because the uncertain-zone user's primary concern is whether they've been accused.

---

### 4. Appeals Workflow

**Who can submit an appeal:** Any creator who submitted content. The system validates that the `creator_id` in the appeal request matches the `creator_id` stored in the original audit log entry. Appeals from non-matching creator IDs are rejected with HTTP 400.

**What they provide:**
- `content_id` (required): the ID returned by the original `/submit` call
- `creator_id` (required): must match the original submitter
- `reason` (required, non-empty): the creator's explanation in free text — e.g. "I wrote this poem about my grandmother; it reflects a memory from my childhood"

**What the system does on receiving a valid appeal:**
1. Looks up the original decision in SQLite by `content_id`
2. Validates `creator_id` matches
3. Validates `reason` is non-empty after stripping whitespace
4. Generates a unique `appeal_id` (`app_` + 12 hex chars)
5. Appends the appeal record (appeal_id, creator_id, reason, timestamp) to the entry's `appeals` JSON array
6. Updates the entry's `status` field from `classified` to `under_review`
7. Returns `{appeal_id, content_id, status: "under_review", message}`

**What a human reviewer sees when opening the appeal queue (`GET /log`):**

Each entry in the log includes:
- `content_id`, `creator_id`, `content_snippet` (first 500 chars)
- `timestamp`, `attribution`, `confidence_score`, `transparency_label`
- `llm_score`, `stylometric_score` (for signal-level inspection)
- `status`: `"classified"` | `"under_review"`
- `appeals`: array of `{appeal_id, creator_id, reason, timestamp}`

A reviewer filtering on `status = "under_review"` sees the original classification, both signal scores, and the creator's stated reason. This gives enough context to make a manual decision — though the system itself takes no automated action beyond marking the status.

**Automated re-classification:** Not implemented. The system surfaces the appeal; human judgment resolves it.

---

### 5. Anticipated Edge Cases

**Edge case 1: Minimalist or repetitive poetry**

A poem that deliberately uses simple vocabulary and repetition — e.g. a haiku, a folk-tradition poem, or a piece like Carl Sandburg's *Fog* ("The fog comes / on little cat feet...") — will score poorly on both stylometric sub-metrics. Very short, uniform sentences → high `variance_score` (AI direction). Repetition of words → low TTR → high `ttr_score` (AI direction). Minimal punctuation → high `punctuation_score` (AI direction). The LLM signal may also see "clean, structured short text" and give a moderate AI score.

**Mitigation available but not implemented:** The minimum-word guard (10 words) catches haikus but not longer minimalist poems. A genre-detection pre-filter could downweight stylometric for poetry — but that's a stretch feature. For now, the system would likely return `uncertain` for most minimalist poems, which is acceptable: it doesn't accuse, and the creator can appeal.

---

**Edge case 2: Human-written technical documentation**

A developer writing a README or API reference uses consistent terminology by necessity (low TTR — words like "endpoint," "parameter," "returns" repeat heavily), uniform imperative sentence structure ("Set the `X` field to...", "Call the endpoint with..."), and minimal expressive punctuation. All three stylometric metrics would flag this as AI-even though the author is human.

The LLM signal is more likely to give a moderate score for technical prose rather than a high AI score — but combined with a high stylometric score, the result could land in the uncertain zone or tip into AI territory.

**Mitigation:** The 0.70 threshold is the primary defense. But this edge case illustrates that the stylometric signal is genre-sensitive and would need per-genre calibration in a production system. The README should note this limitation explicitly.

---

**Edge case 3: AI-generated text with deliberate imperfections**

A user prompts an AI to write "in an imperfect, human style with typos and casual language." The resulting text has irregular sentence lengths, colloquialisms, and expressive punctuation — all features the stylometric signal associates with human writing. The LLM signal might also be partially fooled, especially by a small model.

**Mitigation:** None that the current system can reliably apply. This is an unsolved problem in AI detection. The system's uncertainty range and the appeal mechanism are the honest response: when signals conflict, return `uncertain` rather than a false accusation.

---

## AI Tool Plan

For each milestone, the plan below specifies: which spec sections to give the AI tool as context, the exact generation request, and how to verify the output before proceeding.

### M3: Submission Endpoint + First Signal (LLM)

**Context to provide:** The `## Architecture` section (full narrative + submission flow diagram) + Detection Signals § Signal 1 (LLM) including the implementation detail note.

**What to ask the AI tool to generate:**
1. `app.py` — Flask app factory with `/submit`, `/health`, and `/log` stubs. The `/submit` stub should validate `content` (non-empty, ≤ 10,000 chars, ≥ 10 words) and `creator_id`, call a `run_detection_pipeline(text)` function (to be implemented), log the result, and return the structured JSON response.
2. `detection/llm_signal.py` — `classify_with_llm(text: str) -> LLMResult` using the Groq SDK. LLMResult dataclass with `score`, `raw_response`, `parse_error`, `api_error` fields. System prompt that asks for a single decimal 0–1 only. Clamping to [0.0, 1.0]. Fallback to score=0.5 on parse failure or API error.

**Verification before wiring into the endpoint:**
- Test `classify_with_llm()` directly in a Python shell with: (a) a paragraph of a human creative writing sample, (b) a clearly generic AI-sounding passage, (c) a minimalist poem. Check that scores differ in the expected direction and are in [0, 1].
- Confirm that a bad API key returns `LLMResult(score=0.5, api_error=True)` rather than raising an exception.
- Test the `/submit` stub returns 400 for missing `content`, 400 for missing `creator_id`, 400 for content under 10 words.

---

### M4: Second Signal + Confidence Scoring

**Context to provide:** Detection Signals § Signal 2 (stylometric — full metric table + output format + normalization formulas) + § Combining the Signals (weighted average formula) + Uncertainty Representation (thresholds table + what 0.6 means).

**What to ask the AI tool to generate:**
1. `detection/stylometric.py` — `compute_stylometric_score(text: str) -> StylemetricResult` with `StylemetricResult` dataclass (`score`, `variance_score`, `ttr_score`, `punctuation_score`). Include `_split_sentences`, `_sentence_variance_score`, `_type_token_ratio_score`, `_punctuation_score` as module-private helpers. Fallback to all-0.5 for texts under 10 words or under 2 sentences.
2. `detection/confidence.py` — `combine_scores(llm_score, stylometric_score) -> AttributionResult` using the exact formula and thresholds from the spec. `generate_label(confidence_score, attribution) -> dict` returning the three-field label dict matching the exact text from § Transparency Label Variants.
3. `detection/pipeline.py` — `run_detection_pipeline(text: str) -> PipelineResult` calling both signals and confidence scorer.

**Verification before proceeding to M5:**
- Test `compute_stylometric_score()` with: (a) text with clearly varied sentence lengths, (b) text with very uniform sentence lengths. Confirm varied > uniform in human direction (lower score).
- Test `combine_scores(0.10, 0.10)` → attribution=`human`, `combine_scores(0.80, 0.80)` → `ai`, `combine_scores(0.50, 0.50)` → `uncertain`.
- Test that 0.51 and 0.95 reach different label variants by calling `generate_label` directly.
- Run `run_detection_pipeline()` on 3 texts and print the full `PipelineResult` — confirm all fields are populated and confidence scores differ meaningfully.

---

### M5: Production Layer (Labels, Appeals, Rate Limiting, Audit Log)

**Context to provide:** § Transparency Label Variants (exact text for all three variants) + § Appeals Workflow (who, what, steps 1–7 in the spec) + Rate Limiting section + the appeal flow diagram from `## Architecture`.

**What to ask the AI tool to generate:**
1. `audit/logger.py` — `AuditLogger` class with `init_db()`, `log_decision(...)`, `log_appeal(...)`, `get_entry(content_id)`, `get_all_entries(limit)`. SQLite schema as specified. Appeals stored as a JSON array column. `log_appeal` raises `ValueError` if `content_id` not found.
2. `appeals/handler.py` — `process_appeal(content_id, creator_id, reason, logger) -> dict`. Validates reason non-empty, looks up entry, validates creator_id match, delegates to logger, returns response dict.
3. Wire `app.py` — add `/appeal` endpoint, add Flask-Limiter with `RATE_LIMIT = "10 per hour"`, connect `AuditLogger` singleton via `get_logger()`.

**Verification:**
- POST to `/submit` → POST to `/appeal` with the returned `content_id` → GET `/log`. Confirm the log entry shows `status: "under_review"` and the appeals array has one entry with the correct reason.
- POST to `/appeal` with a non-existent `content_id` → confirm 404.
- POST to `/appeal` with a mismatched `creator_id` → confirm 400.
- Manually trigger all three label variants by checking that `generate_label` is reached with scores in each zone (unit test or direct call).
- Confirm rate limiter returns 429 after 10 rapid-fire `/submit` requests in tests (use `storage_uri="memory://"` in test mode).

---

## Rate Limiting

**Limit:** 10 submissions per hour per IP address. Same limit applies to `/appeal`. `/log` and `/health` are exempt.

**Reasoning:**
- A typical active writer submits 1–5 pieces per day. 10/hour gives legitimate users headroom for testing and resubmission without restriction.
- An adversary probing classification thresholds needs many requests to map the decision boundary; 10/hour makes this expensive across time.
- The free Groq tier allows ~30 req/min. Our ceiling keeps a single bad actor from exhausting the API quota for all users.
- 10/hour is also conservative enough that if a creator hits it, it's a clear signal of unusual activity, not routine use.

---

## File Structure

```
provenance-guard/
├── app.py                    # Flask app factory + all routes
├── config.py                 # Thresholds, weights, rate limit, Groq model ID
├── requirements.txt
├── .env.example
├── planning.md               # This file
├── README.md
├── audit.db                  # Created at runtime
├── detection/
│   ├── __init__.py
│   ├── pipeline.py           # run_detection_pipeline() → PipelineResult
│   ├── llm_signal.py         # classify_with_llm() → LLMResult
│   ├── stylometric.py        # compute_stylometric_score() → StylemetricResult
│   └── confidence.py         # combine_scores() + generate_label()
├── audit/
│   ├── __init__.py
│   └── logger.py             # AuditLogger: init_db, log_decision, log_appeal, get_entry, get_all_entries
├── appeals/
│   ├── __init__.py
│   └── handler.py            # process_appeal() + AppealError
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

## Implementation Tasks (Milestones 3–5)

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

# Confidence score thresholds — asymmetric to protect against false positives
HUMAN_THRESHOLD = 0.35   # below this → high-confidence human
AI_THRESHOLD = 0.70      # above this → high-confidence AI

# Signal weights (sum to 1.0)
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
    score = max(0.0, 1.0 - ttr)
    return min(1.0, score)

def _punctuation_score(text: str) -> float:
    """Low expressive punctuation density → high AI score."""
    expressive = sum(1 for c in text if c in '!?;:—–…()')
    total = len(text)
    if total == 0:
        return 0.5
    density = expressive / total
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

- [ ] **Step 1: Write failing tests (mock Groq — do not hit real API in tests)**

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
        creator_id="user_1", content_snippet="The cat sat on the mat.",
        llm_score=0.15, stylometric_score=0.25, confidence_score=0.19,
        attribution="human", transparency_label="Likely Human-Written",
    )
    assert content_id.startswith("cnt_")

def test_get_entry_returns_logged_decision(logger):
    content_id = logger.log_decision(
        creator_id="user_1", content_snippet="Test content.",
        llm_score=0.80, stylometric_score=0.75, confidence_score=0.78,
        attribution="ai", transparency_label="Likely AI-Generated",
    )
    entry = logger.get_entry(content_id)
    assert entry is not None
    assert entry["attribution"] == "ai"
    assert entry["confidence_score"] == 0.78
    assert entry["status"] == "classified"

def test_log_appeal_updates_status_to_under_review(logger):
    content_id = logger.log_decision(
        creator_id="user_2", content_snippet="Human text.",
        llm_score=0.20, stylometric_score=0.30, confidence_score=0.24,
        attribution="human", transparency_label="Likely Human-Written",
    )
    appeal_id = logger.log_appeal(content_id=content_id, creator_id="user_2",
                                   reason="I wrote this myself.")
    assert appeal_id.startswith("app_")
    entry = logger.get_entry(content_id)
    assert entry["status"] == "under_review"
    assert len(entry["appeals"]) == 1
    assert entry["appeals"][0]["reason"] == "I wrote this myself."

def test_get_all_entries_returns_list(logger):
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
    import app as app_module
    app_module._logger_instance = None
    flask_app = app_module.create_app(testing=True)
    with flask_app.test_client() as c:
        yield c

_VALID_CONTENT = "The starling flew past the window at dusk. It circled twice, then vanished into the elm trees."

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

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASSED (25+ tests across 7 files)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_routes.py
git commit -m "feat: Flask routes — submit, appeal, log, health with rate limiting"
```

---

### Task 9: README Documentation

**Files:**
- Modify: `README.md`

The README must include: setup instructions, all three transparency label variants (verbatim text from § Transparency Label Variants above), rate limiting configuration and reasoning, and a sample audit log with ≥ 3 entries including at least one appeal, pasted directly from `GET /log` output after running the system locally.

- [ ] **Step 1: Run the app and generate real audit log entries**

```bash
python -m dotenv run python app.py  # or: flask --app app run
# Then POST 3 different texts to /submit
# Then POST one /appeal
# Then GET /log and copy the JSON output
```

- [ ] **Step 2: Write README.md with all required sections**

Sections needed: Setup, API Reference, Detection Signals (what each measures + blind spots), Confidence Score Design (thresholds table, what 0.6 means), Transparency Labels (all three verbatim), Rate Limiting (limits + reasoning), Audit Log (sample with ≥ 3 entries), Appeals (how to submit, what happens).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README — setup, label variants, rate limit reasoning, audit log sample"
```

---

## Spec Coverage Check

| Requirement | Task | Verified |
|---|---|---|
| Content Submission Endpoint returns result + score + label | Task 8 | ✅ |
| Multi-Signal Pipeline ≥ 2 distinct signals | Tasks 2, 4, 5 | ✅ LLM (semantic) + stylometric (structural) |
| Confidence Scoring — 0.51 vs 0.95 produce different labels | Task 3 | ✅ three distinct zones |
| Transparency Label — 3 variants verbatim in README | Tasks 3, 9 | ✅ |
| Appeals Workflow — reason + log + status update | Tasks 6, 7, 8 | ✅ |
| Rate Limiting with documented reasoning | Tasks 8, 9 | ✅ 10/hr |
| Audit Log structured, ≥ 3 entries visible | Tasks 6, 9 | ✅ |
| Architecture diagram in planning.md under ## Architecture | This file | ✅ |
| Verbatim label text in README (not just screenshot) | Task 9 | must verify |
| Signals' output format + combination formula documented | § Detection Signals | ✅ |
| Confidence score uncertainty explained | § Uncertainty Representation | ✅ |
| Appeals workflow: who, what, steps, reviewer view | § Appeals Workflow | ✅ |
| Anticipated edge cases (≥ 2 specific) | § Anticipated Edge Cases | ✅ 3 cases |
| AI Tool Plan for M3, M4, M5 | § AI Tool Plan | ✅ |
