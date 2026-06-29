# Provenance Guard

A Flask API that classifies creative text content as human or AI-generated, returns a plain-language transparency label with a confidence score, and handles creator appeals with a structured audit trail.

---

## Setup

### Prerequisites

- Python 3.12+
- A [Groq API key](https://console.groq.com)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### Run

```bash
python app.py
```

The server starts at `http://127.0.0.1:5000`.

### Run Tests

```bash
pytest tests/ -v
```

All 55 tests pass without a real API key (LLM calls are mocked in tests).

---

## API Reference

### `POST /submit`

Classify a piece of content as human or AI-generated.

**Request body:**

```json
{
  "content": "The text to classify...",
  "creator_id": "unique-creator-identifier"
}
```

`text` is accepted as an alias for `content`.

**Response (200):**

```json
{
  "content_id": "cnt_60a5d5ef6f4a",
  "attribution_result": "human",
  "confidence_score": 0.3292,
  "transparency_label": {
    "verdict": "Likely Human-Written",
    "confidence_display": "High",
    "detail": "Our system analyzed this content and found strong indicators of human authorship. This is an automated assessment and is not guaranteed to be correct. If you are the creator and believe this label is incorrect, you may submit an appeal."
  }
}
```

**Errors:** `400` for missing/too-short/too-long content or missing `creator_id`. `429` when rate limit exceeded.

---

### `POST /appeal`

Submit a creator appeal for a classification.

**Request body:**

```json
{
  "content_id": "cnt_6f12448ce267",
  "creator_id": "essayist-anon",
  "reason": "I wrote this essay myself for a media literacy course."
}
```

`creator_reasoning` is accepted as an alias for `reason`.

**Response (200):**

```json
{
  "appeal_id": "app_ee169460cde9",
  "content_id": "cnt_6f12448ce267",
  "status": "under_review",
  "message": "Your appeal has been received. The classification has been marked as under review."
}
```

**Errors:** `400` for missing fields or `creator_id` mismatch. `404` if `content_id` not found.

---

### `GET /log`

Retrieve all audit log entries, newest first (max 100). Rate-limit exempt.

**Response (200):**

```json
{
  "entries": [
    {
      "content_id": "cnt_...",
      "creator_id": "...",
      "content_snippet": "...",
      "timestamp": "...",
      "llm_score": 0.12,
      "stylometric_score": 0.64,
      "confidence_score": 0.33,
      "attribution": "human",
      "transparency_label": "Likely Human-Written",
      "status": "classified",
      "appeals": []
    }
  ]
}
```

---

### `GET /health`

Health check. Returns `{"status": "ok", "version": "1.0.0"}`. Rate-limit exempt.

---

## Detection Signals

### Signal 1: LLM Classification (Groq `llama-3.3-70b-versatile`)

**What it measures:** Holistic semantic and stylistic coherence — whether the text "feels" AI-generated based on sentence flow, phrasing naturalness, structural patterns, and presence or absence of personal voice. The model evaluates word choices, transitions, and register for signs of human perspective versus generated synthesis.

**Output:** A float 0.0–1.0, where 0.0 = strong human signal and 1.0 = strong AI signal. Falls back to 0.5 on API error or unparseable response.

**Weight:** 60% of the combined score.

**Blind spots:**
- Fooled by highly polished human writing (professional editors, academics).
- Cannot detect AI text that deliberately introduces stylistic errors.
- Non-deterministic: the same text may return slightly different scores across calls (mitigated by `temperature=0.1`).

---

### Signal 2: Stylometric Heuristics (Pure Python)

**What it measures:** Three structural statistics computed locally with no API calls:

| Metric | AI direction | Human direction |
|---|---|---|
| Sentence length variance (std dev of word counts) | Low variance → uniform cadence | High variance → varied rhythm |
| Type-token ratio (unique words / total words) | Low TTR → repeated vocabulary | High TTR → diverse word choice |
| Expressive punctuation density (`! ? ; : — – … ()`) | Sparse → clean, minimal | Dense → emotional, expressive |

Combined score: `(0.40 × variance_score) + (0.35 × ttr_score) + (0.25 × punctuation_score)`

**Weight:** 40% of the combined score.

**Minimum text requirement:** Fewer than 10 words or fewer than 2 sentences returns 0.5 (uncertain fallback).

**Blind spots:**
- Genre-sensitive: technical documentation, minimalist poetry, and academic prose all score in the AI direction due to structural uniformity and vocabulary constraints — regardless of actual authorship.
- Does not handle embedded lists, code, or templates.
- Not calibrated against labeled data; thresholds are empirically estimated.

---

### Why these two signals?

The LLM signal was chosen first because it's the only one capable of evaluating *meaning* — whether a sentence "sounds" like something a human would actually say, whether the register is natural, whether the transitions feel considered or mechanical. No structural heuristic can see this.

The stylometric signal was chosen as a deliberate counterbalance: it runs locally, produces the same result on the same input every time, leaves an auditable computation trail, and costs nothing per request. It also catches a specific failure mode — text that's been "humanized" by adding personality but still has the statistical fingerprint of generated output (very uniform sentence cadence, moderate vocabulary reuse).

### What I'd change for real deployment

- **Calibrate the stylometric thresholds against labeled data.** The normalization constants (std dev / 12, density × 20) are empirical estimates. A real system would tune these against a held-out labeled corpus and likely weight variance more heavily than TTR, since sentence length uniformity is a stronger AI signal than vocabulary reuse for short texts.
- **Add genre detection.** Technical documentation, academic abstracts, and minimalist poetry all produce stylometric scores in the AI direction regardless of authorship. A genre-aware system would downweight stylometric for those categories.
- **Apply confidence calibration (Platt scaling or isotonic regression).** The current confidence scores are raw weighted averages, not calibrated probabilities. Two texts with confidence 0.28 and 0.33 are both classified `human`, but 0.33 should be treated as less certain than 0.28. A calibration layer would make the scores interpretable as probabilities.
- **Use a smaller, faster model for the LLM signal.** `llama-3.3-70b-versatile` produces good scores but has non-trivial latency per call. In production I'd benchmark a 7B model on a labeled test set and use the smallest one that doesn't meaningfully degrade accuracy.

### Combining the Signals

```
confidence_score = (0.60 × llm_score) + (0.40 × stylometric_score)
```

The LLM receives more weight because it captures semantic and stylistic properties that structural heuristics cannot detect. Stylometric provides an independent, deterministic check.

---

## Confidence Score Design

### Thresholds

| Score range | Attribution | Reasoning |
|---|---|---|
| 0.00 – 0.34 | `human` | Strong evidence both signals point human |
| 0.35 – 0.70 | `uncertain` | Mixed signals — system withholds judgment |
| 0.71 – 1.00 | `ai` | Strong evidence both signals point AI |

### What does a score of 0.6 mean?

A score of 0.6 means the signals collectively lean toward AI, but not with enough certainty to make that accusation publicly. The system returns the **uncertain** label, which explicitly states the content has **not** been flagged as AI-generated.

This is a deliberate design choice: a 0.6 score is not enough to damage a creator's reputation. The system treats uncertain signals as "we don't know," not "probably AI."

### Why 0.51 and 0.95 produce materially different results

A score of 0.51 lands in the uncertain zone — the creator sees no accusation. A score of 0.95 exceeds the 0.70 AI threshold — the creator sees the high-confidence AI label and the explicit appeal invitation. These are two distinct label texts with different wording, not just a number difference.

### Asymmetric thresholds and false positives

**False positives (human labeled as AI) are considered worse than false negatives.** The wide uncertain zone (0.35–0.70) is the primary defense against wrongful AI labeling.

To reach an `ai` attribution, both signals must agree strongly. A polished human essay scoring LLM=0.65, stylometric=0.50 gives `(0.6×0.65)+(0.4×0.50) = 0.59` — uncertain, not AI. Only texts where the LLM scores ~0.85+ with reasonable stylometric agreement clear the 0.70 bar.

### How I validated the scores are meaningful

I ran five deliberately varied texts through the live pipeline during Milestone 4 calibration and inspected the scores by hand before writing the thresholds into code. The test inputs spanned three expected zones:

- Casual first-person voice + colloquialisms → LLM 0.08–0.23 → combined 0.23–0.33 (clear human range)
- Formal, hedged language resembling generated prose → LLM 0.42–0.82 → combined 0.46–0.70 (uncertain range)
- No sample cleared 0.70 on combined score in the initial calibration, which confirmed the AI threshold was working as intended: both signals must agree strongly before the AI label fires.

The key check was that different texts produced spread across the 0–1 range rather than clustering near 0.5. The two-signal design was also validated by inspecting cases where the signals *disagreed* — a formal human essay could score LLM=0.65 but stylometric=0.45, giving combined 0.57 (uncertain). Without the stylometric counterweight, that essay would have been closer to the AI zone based on LLM alone.

### Two live examples showing meaningful score variation

**Example 1 — High-confidence human** (`confidence_score: 0.2293`)

Input (casual food review by `foodblogger-kai`):
> "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably wont go back unless someone drags me there"

| Signal | Score | Why |
|---|---|---|
| LLM (`llm_score`) | 0.08 | Casual register, colloquialisms, sentence fragments, personal voice — LLM sees clear human markers |
| Stylometric (`stylometric_score`) | 0.4533 | Mixed: low variance (short sentences), but high TTR (casual vocabulary is diverse) and expressive punctuation ("?", "WAY") lower the AI signal |
| **Combined** | **0.2293** | Well below 0.35 → `human`, high confidence |

The LLM and stylometric signals agree: this reads like a person typing quickly, not a model generating evenly.

---

**Example 2 — Uncertain / lower confidence** (`confidence_score: 0.6958`)

Input (formal essay by `essayist-anon`):
> "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment."

| Signal | Score | Why |
|---|---|---|
| LLM (`llm_score`) | 0.82 | Corporate hedging language, passive constructions, "it is important to note" — LLM scores this strongly AI |
| Stylometric (`stylometric_score`) | 0.5095 | Moderate: sentence lengths are somewhat uniform, but vocabulary is not especially repetitive |
| **Combined** | **0.6958** | `(0.6×0.82)+(0.4×0.51) = 0.696` — just below the 0.70 AI threshold → `uncertain` |

The 0.47-point gap between these two examples (0.23 vs 0.70) shows the scoring produces genuinely different readings, not compressed variation. Despite the LLM scoring this text at 0.82, the system withheld the AI label — the asymmetric threshold required both signals to agree strongly before making an accusation, and stylometric returned only 0.51.

---

## Transparency Labels

These are the exact strings returned by the API in `transparency_label` for each attribution zone.

### Variant 1 — Likely Human-Written (`confidence_score < 0.35`)

```
Verdict: Likely Human-Written
Confidence: High

Our system analyzed this content and found strong indicators of human authorship.
This is an automated assessment and is not guaranteed to be correct.
If you are the creator and believe this label is incorrect, you may submit an appeal.
```

### Variant 2 — Authorship Uncertain (`0.35 ≤ confidence_score ≤ 0.70`)

```
Verdict: Authorship Uncertain
Confidence: Low — this content has not been labeled as AI-generated.

Our system detected mixed signals and cannot determine authorship with confidence.
This content has not been flagged as AI-generated.
If you are the creator and disagree with this assessment, you may submit an appeal.
```

### Variant 3 — Likely AI-Generated (`confidence_score > 0.70`)

```
Verdict: Likely AI-Generated
Confidence: High

Our system found strong indicators that this content may have been AI-generated.
This is an automated assessment and may be incorrect.
If you are the creator and believe this label is incorrect, you may submit an appeal.
```

**Design rationale:** All three variants use hedged language ("may," "not guaranteed") — the system does not claim certainty even at high confidence. All three include the appeal path. The uncertain variant leads with what the system did *not* do ("has not been flagged as AI-generated") because the uncertain-zone creator's primary concern is whether they've been accused.

---

## Rate Limiting

**Limit:** 10 requests per hour per IP address, applied to `/submit` and `/appeal`. `/log` and `/health` are exempt.

**Reasoning:**

- A typical active writer submits 1–5 pieces per day. 10/hour gives legitimate users headroom for testing and resubmission without restriction.
- An adversary probing classification thresholds needs many requests to map the decision boundary. 10/hour makes this expensive across time.
- The free Groq tier allows ~30 req/min. The ceiling keeps a single bad actor from exhausting the API quota for all users.
- 10/hour is conservative enough that hitting it signals unusual activity, not routine use.

**429 response when rate limit exceeded:**

```json
{
  "error": "10 per 1 hour",
  "description": "Too Many Requests"
}
```

---

## Audit Log

The `GET /log` endpoint returns all decisions and appeals. Below is a real sample captured after 3 submissions and 1 appeal against the running server:

```json
{
  "entries": [
    {
      "content_id": "cnt_748d76587dd2",
      "creator_id": "foodblogger-kai",
      "content_snippet": "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably wont go back unless someone drags me there",
      "timestamp": "2026-06-29T17:06:58.091228+07:00",
      "llm_score": 0.08,
      "stylometric_score": 0.4533,
      "confidence_score": 0.2293,
      "attribution": "human",
      "transparency_label": "Likely Human-Written",
      "status": "classified",
      "appeals": []
    },
    {
      "content_id": "cnt_6f12448ce267",
      "creator_id": "essayist-anon",
      "content_snippet": "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment.",
      "timestamp": "2026-06-29T17:06:55.693287+07:00",
      "llm_score": 0.82,
      "stylometric_score": 0.5095,
      "confidence_score": 0.6958,
      "attribution": "uncertain",
      "transparency_label": "Authorship Uncertain",
      "status": "under_review",
      "appeals": [
        {
          "appeal_id": "app_ee169460cde9",
          "creator_id": "essayist-anon",
          "reason": "I wrote this essay myself for a media literacy course. My academic writing style may read as formal, but it is entirely my own work.",
          "timestamp": "2026-06-29T17:09:18.332594+07:00"
        }
      ]
    },
    {
      "content_id": "cnt_60a5d5ef6f4a",
      "creator_id": "poet-maria",
      "content_snippet": "The sun dipped below the horizon, painting the sky in hues of amber and rose. I sat on the porch, coffee in hand, watching the neighborhood slowly go quiet. Someone was mowing a lawn three houses down — that particular Saturday sound.",
      "timestamp": "2026-06-29T17:06:53.359105+07:00",
      "llm_score": 0.12,
      "stylometric_score": 0.6431,
      "confidence_score": 0.3292,
      "attribution": "human",
      "transparency_label": "Likely Human-Written",
      "status": "classified",
      "appeals": []
    }
  ]
}
```

### Reading the log

Each entry shows:
- **`llm_score`** and **`stylometric_score`**: the raw outputs of each detection signal (0–1, higher = more AI-like)
- **`confidence_score`**: the weighted combination `(0.60 × llm + 0.40 × stylo)`
- **`attribution`**: the threshold-gated decision (`human` / `uncertain` / `ai`)
- **`status`**: `classified` (initial) or `under_review` (appeal filed)
- **`appeals`**: array of appeal records, each with its own `appeal_id`, creator-provided `reason`, and timestamp

In the sample above, entry `cnt_6f12448ce267` scored LLM=0.82 (formal corp-speak style triggered a strong AI signal) but stylometric=0.51 (moderate — uniform structure, not dramatically so). Combined: `(0.6×0.82)+(0.4×0.51) = 0.696`, which lands just below the 0.70 AI threshold — resulting in `uncertain` rather than `ai`. The asymmetric threshold design protected the creator from an AI label. They then filed an appeal, moving status to `under_review`.

---

## Appeals Workflow

**Who can appeal:** Any creator who submitted content, identified by `creator_id`.

**Required fields:**
- `content_id`: returned by the original `/submit` call
- `creator_id`: must match the original submission
- `reason` (or `creator_reasoning`): free-text explanation, non-empty

**What happens:**
1. System looks up the original decision in SQLite by `content_id`
2. Validates `creator_id` matches the original submitter (400 on mismatch)
3. Validates `reason` is non-empty after stripping whitespace (400 on empty)
4. Generates a unique `appeal_id` (`app_` + 12 hex chars)
5. Appends the appeal record to the entry's `appeals` JSON array
6. Updates `status` from `classified` to `under_review`
7. Returns `{appeal_id, content_id, status: "under_review", message}`

**No automated re-classification occurs.** The status change signals to a human reviewer that this entry needs attention. A reviewer filtering `GET /log` by `status = "under_review"` sees the original classification, both signal scores, and the creator's stated reason — enough context to make a manual decision.

---

## Known Limitations

### Academic and technical prose — the false-positive risk

The most likely failure mode is a human writer with a formal, structured style being classified as `uncertain` (or in an aggressive threshold configuration, as `ai`). A developer writing a README, a student writing a philosophy essay, or a policy analyst writing a report will all produce text with:

- **Low sentence length variance** — structured prose uses consistent sentence cadence by convention
- **Low type-token ratio** — domain vocabulary repeats by necessity ("endpoint," "parameter," "returns" in API docs; "therefore," "however," "the argument" in essays)
- **Minimal expressive punctuation** — formal writing avoids `!`, `—`, `()`

All three stylometric sub-scores push toward AI, even when the author is human. The LLM signal partially compensates — it can recognize that the specific content feels argued rather than generated — but a strongly structured essay can still score LLM=0.65+, pushing the combined score into the uncertain zone.

This isn't a data or model problem. It's a property of the signals: stylometric heuristics were designed around the idea that AI text is *uniformly smooth*, but so is well-structured human writing. Without per-genre calibration or a separate signal for structural intent vs. generative uniformity, the system cannot distinguish between the two.

### Short or unconventional creative writing

Texts under 10 words receive a flat 0.5 (uncertain) fallback from the stylometric signal, and texts under ~50 words have too little signal for reliable TTR or variance measurement. This means flash fiction, short poems, and social media posts that happen to be exactly at the minimum-word boundary will be scored largely on the LLM signal alone — which is less stable for very short texts.

---

## Spec Reflection

### One way the spec guided the implementation well

The spec's insistence that false positives are worse than false negatives was written down before any code existed, and it directly shaped every threshold decision. Without that explicit priority, the natural engineering instinct would be to split the thresholds symmetrically (say, 0.33/0.67), which would label more content as AI and feel "more decisive." Having the priority stated upfront made the asymmetric zone (0.35–0.70) feel correct rather than conservative — I had a written reason to make the AI threshold harder to reach than the human threshold.

### One way the implementation diverged from the spec

The spec described two "equally weighted" signals that would each contribute meaningfully to the final score. In practice, the stylometric signal clusters in a much narrower range (roughly 0.44–0.65 across diverse texts) than the LLM signal (0.08–0.82 in the same tests). The LLM dominates the combined score in most cases — stylometric rarely shifts the attribution zone on its own.

This isn't a bug, but it means the system functions more like "LLM classification with a stylometric sanity check" than "two signals of comparable discrimination power." If I were speccing this again, I'd either accept that framing explicitly, or I'd design the stylometric signal to be a binary modifier (e.g., "if stylometric is strongly human and LLM is uncertain, cap at uncertain") rather than a linear weight in the average.

---

## AI Usage

### Instance 1: Generating the Groq API integration

I directed the AI tool to implement `classify_with_llm()` with a specific contract: accept a string, return a `LLMResult` dataclass with `score`, `raw_response`, `parse_error`, and `api_error` fields; clamp the output to [0, 1]; return `score=0.5` on any failure rather than raising.

The AI produced a working implementation, but the initial system prompt was lengthy — it explained the scoring scale in detail and asked the model to "consider multiple factors." I revised it to the minimal form now in the code: just the 0.0/1.0/0.5 mapping and the instruction to return only the number. The reason was that a longer prompt introduced variability — the model would sometimes preface its answer with a phrase before the number, causing parse failures. The tighter prompt produced more consistent single-number responses.

### Instance 2: Stylometric sub-score weighting

When implementing `compute_stylometric_score()`, I directed the AI to combine the three sub-scores (variance, TTR, punctuation) into a single float. The initial suggestion used equal weights — one-third each.

I overrode this with `(0.40 × variance + 0.35 × ttr + 0.25 × punctuation)` based on my own reasoning: sentence length variance is the most consistent AI fingerprint because LLM generation has a statistically narrow sentence-length distribution that persists across prompts and styles. TTR is second because vocabulary reuse is also a generation artifact, though genre-dependent. Punctuation is weakest because it's almost entirely genre-determined and adds noise more than signal for most content types. Giving punctuation a full one-third would have over-penalized formal human writing with minimal punctuation, increasing false positives for exactly the population we most need to protect.

---

## Architecture

### Submission path: from input to transparency label

1. **Rate Limiter** — checks the requesting IP against the 10/hour counter. Returns 429 if exceeded.
2. **Request Validator** — confirms `content` is present, 10–10,000 words, and `creator_id` is non-empty. Returns 400 if not.
3. **LLM Classifier** (`detect/llm_signal.py`) — sends the text to Groq (`llama-3.3-70b-versatile`) with a system prompt asking for a single float 0–1. Returns `llm_score`. Falls back to 0.5 on API error or parse failure.
4. **Stylometric Analyzer** (`detect/stylometric.py`) — computes sentence length variance, type-token ratio, and expressive punctuation density locally, returns `stylometric_score` as a weighted combination. No API call.
5. **Confidence Scorer** (`detect/confidence.py`) — combines both scores: `(0.60 × llm_score) + (0.40 × stylometric_score)`. Applies asymmetric thresholds: `< 0.35 → human`, `0.35–0.70 → uncertain`, `> 0.70 → ai`.
6. **Label Generator** — maps the attribution + confidence score to one of three transparency label texts (see [Transparency Labels](#transparency-labels)).
7. **Audit Logger** — writes the full decision record (both signal scores, confidence, attribution, label, timestamp) to SQLite.
8. **Response** — returns `content_id`, `attribution_result`, `confidence_score`, and `transparency_label` as JSON.

The appeal path branches from step 7: a POST to `/appeal` looks up the existing decision, validates `creator_id` and `reason`, appends an appeal record to the entry, sets `status = "under_review"`, and returns `appeal_id`.

For the full ASCII flow diagrams with annotated data payloads at each step, see [`planning.md`](planning.md) under `## Architecture`.
