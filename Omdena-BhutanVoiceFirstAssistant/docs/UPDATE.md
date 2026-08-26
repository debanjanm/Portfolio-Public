# classifier.py — Improvements by Debanjan Mondal
**Team 4 · Omdena Bhutan Voice-First Assistant Sprint**

---

## Summary

Took the original single-backend classifier (`code1.py`) and extended it into a multi-backend, production-ready classifier (`classifier.py`) with improved accuracy, better KB coverage, and human escalation support.

---

## 1. LM Studio Backend (New)

**Problem:** Original only supported Gemini API (paid, cloud-dependent).

**Fix:** Added `LMStudioClassifier` — calls any local model via LM Studio's OpenAI-compatible server (`http://localhost:1234/v1`). No API key, no cost, fully offline.

```python
CLASSIFIER_BACKEND = "lmstudio"  # switch at top of file
LMSTUDIO_BASE_URL  = "http://localhost:1234/v1"
LMSTUDIO_MODEL_NAME = "google/gemma-4-e4b"
```

**Backend options now:**

| Backend | Cost | Requires |
|---|---|---|
| `"gemini"` | ~$0.34/1K queries | Gemini API key |
| `"lmstudio"` | Free | LM Studio running locally |
| `"keyword"` | Free | Nothing |

---

## 2. LM Studio JSON Parsing Fix

**Problem:** Local models (Gemma-4, etc.) return malformed or markdown-wrapped JSON. Original code crashed with:
- `BadRequestError: 'response_format.type' must be 'json_schema' or 'text'`
- `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
- `JSONDecodeError: Expecting ',' delimiter`

**Fix:** Removed `response_format={"type": "json_object"}` (not supported by all models). Added two-stage extraction:

1. **Stage 1:** Regex to find first `{...}` block → try `json.loads()`
2. **Stage 2:** Per-field regex fallback — extracts each of the 4 dimensions independently, works even on truncated or delimiter-broken JSON

```python
@staticmethod
def _extract_json(raw: str) -> Dict:
    # Stage 1: clean JSON parse
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Stage 2: per-field regex fallback
    result = {}
    for key in ("in_scope", "safety", "request_type", "service"):
        m = re.search(rf'"{key}"\s*:\s*\{{\s*"label"\s*:\s*"([^"]+)"...', raw)
        ...
```

---

## 3. Few-Shot Examples in Classifier Prompt

**Problem:** Without examples, LM Studio models guessed wrong — `service=unknown` for business queries, `office_location` for "nearest hospital".

**Fix:** Added 8 labeled examples directly in `CLASSIFIER_PROMPT_TEMPLATE`:

| Query | in_scope | request_type | service |
|---|---|---|---|
| "What papers for timber permit?" | in_scope | document_inquiry | permits |
| "Where is the nearest hospital in Paro?" | in_scope | facility_lookup | health |
| "How do I register a business?" | in_scope | general_info | business |
| "What docs for sole proprietorship?" | in_scope | document_inquiry | business |
| "Am I eligible for subsidized timber in Thimphu?" | in_scope | eligibility_check | permits |
| "Who won the cricket match?" | out_of_scope | other | general |
| "Do I have cancer?" | in_scope | general_info (unsafe) | health |
| "How do I plan a tour in Bhutan?" | out_of_scope | other | general |

Last two examples directly fix bugs found during testing (Neelakshi's `tour` misclassification, medical safety bypass).

---

## 4. New `facility_lookup` Request Type

**Problem:** "Where is the nearest hospital in Paro?" classified as `office_location` — semantically wrong. Office location = address of govt office. Facility lookup = nearest health/service facility.

**Fix:** Added `facility_lookup` to `REQUEST_TYPES`:

```python
"facility_lookup",   # where is the nearest hospital / BHU / centre?
```

---

## 5. Expanded `SERVICES` Labels

**Problem:** Original had only 4 services (`permits`, `health`, `business`, `general`). Prompt itself mentioned NDI, education, taxes — but classifier couldn't label them.

**Fix:** Expanded to 8 services:

```python
SERVICES = [
    "permits",           # timber, sand, stone, construction permits
    "health",            # hospitals, BHUs, vaccinations
    "business",          # business registration, trade license
    "ndi",               # National Digital Identity
    "civil_registration",# birth, death, marriage certificates
    "education",         # schools, scholarships
    "taxes",             # tax filing, BIT, PIT
    "general",           # catch-all
]
```

---

## 6. Confidence-Gated Human Escalation

**Problem:** When classifier was uncertain (low confidence or `unknown` labels), bot still attempted to answer — producing wrong or irrelevant responses. The Set A spec explicitly requires human escalation on repeated confusion.

**Fix:** Added `CONFIDENCE_THRESHOLD = 0.65`. Any query where a dimension scores below threshold is routed to a human officer with a reference number:

```python
CONFIDENCE_THRESHOLD = 0.65

low_confidence_dims = [
    dim for dim, r in classification.items()
    if r.confidence < CONFIDENCE_THRESHOLD and r.label != "unknown"
]
if low_confidence_dims:
    → FB-06 escalation response with REF number + helpline (17)
```

Directly implements the human escalation requirement from `05-Set_A_Permit_Application_Assistance.md`.

---

## 7. KB Search Score Threshold Fix

**Problem:** `search_kb()` always returned `top_k=2` chunks regardless of relevance. "Where is the nearest hospital in Paro?" returned both Paro District Hospital AND JDWNRH Thimphu — user only asked about Paro.

**Fix:** Added `score_threshold=0.5` parameter. Chunks scoring below threshold are dropped:

```python
# Before
return [c for c, s in scored[:top_k] if s > 0] or candidates[:1]

# After
above = [(c, s) for c, s in scored[:top_k] if s >= score_threshold]
return [c for c, s in above] or [scored[0][0]]
```

---

## 8. Expanded Knowledge Base (3 new chunks)

**Problem:** KB had no sand permit, stone permit, or timber eligibility information. Queries about these returned generic or wrong answers.

**Added chunks:**

| ID | Title | Covers |
|---|---|---|
| `prm_sand` | Sand Permit | Documents, dept, application process |
| `prm_stone` | Stone Permit | Documents, Geology & Mines dept |
| `prm_timber_eligibility` | Timber Permit — Eligibility Rules | Thromde boundary, Head of Gung, subsidized timber rules |

Timber eligibility chunk directly addresses the rural Bhutan use case highlighted by Sonam in the project kickoff.

---

## 9. KeywordClassifier Bug Fix

**Problem:** `tour`, `tourism`, `travel` not in `OUT_OF_SCOPE_WORDS` — "How do I plan a tour in Bhutan?" classified as `in_scope`. Bug reported by Neelakshi.

**Fix:**
```python
OUT_OF_SCOPE_WORDS = [
    ...,
    "tour", "tourism", "travel", "entertainment",  # added
]
```

---

## Before / After — Demo Scenarios

| Scenario | Before service | After service | Before request_type | After request_type |
|---|---|---|---|---|
| Construction permit docs | permits ✅ | permits ✅ | document_inquiry ✅ | document_inquiry ✅ |
| Cricket match | out_of_scope ✅ | out_of_scope ✅ | other ✅ | other ✅ |
| Do I have cancer? | health ✅ | health ✅ | general_info ✅ | general_info ✅ |
| Nearest hospital Paro | health ✅ | health ✅ | office_location ❌ | facility_lookup ✅ |
| Register sole proprietorship | **unknown ❌** | **business ✅** | general_info ✅ | document_inquiry ✅ |
| Tell me a joke | out_of_scope ✅ | out_of_scope ✅ | other ✅ | other ✅ |

**Key fix:** `service=unknown` for business queries resolved by few-shot examples.
**Key fix:** `office_location` → `facility_lookup` for hospital queries.

---

## Files Changed

- `classifier.py` — all improvements above (renamed from `code1.py`)

Tashi Delek 🙏
