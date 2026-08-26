"""
═══════════════════════════════════════════════════════════════════════
  BHUTAN VOICE-FIRST PUBLIC SERVICE ASSISTANT
  Conversation + Multi-Backend Request Classifier (Gemini / LM Studio / Keyword)
  Omdena Sprint · Team 4 deliverable
───────────────────────────────────────────────────────────────────────

  WHAT THIS SCRIPT DOES
  ─────────────────────
  Same conversation output style as Team 3's prototype, PLUS a
  classification table for every user turn showing four dimensions
  with confidence scores:

      ┌──────────────────┬──────────────────────┬────────────┐
      │ In-scope         │ in_scope             │ 0.97       │
      │ Safety           │ safe                 │ 0.99       │
      │ Request type     │ document_inquiry     │ 0.94       │
      │ Service          │ permits              │ 0.98       │
      └──────────────────┴──────────────────────┴────────────┘

  HOW TO RUN IN COLAB
  ───────────────────
    Cell 1:  !pip install -q google-generativeai openai
    Cell 2:  paste this whole file, set GEMINI_API_KEY or LMSTUDIO_BASE_URL
             below, then set CLASSIFIER_BACKEND to "gemini", "lmstudio",
             or "keyword". Run.
    Cell 3:  run_demo()         # runs the 6 sample scenarios
             run_interactive()  # type your own messages
═══════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════
# 🔑  STEP 1a — GEMINI API KEY (only needed if CLASSIFIER_BACKEND = "gemini")
#     Set the GEMINI_API_KEY environment variable — do not hardcode it here.
# ═══════════════════════════════════════════════════════════════════
import os
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-2.5-flash"   # ← swap for any Gemini model

# ═══════════════════════════════════════════════════════════════════
# 🖥  STEP 1b — LM STUDIO SETTINGS (local, no API key needed)
# ═══════════════════════════════════════════════════════════════════
# Setup:
#   1. Open LM Studio → load any instruction-following model
#   2. Go to Local Server tab → Start Server (default port 1234)
#   3. Set CLASSIFIER_BACKEND = "lmstudio" below and run.
LMSTUDIO_BASE_URL   = "http://localhost:1234/v1"
LMSTUDIO_MODEL_NAME = "google/gemma-4-e4b"   # LM Studio ignores this; any string works

# ═══════════════════════════════════════════════════════════════════
# 🔀  STEP 1c — CHOOSE YOUR BACKEND
# ═══════════════════════════════════════════════════════════════════
# "gemini"   → Gemini API  (needs GEMINI_API_KEY, costs ~$0.34/1K queries)
# "lmstudio" → Local LM Studio server (free, no key, runs on your machine)
# "keyword"  → Pure keyword matching, no model at all (instant, offline)
CLASSIFIER_BACKEND = "lmstudio"   # ← change this to switch backends


# ═══════════════════════════════════════════════════════════════════
# 📦  IMPORTS
# ═══════════════════════════════════════════════════════════════════
import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI as _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# ✏️  EDIT ME #1 — CLASSIFIER PROMPT
# ═══════════════════════════════════════════════════════════════════
# Shared by GeminiClassifier and LMStudioClassifier.
# Keep the JSON schema at the bottom intact — the parser depends
# on the four keys: in_scope / safety / request_type / service.
# ═══════════════════════════════════════════════════════════════════
CLASSIFIER_PROMPT_TEMPLATE = """You are a request classifier for a Bhutan public-service voice assistant. \
Classify the user's request along 4 dimensions and return a JSON object.

USER REQUEST: "{user_text}"

DIMENSIONS

1) in_scope — is this about Bhutanese public services?
   • "in_scope"     — permits, health services, business registration, NDI,
                      civil registration, taxes, education, etc.
   • "out_of_scope" — sports, entertainment, weather, jokes, stock prices,
                      general chat, tourism planning, anything unrelated to govt services.

2) safety — is this safe to answer with general/procedural information?
   • "safe"   — information, navigation, document checklists, office locations,
                eligibility queries, facility lookups.
   • "unsafe" — requests for medical diagnosis, treatment, prescriptions,
                specific legal advice, or financial advice.

3) request_type — what kind of help is the user asking for? Pick ONE from:
   {request_types}

4) service — which government domain does this belong to? Pick ONE from:
   {services}

EXAMPLES (use these to calibrate your output)

Q: "What papers do I need for a timber permit?"
A: {{"in_scope": {{"label": "in_scope", "confidence": 0.98}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "document_inquiry", "confidence": 0.97}}, "service": {{"label": "permits", "confidence": 0.98}}}}

Q: "Where is the nearest hospital in Paro?"
A: {{"in_scope": {{"label": "in_scope", "confidence": 0.97}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "facility_lookup", "confidence": 0.95}}, "service": {{"label": "health", "confidence": 0.97}}}}

Q: "How do I register a business in Bhutan?"
A: {{"in_scope": {{"label": "in_scope", "confidence": 0.98}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "general_info", "confidence": 0.90}}, "service": {{"label": "business", "confidence": 0.97}}}}

Q: "What documents do I need to register a sole proprietorship?"
A: {{"in_scope": {{"label": "in_scope", "confidence": 0.98}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "document_inquiry", "confidence": 0.96}}, "service": {{"label": "business", "confidence": 0.97}}}}

Q: "Am I eligible for subsidized timber if I live in Thimphu Thromde?"
A: {{"in_scope": {{"label": "in_scope", "confidence": 0.97}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "eligibility_check", "confidence": 0.96}}, "service": {{"label": "permits", "confidence": 0.96}}}}

Q: "Who won the cricket match yesterday?"
A: {{"in_scope": {{"label": "out_of_scope", "confidence": 0.99}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "other", "confidence": 0.99}}, "service": {{"label": "general", "confidence": 0.99}}}}

Q: "My stomach hurts badly, do I have cancer?"
A: {{"in_scope": {{"label": "in_scope", "confidence": 0.80}}, "safety": {{"label": "unsafe", "confidence": 0.97}}, "request_type": {{"label": "general_info", "confidence": 0.70}}, "service": {{"label": "health", "confidence": 0.85}}}}

Q: "How do I plan a tour in Bhutan?"
A: {{"in_scope": {{"label": "out_of_scope", "confidence": 0.95}}, "safety": {{"label": "safe", "confidence": 0.99}}, "request_type": {{"label": "other", "confidence": 0.95}}, "service": {{"label": "general", "confidence": 0.95}}}}

OUTPUT FORMAT — return ONLY this JSON object, nothing else:
{{
  "in_scope":     {{"label": "...", "confidence": 0.0}},
  "safety":       {{"label": "...", "confidence": 0.0}},
  "request_type": {{"label": "...", "confidence": 0.0}},
  "service":      {{"label": "...", "confidence": 0.0}}
}}

Confidence is a number between 0.0 and 1.0 expressing how sure you are."""


# ═══════════════════════════════════════════════════════════════════
# ✏️  EDIT ME #2 — LABEL CATEGORIES
# ═══════════════════════════════════════════════════════════════════
REQUEST_TYPES = [
    "document_inquiry",     # what documents do I need?
    "status_check",         # check application status
    "office_location",      # where is the office / address?
    "facility_lookup",      # where is the nearest hospital / BHU / centre?
    "application_start",    # start a new application
    "eligibility_check",    # am I eligible?
    "general_info",         # explain / general question
    "other",
]

SERVICES = [
    "permits",              # timber, sand, stone, construction permits
    "health",               # hospitals, BHUs, vaccinations
    "business",             # business registration, trade license
    "ndi",                  # National Digital Identity
    "civil_registration",   # birth, death, marriage certificates
    "education",            # schools, scholarships
    "taxes",                # tax filing, BIT, PIT
    "general",              # catch-all
]

# ═══════════════════════════════════════════════════════════════════
# ⚠️  CONFIDENCE THRESHOLD — queries below this get escalated to human
# ═══════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.65   # lower = more tolerant, higher = stricter escalation


# ═══════════════════════════════════════════════════════════════════
# 🔌  CLASSIFIER INTERFACE
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ClassificationResult:
    label: str
    confidence: float


class BaseClassifier:
    """Implement classify() in your subclass."""
    def classify(self, user_text: str) -> Dict[str, ClassificationResult]:
        raise NotImplementedError


class GeminiClassifier(BaseClassifier):
    """One Gemini API call returns all 4 classifications as JSON."""

    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL_NAME):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed.\n"
                "Run: !pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"},
        )

    def classify(self, user_text: str) -> Dict[str, ClassificationResult]:
        prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
            user_text=user_text.replace('"', "'"),
            request_types=", ".join(REQUEST_TYPES),
            services=", ".join(SERVICES),
        )
        try:
            response = self.model.generate_content(prompt)
            data = json.loads(response.text)
            return self._parse(data)
        except Exception as e:
            print(f"[Gemini classifier error: {type(e).__name__}: {e}]")
            return self._unknown()

    def _parse(self, data: Dict) -> Dict[str, ClassificationResult]:
        out: Dict[str, ClassificationResult] = {}
        for key in ("in_scope", "safety", "request_type", "service"):
            entry = data.get(key, {})
            if not isinstance(entry, dict):
                entry = {}
            out[key] = ClassificationResult(
                label=str(entry.get("label", "unknown")),
                confidence=float(entry.get("confidence", 0.0)),
            )
        return out

    @staticmethod
    def _unknown() -> Dict[str, ClassificationResult]:
        return {k: ClassificationResult("unknown", 0.0)
                for k in ("in_scope", "safety", "request_type", "service")}


class LMStudioClassifier(BaseClassifier):
    """
    Calls a local LM Studio server via its OpenAI-compatible API.
    No API key required — runs entirely on your machine.

    Setup:
      1. Open LM Studio → load any instruction-following model
      2. Go to Local Server tab → Start Server (default port 1234)
      3. Set CLASSIFIER_BACKEND = "lmstudio" at the top and run.
    """

    def __init__(self, base_url: str = LMSTUDIO_BASE_URL,
                 model: str = LMSTUDIO_MODEL_NAME):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed.\n"
                "Run: !pip install openai"
            )
        self.client = _OpenAI(base_url=base_url, api_key="lm-studio")
        self.model  = model

    @staticmethod
    def _extract_json(raw: str) -> Dict:
        """
        Two-stage extraction for local models that emit malformed JSON:
        1. Try to parse the first {...} block directly.
        2. Fall back to per-field regex — works even on truncated/invalid JSON.
        """
        import re
        if not raw or not raw.strip():
            raise ValueError("Model returned empty response")

        # Stage 1: try clean JSON parse
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Stage 2: extract each field with regex (tolerates bad delimiters / truncation)
        result: Dict = {}
        for key in ("in_scope", "safety", "request_type", "service"):
            m = re.search(
                rf'"{key}"\s*:\s*\{{\s*"label"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([0-9.]+)',
                raw,
            )
            if m:
                result[key] = {"label": m.group(1), "confidence": float(m.group(2))}
            else:
                result[key] = {"label": "unknown", "confidence": 0.0}
        return result

    def classify(self, user_text: str) -> Dict[str, ClassificationResult]:
        prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
            user_text=user_text.replace('"', "'"),
            request_types=", ".join(REQUEST_TYPES),
            services=", ".join(SERVICES),
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            data = self._extract_json(raw)
            return self._parse(data)
        except Exception as e:
            print(f"[LMStudio classifier error: {type(e).__name__}: {e}]")
            return self._unknown()

    def _parse(self, data: Dict) -> Dict[str, ClassificationResult]:
        out: Dict[str, ClassificationResult] = {}
        for key in ("in_scope", "safety", "request_type", "service"):
            entry = data.get(key, {})
            if not isinstance(entry, dict):
                entry = {}
            out[key] = ClassificationResult(
                label=str(entry.get("label", "unknown")),
                confidence=float(entry.get("confidence", 0.0)),
            )
        return out

    @staticmethod
    def _unknown() -> Dict[str, ClassificationResult]:
        return {k: ClassificationResult("unknown", 0.0)
                for k in ("in_scope", "safety", "request_type", "service")}


class KeywordClassifier(BaseClassifier):
    """
    No-API fallback. Pure keyword matching — no model, no internet needed.
    """

    OUT_OF_SCOPE_WORDS = [
        "cricket", "football", "soccer", "movie", "joke", "weather",
        "stock", "forex", "election", "recipe", "song", "celebrity",
        "tour", "tourism", "travel", "entertainment",
    ]
    UNSAFE_WORDS = [
        "diagnose", "diagnosis", "cancer", "tumour", "tumor",
        "treatment", "prescription", "should i take",
        "what disease", "am i sick", "do i have",
    ]
    SERVICE_KEYWORDS = {
        "permits":  ["permit", "license", "licence", "construction",
                     "timber", "stone", "sand"],
        "health":   ["hospital", "doctor", "clinic", "bhu", "vaccin",
                     "medic", "health", "sick", "pain"],
        "business": ["business", "register", "registration", "company",
                     "proprietor", "trade"],
    }
    TYPE_KEYWORDS = {
        "document_inquiry":  ["document", "papers", "what do i need",
                              "checklist", "requirement"],
        "status_check":      ["status", "track", "reference",
                              "where is my", "progress"],
        "office_location":   ["office", "where", "address", "location",
                              "directions"],
        "application_start": ["apply", "start", "submit", "register",
                              "begin", "new"],
        "eligibility_check": ["eligible", "can i", "qualify", "allowed"],
    }

    def classify(self, text: str) -> Dict[str, ClassificationResult]:
        t = text.lower()
        return {
            "in_scope":     self._in_scope(t),
            "safety":       self._safety(t),
            "service":      self._best_match(t, self.SERVICE_KEYWORDS, "general"),
            "request_type": self._best_match(t, self.TYPE_KEYWORDS, "general_info"),
        }

    def _in_scope(self, t: str) -> ClassificationResult:
        if any(w in t for w in self.OUT_OF_SCOPE_WORDS):
            return ClassificationResult("out_of_scope", 0.8)
        return ClassificationResult("in_scope", 0.7)

    def _safety(self, t: str) -> ClassificationResult:
        if any(w in t for w in self.UNSAFE_WORDS):
            return ClassificationResult("unsafe", 0.8)
        return ClassificationResult("safe", 0.8)

    @staticmethod
    def _best_match(t: str, mapping: Dict[str, List[str]],
                    default: str) -> ClassificationResult:
        scores = {k: sum(1 for w in v if w in t) for k, v in mapping.items()}
        if not any(scores.values()):
            return ClassificationResult(default, 0.4)
        best = max(scores, key=scores.get)
        return ClassificationResult(best, min(1.0, scores[best] / 2.0))


# ═══════════════════════════════════════════════════════════════════
# 📊  TABLE FORMATTER
# ═══════════════════════════════════════════════════════════════════
def format_classification_table(c: Dict[str, ClassificationResult]) -> str:
    rows = [
        ("In-scope",     c["in_scope"]),
        ("Safety",       c["safety"]),
        ("Request type", c["request_type"]),
        ("Service",      c["service"]),
    ]
    out = [
        "┌──────────────────┬──────────────────────┬────────────┐",
        "│ Dimension        │ Classification       │ Confidence │",
        "├──────────────────┼──────────────────────┼────────────┤",
    ]
    for name, r in rows:
        label = r.label if len(r.label) <= 20 else r.label[:17] + "..."
        out.append(f"│ {name:<16} │ {label:<20} │ {r.confidence:>10.2f} │")
    out.append("└──────────────────┴──────────────────────┴────────────┘")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════
#                       ━━━━  ENGINE BELOW  ━━━━
# ═══════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────
# Knowledge base
# ───────────────────────────────────────────────────────────────────
@dataclass
class KnowledgeChunk:
    id: str
    service: str
    title: str
    text: str
    keywords: Tuple[str, ...]
    source: str


KNOWLEDGE_BASE: List[KnowledgeChunk] = [
    # ── PERMITS ─────────────────────────────────────────────────────
    KnowledgeChunk(
        id="prm_construction",
        service="permits",
        title="Construction Permit — Thimphu",
        text=("For a construction permit in Thimphu you will need:\n"
              "  • CID card (original + photocopy)\n"
              "  • Land Thram (land ownership document)\n"
              "  • No Objection Certificate (NOC) from the landlord\n"
              "  • Completed application form from the dzongkhag office\n"
              "Submit at the Department of Urban Development, "
              "Thadrak, Thimphu (Mon–Fri, 9am–5pm)."),
        keywords=("construction", "permit", "thimphu", "document"),
        source="bhutan_flow_docs/permits",
    ),
    KnowledgeChunk(
        id="prm_timber",
        service="permits",
        title="Timber Permit",
        text=("For a timber permit you will need:\n"
              "  • CID card\n"
              "  • Land Thram showing forest plot\n"
              "  • Approval from gewog forest officer\n"
              "  • Completed timber permit form\n"
              "Apply at your local Range Office under the Department of Forests."),
        keywords=("timber", "permit", "forest", "wood"),
        source="bhutan_flow_docs/permits",
    ),
    KnowledgeChunk(
        id="prm_office",
        service="permits",
        title="Permit Office — Thimphu",
        text=("Permit office in Thimphu: Department of Urban Development, "
              "Thadrak, Thimphu. Phone: +975-2-xxx-xxxx. "
              "Open Monday to Friday, 9am to 5pm. "
              "Bring original documents and two photocopies."),
        keywords=("office", "location", "where", "address", "thimphu"),
        source="bhutan_flow_docs/permits",
    ),

    # ── HEALTH ──────────────────────────────────────────────────────
    KnowledgeChunk(
        id="hlt_facility_thimphu",
        service="health",
        title="JDWNRH — Thimphu",
        text=("Nearest hospital in Thimphu: Jigme Dorji Wangchuck National "
              "Referral Hospital (JDWNRH), Gongphel Lam, Thimphu. "
              "Phone: 112 or +975-2-xxx-xxxx. "
              "OPD hours: Mon–Sat 8am–4pm. Emergency: 24 hours."),
        keywords=("hospital", "thimphu", "facility", "nearest"),
        source="bhutan_flow_docs/health",
    ),
    KnowledgeChunk(
        id="hlt_facility_paro",
        service="health",
        title="Paro District Hospital",
        text=("Nearest hospital in Paro: Paro District Hospital, Paro town. "
              "Phone: +975-8-xxx-xxxx. Open Mon–Sat, 8am–4pm."),
        keywords=("hospital", "paro", "facility"),
        source="bhutan_flow_docs/health",
    ),
    KnowledgeChunk(
        id="hlt_bhu",
        service="health",
        title="Basic Health Units (BHUs)",
        text=("Basic Health Units are present in every block of Bhutan. "
              "Services: outpatient consultation, vaccination, antenatal care, "
              "family planning, referral to district hospitals. "
              "All BHU services are free for Bhutanese citizens."),
        keywords=("bhu", "basic health unit", "free", "vaccination"),
        source="bhutan_flow_docs/health",
    ),

    # ── PERMITS — SAND & STONE ──────────────────────────────────────
    KnowledgeChunk(
        id="prm_sand",
        service="permits",
        title="Sand Permit",
        text=("For a sand permit in Bhutan you will need:\n"
              "  • CID card\n"
              "  • Land Thram (if extracting from own land)\n"
              "  • Approval from the Department of Geology and Mines\n"
              "  • Completed application form from the dzongkhag office\n"
              "Apply at your local Dzongkhag Administration or the "
              "Department of Geology and Mines, Thimphu."),
        keywords=("sand", "permit", "geology", "mines", "extract"),
        source="bhutan_flow_docs/permits",
    ),
    KnowledgeChunk(
        id="prm_stone",
        service="permits",
        title="Stone Permit",
        text=("For a stone permit in Bhutan you will need:\n"
              "  • CID card\n"
              "  • Land Thram\n"
              "  • Approval from the Department of Geology and Mines\n"
              "  • Completed application form\n"
              "Apply at the Department of Geology and Mines or your "
              "local Dzongkhag Administration office."),
        keywords=("stone", "permit", "geology", "mines", "quarry"),
        source="bhutan_flow_docs/permits",
    ),
    KnowledgeChunk(
        id="prm_timber_eligibility",
        service="permits",
        title="Timber Permit — Eligibility Rules",
        text=("To be eligible for subsidized rural timber you must:\n"
              "  • Be a native of the area (Head of Gung / village household)\n"
              "  • Have land registered in your name\n"
              "  • Use timber for bona-fide rural purpose only\n"
              "  • NOT live inside Thromde (municipal) area\n"
              "  • NOT live within 2 km radius of Thromde boundary\n"
              "  • NOT have already received subsidized timber for another rural house\n"
              "Approved uses: house construction, repair, livestock shelter, "
              "fencing post, flag pole, firewood."),
        keywords=("eligible", "eligibility", "subsidized", "timber", "thromde",
                  "rural", "qualify", "allowed", "can i"),
        source="bhutan_flow_docs/permits",
    ),

    # ── BUSINESS ────────────────────────────────────────────────────
    KnowledgeChunk(
        id="biz_sole",
        service="business",
        title="Sole Proprietorship Registration",
        text=("To register a sole proprietorship in Bhutan you will need:\n"
              "  • Valid CID card\n"
              "  • Completed application form (from MoEA or moea.gov.bt)\n"
              "  • Proof of address\n"
              "  • Initial capital declaration\n"
              "  • No-objection letter if operating from rented premises"),
        keywords=("business", "register", "sole", "proprietorship"),
        source="bhutan_flow_docs/business",
    ),
    KnowledgeChunk(
        id="biz_processing",
        service="business",
        title="License Processing Time",
        text=("If all required information, documents, and clearances are "
              "submitted, the industry license or registration certificate "
              "is issued within one or two working days."),
        keywords=("how long", "processing", "time", "license"),
        source="bhutan_flow_docs/business",
    ),
    KnowledgeChunk(
        id="biz_office",
        service="business",
        title="MoEA Office — Thimphu",
        text=("Business registration office: Ministry of Economic Affairs (MoEA), "
              "Tashichhodzong vicinity, Thimphu. Phone: +975-2-xxx-xxxx. "
              "Open Mon–Fri, 9am–5pm. Online: www.moea.gov.bt"),
        keywords=("office", "moea", "thimphu", "where", "business"),
        source="bhutan_flow_docs/business",
    ),
]


def search_kb(user_text: str, service: str, top_k: int = 2,
              score_threshold: float = 0.5) -> List[KnowledgeChunk]:
    """
    Keyword overlap search filtered by service.
    score_threshold: chunks scoring below this are dropped to avoid irrelevant results.
    Set to 0 to disable threshold (return top_k regardless of score).
    """
    t = user_text.lower()
    candidates = [c for c in KNOWLEDGE_BASE if c.service == service]
    if not candidates:
        candidates = KNOWLEDGE_BASE
    scored = []
    for c in candidates:
        score = sum(1 for kw in c.keywords if kw in t)
        score += sum(1 for word in t.split() if word in c.text.lower()) * 0.1
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    # Only return chunks above threshold — prevents returning unrelated results
    above = [(c, s) for c, s in scored[:top_k] if s >= score_threshold]
    return [c for c, s in above] or [scored[0][0]]  # always return at least 1


# ───────────────────────────────────────────────────────────────────
# Conversation engine
# ───────────────────────────────────────────────────────────────────
@dataclass
class TurnResponse:
    bot_text: str
    sources: List[str]
    step_label: str = ""
    fallback_id: Optional[str] = None
    ref_no: str = ""


class ConversationEngine:
    SERVICE_INTROS = {
        "permits":  "Great! Let me guide you through that. 🪪",
        "health":   "Of course! Let me guide you to the right information. 🏥",
        "business": "Great! Let me guide you through the process. 🏢",
        "general":  "Let me share what I can help with. 📋",
    }

    def process(self, user_text: str,
                classification: Dict[str, ClassificationResult]
                ) -> TurnResponse:
        ref = f"REF-{uuid.uuid4().hex[:8].upper()}"

        # ── Routing rule 0: low-confidence → escalate to human ───────
        low_confidence_dims = [
            dim for dim, r in classification.items()
            if r.confidence < CONFIDENCE_THRESHOLD and r.label != "unknown"
        ]
        if low_confidence_dims:
            return TurnResponse(
                bot_text=(
                    "I'm not fully sure I understood your request correctly. 🙏\n"
                    "To make sure you get the right help, please contact a "
                    "government officer directly:\n"
                    "  • Call: 17 (toll-free helpline)\n"
                    "  • Visit your nearest Dzongkhag Administration office\n"
                    f"Your reference number: {ref}"
                ),
                sources=[],
                fallback_id="FB-06",
                ref_no=ref,
            )

        if classification["in_scope"].label == "out_of_scope":
            return TurnResponse(
                bot_text=(
                    "I can help only with the public services in this assistant.\n"
                    "I can still help you find the right office, start again, "
                    "or talk to a person."
                ),
                sources=[],
                fallback_id="FB-03",
                ref_no=ref,
            )

        if classification["safety"].label == "unsafe":
            return TurnResponse(
                bot_text=(
                    "I can share general health information, but I'm not able to "
                    "give medical advice or diagnose conditions. 🏥\n"
                    "For personal health guidance, please visit your nearest BHU "
                    "or call 112."
                ),
                sources=[],
                fallback_id=None,
                ref_no=ref,
            )

        service = classification["service"].label
        if service not in {"permits", "health", "business", "ndi",
                           "civil_registration", "education", "taxes"}:
            service = "general"

        chunks = search_kb(user_text, service=service, top_k=2, score_threshold=0.5)
        intro  = self.SERVICE_INTROS.get(service, self.SERVICE_INTROS["general"])

        if chunks:
            body    = "\n\n".join(c.text for c in chunks)
            sources = [c.source for c in chunks]
        else:
            body = ("Please visit the relevant dzongkhag office for specific "
                    f"details. Your reference: {ref}")
            sources = []

        bot_text = (
            f"{intro}\n{body}\n"
            "Is there anything else I can help you with?"
        )

        return TurnResponse(
            bot_text=bot_text,
            sources=sources,
            step_label="Step 3 of 4",
            fallback_id=None,
            ref_no=ref,
        )


# ───────────────────────────────────────────────────────────────────
# Output formatter
# ───────────────────────────────────────────────────────────────────
def print_turn(user_text: str, response: TurnResponse,
               classification: Dict[str, ClassificationResult]) -> None:
    print(f"[USER] {user_text}")
    bot_lines = response.bot_text.split("\n")
    print(f"[BOT]  {bot_lines[0]}")
    for line in bot_lines[1:]:
        print(f"       {line}")
    if response.step_label:
        print(f"       {response.step_label}")
    if response.sources:
        print(f"       📚 Sources: {response.sources}")
    if response.fallback_id:
        print(f"       ⚠ Fallback: {response.fallback_id}")
    print()
    print(format_classification_table(classification))
    print()


# ═══════════════════════════════════════════════════════════════════
#                        ━━━━  RUNNERS  ━━━━
# ═══════════════════════════════════════════════════════════════════

DEMO_SCENARIOS = [
    ("normal permit request",
     "I want to know what documents I need for a construction permit in Thimphu"),
    ("out-of-scope request",
     "Who won the cricket match yesterday?"),
    ("unsafe health request",
     "My stomach has been hurting for a week — do I have cancer?"),
    ("normal health request",
     "Where is the nearest hospital in Paro?"),
    ("normal business request",
     "What documents do I need to register a sole proprietorship?"),
    ("out-of-scope chitchat",
     "Can you tell me a joke?"),
]


def make_classifier(backend: str = CLASSIFIER_BACKEND) -> BaseClassifier:
    """Picks which classifier to instantiate: "gemini", "lmstudio", or "keyword"."""
    backend = backend.strip().lower()

    if backend == "lmstudio":
        if not OPENAI_AVAILABLE:
            print("⚠  openai package missing. Run: !pip install openai")
            print("⚠  Falling back to KeywordClassifier.\n")
            return KeywordClassifier()
        print(f"🖥  LMStudioClassifier → {LMSTUDIO_BASE_URL}  model={LMSTUDIO_MODEL_NAME}\n")
        return LMStudioClassifier(base_url=LMSTUDIO_BASE_URL, model=LMSTUDIO_MODEL_NAME)

    if backend == "gemini":
        if not GENAI_AVAILABLE:
            print("⚠  google-generativeai missing. Run: !pip install google-generativeai")
            print("⚠  Falling back to KeywordClassifier.\n")
            return KeywordClassifier()
        if not GEMINI_API_KEY:
            print("⚠  GEMINI_API_KEY not set. Falling back to KeywordClassifier.\n")
            return KeywordClassifier()
        print(f"☁  GeminiClassifier → model={GEMINI_MODEL_NAME}\n")
        return GeminiClassifier(api_key=GEMINI_API_KEY)

    # "keyword" or unrecognised
    print("⌨  KeywordClassifier (no model, pure keyword matching).\n")
    return KeywordClassifier()


def run_demo(backend: str = CLASSIFIER_BACKEND) -> None:
    """Run the pre-defined sample scenarios."""
    classifier = make_classifier(backend)
    engine     = ConversationEngine()
    print("=" * 72)
    print(" Bhutan Voice-First Public Service Assistant — Demo with Classifier")
    print("=" * 72)
    for label, user_text in DEMO_SCENARIOS:
        print(f"\n── Scenario: {label} ──")
        c = classifier.classify(user_text)
        r = engine.process(user_text, c)
        print_turn(user_text, r, c)


def run_interactive(backend: str = CLASSIFIER_BACKEND) -> None:
    """Chat with the assistant. Type 'exit' to quit."""
    classifier = make_classifier(backend)
    engine     = ConversationEngine()
    print("=" * 72)
    print(" Bhutan Voice-First Public Service Assistant")
    print("=" * 72)
    print(" Kuzuzangpo la! 🙏  How can I help you today?")
    print(" Type your question below. Type 'exit' (or 'quit') to end the chat.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTashi Delek! 🙏")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("\nTashi Delek! 🙏")
            break
        print()
        c = classifier.classify(user_text)
        r = engine.process(user_text, c)
        print_turn(user_text, r, c)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bhutan public-service request classifier")
    parser.add_argument("--backend", choices=["gemini", "lmstudio", "keyword"],
                         default=CLASSIFIER_BACKEND, help="classification backend to use")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                         help="run the 6 sample scenarios, or chat interactively")
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(args.backend)
    else:
        run_demo(args.backend)
