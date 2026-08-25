"""Hybrid Retrieval Engine for Voice + Chat Prototyping
========================================================

Updated architecture:
- Voice / chat input normalization
- STT / query preprocessing
- Lightweight intent classification
- Query expansion
- DPR dense retrieval (query encoder + passage encoder)
- Metadata filtering / routing tags
- Re-ranking
- Context injection for downstream LLM
- Response payload formatting for chat / TTS

This version removes BM25 and uses DPR-style dense retrieval throughout.

Dependencies (recommended):
    pip install numpy transformers torch faiss-cpu

Optional:
    pip install sentence-transformers

Usage:
    engine = RetrievalEngine.from_documents(docs)
    response = engine.answer("What is the eligibility for senior citizen pension?", channel="chat")
    print(response.final_answer)

    To implement this architecture in the main system and to bring it to live follow these instructions:
        Keep as-is
QueryPreprocessor
IntentClassifier for routing
QueryExpander
ContextManager
ResponseFormatter
the overall RetrievalEngine orchestration pattern
Change these parts
1) Replace the stub answer generator

Right now _generate_stub_answer() is only summarizing the top chunk.
For live use, replace it with your actual LLM call:

input: query + context
output: final answer
add guardrails for refusal / escalation / missing context

This is the biggest change.

2) Move retrieval index creation offline

Your DPRRetriever should not rebuild everything every time the app starts.

Change it so that:

document ingestion happens in a separate pipeline
embeddings are precomputed
FAISS index is saved to disk or a vector DB
app startup only loads the index

That is necessary for live latency.

3) Add a persistent storage layer

Right now everything is in memory.

For live:

documents and chunks should come from a database or object store
embeddings/index should persist
chat/session history should persist
feedback logs should persist
4) Add an API layer

 frontend should not call the engine directly in a real deployment.

Create a service boundary:

/chat
/voice
/ingest
/search
/feedback

This makes Streamlit, web, and mobile clients all use the same backend.

5) Add streaming and async handling

For live chat, we will want:

streaming tokens from the LLM
async request handling
timeouts and retries
cancellation support

Streamlit can still consume this, but the backend should support it first.

6) Add voice pipeline services

For the voice path, connect:

STT in
retrieval engine
LLM response
TTS out

 current is_voice=True flag is only a placeholder.

 7) Add observability

For live systems, add:

request logs
retrieval traces
latency metrics
top-k hit quality
fallback rate
clarification rate
user feedback capture

That is what lets you debug bad answers in production.

8) Add auth and access control

If this connects to live data, add:

user auth
tenant or role filters
source-level permissions
audit logging

Especially important if documents are restricted.

Practical live architecture

A clean production split is:

Frontend: Streamlit / web app
API Orchestrator: receives chat/voice requests
Retrieval Service: DPR + filters + reranker
LLM Service: answer generation
Ingestion Pipeline: chunking, embedding, indexing
Storage: documents, embeddings, chat logs, feedback
The exact code hotspots to change

In current code, focus on these methods/classes:

RetrievalEngine._generate_stub_answer() → replace with real LLM call
DPRRetriever.add() → make offline ingestion only
DPRRetriever.__init__() → load models once, cache them
RetrievalEngine.from_documents() → replace with loading a persisted index
RetrievalEngine.answer() → add streaming / error handling / fallback logic
ResponseFormatter → shape output for API + Streamlit + voice
build_demo_documents() → remove from production

"""

"""
bhutan_rag_flow.py
==================
Conversational Flow + DPR-RAG Pipeline — Bhutan Voice-First Public Service Assistant
Omdena Sprint · Week 2 · T2-04 / T2-05 / T2-06

Architecture Overview
---------------------
                      User Input
                          │
               ┌──────────▼──────────┐
               │  ConversationalRAG   │  ← main entry point
               │      Engine          │
               └──────────┬──────────┘
                          │
           ┌──────────────┼─────────────────┐
           │              │                 │
     ┌─────▼──────┐ ┌─────▼──────┐ ┌───────▼───────┐
     │  FlowRouter │ │  RAG Layer  │ │FallbackHandler│
     │  (state +   │ │  DPR +      │ │ (FB-01→FB-08) │
     │   scripted  │ │  retrieval) │ └───────────────┘
     │   dialog)   │ └────────────┘
     └─────────────┘

Key Rules (from T2-06 Golden Rules):
  Rule 1  – One question per turn
  Rule 2  – Never re-ask already collected info
  Rule 3  – Show Step X of Y at every service node
  Rule 4  – Every flow ends with REF no. + next step + office contact
  Rule 5  – In-person steps: name, address, opening hours
  Rule 6  – Escalate after 2 failed turns → phone + email + REF no.
  Rule 7  – BHU validation mandatory
  Rule 8  – SMS summary after confirmation

Dependencies: numpy, torch, transformers, faiss-cpu
"""

import uuid
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Third-party imports (guarded) ────────────────────────────────────────────
try:
    import faiss
except ImportError:
    faiss = None

try:
    import torch
    from transformers import (
        DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
        DPRContextEncoder, DPRContextEncoderTokenizer,
    )
except ImportError:
    torch = None
    DPRQuestionEncoder = DPRQuestionEncoderTokenizer = None
    DPRContextEncoder = DPRContextEncoderTokenizer = None


# =============================================================================
# SECTION 1 — Core Data Models
# =============================================================================

class Domain(str, Enum):
    PERMIT   = "permits"
    HEALTH   = "health"
    BUSINESS = "business"
    GENERAL  = "general"


class NodeType(str, Enum):
    GREETING            = "GREETING"
    SYSTEM              = "SYSTEM"
    INFO                = "INFO"
    INPUT               = "INPUT"
    DECISION            = "DECISION"
    CONFIRM             = "CONFIRM"
    ROUTE               = "ROUTE"
    FALLBACK            = "FALLBACK"
    TERMINAL            = "TERMINAL"
    SAFETY_REDIRECT     = "SAFETY_REDIRECT"
    COMPLEXITY_REDIRECT = "COMPLEXITY_REDIRECT"


class Channel(str, Enum):
    CHAT  = "chat"
    VOICE = "voice"


@dataclass
class FlowNode:
    """One node in a conversation flow graph."""
    node_id:          str
    name:             str
    node_type:        NodeType
    bot_text:         str
    voice_text:       str
    entity_captured:  Optional[str]        = None
    next_node:        Optional[str]        = None    # default next
    alt_next:         Dict[str, str]       = field(default_factory=dict)  # condition → node_id
    step:             Optional[str]        = None    # e.g. "Step 2 of 4"
    domain:           Domain               = Domain.GENERAL
    triggers_rag:     bool                 = False   # True at SERVICE_HANDOFF nodes
    safety_check:     bool                 = False   # SAFETY_DEC node
    scope_check:      bool                 = False   # SCOPE_DEC node
    is_terminal:      bool                 = False


@dataclass
class ConversationSession:
    """Per-session stateful context — one instance per user conversation."""
    session_id:      str                  = field(default_factory=lambda: f"REF-{uuid.uuid4().hex[:8].upper()}")
    domain:          Optional[Domain]     = None
    current_node_id: str                  = "WELCOME"
    entities:        Dict[str, Any]       = field(default_factory=dict)
    fallback_count:  int                  = 0
    field_fail_count: Dict[str, int]      = field(default_factory=dict)   # per-field retry (FB-02, FB-06)
    language_pref:   str                  = "EN"
    channel:         Channel              = Channel.CHAT
    turn_count:      int                  = 0
    is_terminal:     bool                 = False
    start_time:      float                = field(default_factory=time.time)
    last_activity:   float                = field(default_factory=time.time)
    intent_id:       Optional[str]        = None
    confirmed_intent: Optional[str]       = None
    step_current:    int                  = 0
    step_total:      int                  = 0


@dataclass
class TurnResponse:
    """Result of one conversation turn."""
    bot_text:       str
    voice_text:     str
    next_node_id:   str
    entities_collected: Dict[str, Any]
    is_terminal:    bool          = False
    triggers_rag:   bool          = False
    rag_context:    str           = ""
    rag_citations:  List[Dict]    = field(default_factory=list)
    session_ref:    str           = ""
    step_label:     str           = ""
    fallback_id:    Optional[str] = None


@dataclass(frozen=True)
class DocumentChunk:
    """Retrievable knowledge unit — unchanged from original pipeline."""
    id:       str
    text:     str
    source:   str              = ""
    domain:   str              = "general"
    doc_type: str              = "generic"
    language: str              = "en"
    priority: int              = 0
    phase:    str              = "live"
    tags:     Tuple[str, ...]  = ()
    metadata: Dict[str, Any]   = field(default_factory=dict)


# =============================================================================
# SECTION 2 — Flow Registry (all 3 domain flows from the design docs)
# =============================================================================

class FlowRegistry:
    """
    Holds the complete node graphs for all 3 domains derived directly from
    the uploaded flow design documents (PRM, HLT, BIZ NodeSpec files +
    Service Query Handling Flow xlsx + Fallback xlsx).
    """

    def __init__(self):
        self._nodes: Dict[str, FlowNode] = {}
        self._build_permit_flow()
        self._build_health_flow()
        self._build_business_flow()
        self._build_service_query_nodes()
        self._build_fallback_nodes()

    def get(self, node_id: str) -> Optional[FlowNode]:
        return self._nodes.get(node_id)

    def register(self, node: FlowNode) -> None:
        self._nodes[node.node_id] = node

    # ── Permit Application Flow (PRM-OPN-xx nodes) ────────────────────────────

    def _build_permit_flow(self):
        nodes = [
            FlowNode(
                node_id="PRM-OPN-01", name="WELCOME", node_type=NodeType.GREETING,
                domain=Domain.PERMIT,
                bot_text=(
                    "Kuzuzangpo la! 🙏\n"
                    "Hello! I am your Bhutan Permit Assistant.\n"
                    "I can help with permit requirements, checklists, and office directions.\n"
                    "How can I help you today?"
                ),
                voice_text=(
                    "Kuzuzangpo la! Hello! I am your Bhutan Permit Assistant. "
                    "I can help you with permits, checklists, and office directions. "
                    "How can I help you today?"
                ),
                next_node="PRM-OPN-02",
            ),
            FlowNode(
                node_id="PRM-OPN-02", name="LANGUAGE_SELECT", node_type=NodeType.SYSTEM,
                domain=Domain.PERMIT,
                bot_text=(
                    "Which language would you prefer?\n"
                    "1️⃣ English\n"
                    "2️⃣ Dzongkha (call +975-2-xxx-xxxx)"
                ),
                voice_text=(
                    "Which language do you prefer? "
                    "Say ONE for English, or TWO for Dzongkha support by phone."
                ),
                entity_captured="language_pref",
                next_node="PRM-OPN-03",
                alt_next={"DZ": "PRM-OPN-03b"},
            ),
            FlowNode(
                node_id="PRM-OPN-03b", name="DZONGKHA_REDIRECT", node_type=NodeType.TERMINAL,
                domain=Domain.PERMIT, is_terminal=True,
                bot_text=(
                    "ཀུཟུཟང་པོ་ལགས། 🙏\n"
                    "For Dzongkha assistance, please call:\n"
                    "📞 +975-2-xxx-xxxx (Mon–Fri, 9 am–5 pm)"
                ),
                voice_text=(
                    "Kuzuzangpo la! For Dzongkha support, "
                    "please call the helpline number shown on screen."
                ),
                entity_captured="language_pref",
            ),
            FlowNode(
                node_id="PRM-OPN-03", name="CAPABILITY_INTRO", node_type=NodeType.INFO,
                domain=Domain.PERMIT,
                bot_text=(
                    "Great! I can help you with:\n"
                    "1. Permit requirements & documents\n"
                    "2. Readiness check before office visit\n"
                    "3. Application status & office locations\n"
                    "What do you need help with?"
                ),
                voice_text=(
                    "Great! I can help you with permit requirements, "
                    "document checklists, application status, and office locations. "
                    "What do you need help with?"
                ),
                next_node="PRM-OPN-04",
            ),
            FlowNode(
                node_id="PRM-OPN-04", name="INTENT_PROMPT", node_type=NodeType.INPUT,
                domain=Domain.PERMIT,
                bot_text=(
                    "Please tell me what you need. "
                    "You can type freely — or choose a number from the list above."
                ),
                voice_text=(
                    "Please tell me what you need. "
                    "You can speak freely, or choose a number from the menu."
                ),
                entity_captured="user_intent",
                next_node="PRM-OPN-05",
                alt_next={"fail": "PRM-OPN-07"},
            ),
            FlowNode(
                node_id="PRM-OPN-07", name="NOT_UNDERSTOOD_1", node_type=NodeType.FALLBACK,
                domain=Domain.PERMIT,
                bot_text=(
                    "I didn't quite catch that.\n"
                    "Try again, or choose from the menu:\n"
                    "1. Requirements  2. Readiness  3. Status  4. Office"
                ),
                voice_text=(
                    "I did not catch that. Please try again. "
                    "Or say ONE for requirements, TWO for readiness, "
                    "THREE for status, or FOUR for office location."
                ),
                next_node="PRM-OPN-04",
                alt_next={"fail2": "PRM-OPN-08"},
            ),
            FlowNode(
                node_id="PRM-OPN-08", name="NOT_UNDERSTOOD_2", node_type=NodeType.FALLBACK,
                domain=Domain.PERMIT,
                bot_text=(
                    "I'm having trouble understanding. "
                    "Let me connect you with a human agent who can help. 📞"
                ),
                voice_text=(
                    "I am having trouble understanding. "
                    "Let me connect you to a human agent who can help directly."
                ),
                next_node="PRM-OPN-ESC",
            ),
            FlowNode(
                node_id="PRM-OPN-05", name="INTENT_CONFIRM", node_type=NodeType.CONFIRM,
                domain=Domain.PERMIT,
                bot_text=(
                    "You'd like help with: {intent_summary}.\n"
                    "Is that right?\n"
                    "1️⃣ Yes   2️⃣ No — let me rephrase"
                ),
                voice_text=(
                    "You would like help with {intent_summary}. "
                    "Is that correct? Say YES to continue or NO to rephrase."
                ),
                entity_captured="intent_confirmed",
                next_node="PRM-OPN-06",
                alt_next={"no": "PRM-OPN-04"},
            ),
            FlowNode(
                node_id="PRM-OPN-06", name="SERVICE_HANDOFF", node_type=NodeType.ROUTE,
                domain=Domain.PERMIT, triggers_rag=True,
                bot_text=(
                    "Perfect! Let me guide you through that. 🪪\n"
                    "{rag_answer}"
                ),
                voice_text=(
                    "Perfect. Let me guide you through that now. {rag_answer}"
                ),
                entity_captured="service_type",
                next_node="PRM-SVC-CONFIRM",
            ),
            FlowNode(
                node_id="PRM-OPN-ESC", name="HUMAN_ESCALATION", node_type=NodeType.TERMINAL,
                domain=Domain.PERMIT, is_terminal=True,
                bot_text=(
                    "Your details have been noted. 🙏\n"
                    "A service officer will follow up with you.\n"
                    "Ref: {ref_no}\n"
                    "📞 17  │  📧 help@gov.bt"
                ),
                voice_text=(
                    "Your details have been noted. "
                    "A service officer will follow up with you shortly. "
                    "Your reference number is shown on screen. Tashi Delek."
                ),
            ),
            # Service query confirmation node for permits
            FlowNode(
                node_id="PRM-SVC-CONFIRM", name="CONFIRMATION", node_type=NodeType.CONFIRM,
                domain=Domain.PERMIT,
                bot_text=(
                    "Great! You have everything you need. ✅\n"
                    "Your next step is to visit {office_name} at {office_address}.\n"
                    "They are open Monday to Friday, 9am to 5pm.\n"
                    "📱 SMS summary sent.\n"
                    "Your session reference number is {ref_no}.\n"
                    "Is there anything else I can help you with?"
                ),
                voice_text=(
                    "Great! Your next step is to visit the office. "
                    "Details are shown on screen. Your reference number is {ref_no}. "
                    "Is there anything else I can help you with?"
                ),
                is_terminal=False,
                next_node="PRM-OPN-03",  # loop back to main menu
            ),
        ]
        for n in nodes:
            self.register(n)

    # ── Health Information Flow (HLT-OPN-xx nodes) ───────────────────────────

    def _build_health_flow(self):
        nodes = [
            FlowNode(
                node_id="HLT-OPN-01", name="WELCOME", node_type=NodeType.GREETING,
                domain=Domain.HEALTH,
                bot_text=(
                    "Kuzuzangpo la! 🙏\n"
                    "Hello! I am your Bhutan Health Assistant.\n"
                    "I can help you find health services, understand health information, "
                    "and navigate care options.\n"
                    "How can I help you today?"
                ),
                voice_text=(
                    "Kuzuzangpo la! Hello! I am your Bhutan Health Assistant. "
                    "I can help you find health services, understand health information, "
                    "and navigate your care options. How can I help you today?"
                ),
                next_node="HLT-OPN-02",
            ),
            FlowNode(
                node_id="HLT-OPN-02", name="LANGUAGE_SELECT", node_type=NodeType.SYSTEM,
                domain=Domain.HEALTH,
                bot_text=(
                    "Which language would you prefer?\n"
                    "1️⃣ English\n"
                    "2️⃣ Dzongkha (call +975-2-xxx-xxxx)"
                ),
                voice_text=(
                    "Which language do you prefer? "
                    "Say ONE for English, or TWO for Dzongkha support by phone."
                ),
                entity_captured="language_pref",
                next_node="HLT-OPN-03",
                alt_next={"DZ": "HLT-OPN-03b"},
            ),
            FlowNode(
                node_id="HLT-OPN-03b", name="DZONGKHA_REDIRECT", node_type=NodeType.TERMINAL,
                domain=Domain.HEALTH, is_terminal=True,
                bot_text=(
                    "ཀུཟུཟང་པོ་ལགས། 🙏\n"
                    "For Dzongkha health assistance, please call:\n"
                    "📞 +975-2-xxx-xxxx (Mon–Fri, 9 am–5 pm)"
                ),
                voice_text=(
                    "Kuzuzangpo la! For Dzongkha health support, "
                    "please call the helpline number shown on screen."
                ),
            ),
            FlowNode(
                node_id="HLT-OPN-03", name="CAPABILITY_INTRO", node_type=NodeType.INFO,
                domain=Domain.HEALTH,
                bot_text=(
                    "Great! I can help you with:\n"
                    "1. Health facilities & service locations\n"
                    "2. Understanding health information simply\n"
                    "3. Service eligibility & what's free\n"
                    "4. Registration, appointments & next steps\n"
                    "What would you like to know?"
                ),
                voice_text=(
                    "Great! I can help you find health facilities, "
                    "understand health information, check eligibility, "
                    "and guide you on next steps. What would you like to know?"
                ),
                next_node="HLT-OPN-04",
            ),
            FlowNode(
                node_id="HLT-OPN-04", name="INTENT_PROMPT", node_type=NodeType.INPUT,
                domain=Domain.HEALTH,
                bot_text=(
                    "Please tell me what you need. "
                    "You can type freely — or choose a number from the list above."
                ),
                voice_text=(
                    "Please tell me what you need. "
                    "You can speak freely, or choose a number from the menu."
                ),
                entity_captured="user_intent",
                next_node="HLT-OPN-05",
                alt_next={"fail": "HLT-OPN-07"},
            ),
            FlowNode(
                node_id="HLT-OPN-07", name="NOT_UNDERSTOOD_1", node_type=NodeType.FALLBACK,
                domain=Domain.HEALTH,
                bot_text=(
                    "I didn't quite catch that.\n"
                    "Try again, or choose from the menu:\n"
                    "1. Facilities  2. Eligibility  3. Information  4. Appointment"
                ),
                voice_text=(
                    "I did not catch that. Please try again. "
                    "Or say ONE for facilities, TWO for eligibility, "
                    "THREE for health information, or FOUR for appointments."
                ),
                next_node="HLT-OPN-04",
                alt_next={"fail2": "HLT-OPN-08"},
            ),
            FlowNode(
                node_id="HLT-OPN-08", name="NOT_UNDERSTOOD_2", node_type=NodeType.FALLBACK,
                domain=Domain.HEALTH,
                bot_text=(
                    "I'm having trouble understanding. "
                    "Let me connect you with a health worker who can help. 🏥"
                ),
                voice_text=(
                    "I am having trouble understanding. "
                    "Let me connect you to a health worker who can help directly."
                ),
                next_node="HLT-OPN-ESC",
            ),
            FlowNode(
                node_id="HLT-OPN-05", name="INTENT_CONFIRM", node_type=NodeType.CONFIRM,
                domain=Domain.HEALTH,
                bot_text=(
                    "You'd like help with: {intent_summary}.\n"
                    "Is that right?\n"
                    "1️⃣ Yes   2️⃣ No — let me rephrase"
                ),
                voice_text=(
                    "You would like help with {intent_summary}. "
                    "Is that correct? Say YES to continue or NO to rephrase."
                ),
                entity_captured="intent_confirmed",
                next_node="HLT-OPN-06",      # after SAFETY_DEC passes
                alt_next={"no": "HLT-OPN-04", "unsafe": "HLT-OPN-SAFE"},
                safety_check=True,
            ),
            FlowNode(
                node_id="HLT-OPN-SAFE", name="SAFETY_REDIRECT", node_type=NodeType.SAFETY_REDIRECT,
                domain=Domain.HEALTH, is_terminal=True,
                bot_text=(
                    "I can share general health information, but I'm not able to give "
                    "medical advice or diagnose conditions. 🏥\n"
                    "For personal health guidance, please visit your nearest BHU or call 112."
                ),
                voice_text=(
                    "I can share general health information, but I am not able to give "
                    "medical advice or diagnose conditions. "
                    "For personal health guidance, please visit your nearest "
                    "Basic Health Unit or call one one two."
                ),
            ),
            FlowNode(
                node_id="HLT-OPN-06", name="SERVICE_HANDOFF", node_type=NodeType.ROUTE,
                domain=Domain.HEALTH, triggers_rag=True,
                bot_text=(
                    "Of course! Let me guide you to the right information. 🏥\n"
                    "{rag_answer}"
                ),
                voice_text=(
                    "Of course! Let me guide you to the right information. {rag_answer}"
                ),
                entity_captured="service_type",
                next_node="HLT-SVC-CONFIRM",
            ),
            FlowNode(
                node_id="HLT-OPN-ESC", name="HUMAN_ESCALATION", node_type=NodeType.TERMINAL,
                domain=Domain.HEALTH, is_terminal=True,
                bot_text=(
                    "Your details have been noted. 🙏\n"
                    "A health worker will follow up with you.\n"
                    "Ref: {ref_no}\n"
                    "📞 112  │  📧 health@gov.bt"
                ),
                voice_text=(
                    "Your details have been noted. "
                    "A health worker will follow up with you shortly. "
                    "Your reference number is shown on screen. Tashi Delek."
                ),
            ),
            FlowNode(
                node_id="HLT-SVC-CONFIRM", name="CONFIRMATION", node_type=NodeType.CONFIRM,
                domain=Domain.HEALTH,
                bot_text=(
                    "I hope that was helpful. ✅\n"
                    "Your next step is {next_step}.\n"
                    "If you need more help, call {health_office_number}.\n"
                    "📱 Session reference: {ref_no}.\n"
                    "Is there anything else I can help you with?"
                ),
                voice_text=(
                    "I hope that was helpful. Your next step is shown on screen. "
                    "Your reference number is {ref_no}. Is there anything else?"
                ),
                is_terminal=False,
                next_node="HLT-OPN-03",
            ),
        ]
        for n in nodes:
            self.register(n)

    # ── Business Registration Flow (BIZ-OPN-xx nodes) ────────────────────────

    def _build_business_flow(self):
        nodes = [
            FlowNode(
                node_id="BIZ-OPN-01", name="WELCOME", node_type=NodeType.GREETING,
                domain=Domain.BUSINESS,
                bot_text=(
                    "Kuzuzangpo la! 🙏\n"
                    "Hello! I am your Bhutan Business Assistant.\n"
                    "I can help you with registration, licensing, documents, "
                    "and process guidance.\n"
                    "How can I help you today?"
                ),
                voice_text=(
                    "Kuzuzangpo la! Hello! I am your Bhutan Business Assistant. "
                    "I can help you with registration, licensing, documents, "
                    "and process guidance. How can I help you today?"
                ),
                next_node="BIZ-OPN-02",
            ),
            FlowNode(
                node_id="BIZ-OPN-02", name="LANGUAGE_SELECT", node_type=NodeType.SYSTEM,
                domain=Domain.BUSINESS,
                bot_text=(
                    "Which language would you prefer?\n"
                    "1️⃣ English\n"
                    "2️⃣ Dzongkha (call +975-2-xxx-xxxx)"
                ),
                voice_text=(
                    "Which language do you prefer? "
                    "Say ONE for English, or TWO for Dzongkha support by phone."
                ),
                entity_captured="language_pref",
                next_node="BIZ-OPN-03",
                alt_next={"DZ": "BIZ-OPN-03b"},
            ),
            FlowNode(
                node_id="BIZ-OPN-03b", name="DZONGKHA_REDIRECT", node_type=NodeType.TERMINAL,
                domain=Domain.BUSINESS, is_terminal=True,
                bot_text=(
                    "ཀུཟུཟང་པོ་ལགས། 🙏\n"
                    "For Dzongkha business assistance, please call:\n"
                    "📞 +975-2-xxx-xxxx (Mon–Fri, 9 am–5 pm)"
                ),
                voice_text=(
                    "Kuzuzangpo la! For Dzongkha business support, "
                    "please call the helpline number shown on screen."
                ),
            ),
            FlowNode(
                node_id="BIZ-OPN-03", name="CAPABILITY_INTRO", node_type=NodeType.INFO,
                domain=Domain.BUSINESS,
                bot_text=(
                    "Great! I can help you with:\n"
                    "1. Registration steps & process overview\n"
                    "2. Required documents & checklists\n"
                    "3. What to do first — licensing or registration?\n"
                    "4. Eligibility & office locations\n"
                    "What would you like help with?"
                ),
                voice_text=(
                    "Great! I can help you with registration steps, "
                    "document checklists, process sequencing, eligibility, "
                    "and office locations. What would you like help with?"
                ),
                next_node="BIZ-OPN-04",
            ),
            FlowNode(
                node_id="BIZ-OPN-04", name="INTENT_PROMPT", node_type=NodeType.INPUT,
                domain=Domain.BUSINESS,
                bot_text=(
                    "Please tell me what you need. "
                    "You can type freely — or choose a number from the list above."
                ),
                voice_text=(
                    "Please tell me what you need. "
                    "You can speak freely, or choose a number from the menu."
                ),
                entity_captured="user_intent",
                next_node="BIZ-OPN-05",
                alt_next={"fail": "BIZ-OPN-07"},
            ),
            FlowNode(
                node_id="BIZ-OPN-07", name="NOT_UNDERSTOOD_1", node_type=NodeType.FALLBACK,
                domain=Domain.BUSINESS,
                bot_text=(
                    "I didn't quite catch that.\n"
                    "Try again, or choose from the menu:\n"
                    "1. Registration  2. Documents  3. Licensing  4. Office"
                ),
                voice_text=(
                    "I did not catch that. Please try again. "
                    "Or say ONE for registration, TWO for documents, "
                    "THREE for licensing, or FOUR for office location."
                ),
                next_node="BIZ-OPN-04",
                alt_next={"fail2": "BIZ-OPN-08"},
            ),
            FlowNode(
                node_id="BIZ-OPN-08", name="NOT_UNDERSTOOD_2", node_type=NodeType.FALLBACK,
                domain=Domain.BUSINESS,
                bot_text=(
                    "I'm having trouble understanding. "
                    "Let me connect you with a registration officer who can help. 🏢"
                ),
                voice_text=(
                    "I am having trouble understanding. "
                    "Let me connect you to a registration officer who can help directly."
                ),
                next_node="BIZ-OPN-ESC",
            ),
            FlowNode(
                node_id="BIZ-OPN-05", name="INTENT_CONFIRM", node_type=NodeType.CONFIRM,
                domain=Domain.BUSINESS,
                bot_text=(
                    "You'd like help with: {intent_summary}.\n"
                    "Is that right?\n"
                    "1️⃣ Yes   2️⃣ No — let me rephrase"
                ),
                voice_text=(
                    "You would like help with {intent_summary}. "
                    "Is that correct? Say YES to continue or NO to rephrase."
                ),
                entity_captured="intent_confirmed",
                next_node="BIZ-OPN-06",
                alt_next={"no": "BIZ-OPN-04", "complex": "BIZ-OPN-COMPLEX"},
                scope_check=True,
            ),
            FlowNode(
                node_id="BIZ-OPN-COMPLEX", name="COMPLEXITY_REDIRECT",
                node_type=NodeType.COMPLEXITY_REDIRECT,
                domain=Domain.BUSINESS, is_terminal=True,
                bot_text=(
                    "Your query involves some specific requirements best discussed "
                    "with a registration officer. 🏢\n"
                    "Please visit the Ministry of Economic Affairs or call +975-2-xxx-xxxx."
                ),
                voice_text=(
                    "Your query involves requirements best discussed with a registration officer. "
                    "Please visit the Ministry of Economic Affairs "
                    "or call the number shown on screen."
                ),
            ),
            FlowNode(
                node_id="BIZ-OPN-06", name="SERVICE_HANDOFF", node_type=NodeType.ROUTE,
                domain=Domain.BUSINESS, triggers_rag=True,
                bot_text=(
                    "Great! Let me guide you through the process. 🏢\n"
                    "{rag_answer}"
                ),
                voice_text=(
                    "Great! Let me guide you through the process. {rag_answer}"
                ),
                entity_captured="service_type",
                next_node="BIZ-SVC-CONFIRM",
            ),
            FlowNode(
                node_id="BIZ-OPN-ESC", name="HUMAN_ESCALATION", node_type=NodeType.TERMINAL,
                domain=Domain.BUSINESS, is_terminal=True,
                bot_text=(
                    "Your details have been noted. 🙏\n"
                    "A registration officer will follow up with you.\n"
                    "Ref: {ref_no}\n"
                    "📞 +975-2-xxx-xxxx  │  📧 biz@gov.bt"
                ),
                voice_text=(
                    "Your details have been noted. "
                    "A registration officer will follow up shortly. "
                    "Your reference number is shown on screen. Tashi Delek."
                ),
            ),
            FlowNode(
                node_id="BIZ-SVC-CONFIRM", name="CONFIRMATION", node_type=NodeType.CONFIRM,
                domain=Domain.BUSINESS,
                bot_text=(
                    "Here is your personalised summary: ✅\n"
                    "• Business type: {business_type}\n"
                    "• Location: {dzongkhag}\n"
                    "• Documents required: {document_list}\n"
                    "• Next step: Visit {office_name} at {office_address}. Open Mon–Fri, 9am–5pm.\n"
                    "• Online: www.moea.gov.bt\n"
                    "📱 SMS summary sent.  Session reference: {ref_no}\n"
                    "Is there anything else I can help you with?"
                ),
                voice_text=(
                    "Here is your summary. Details are shown on screen. "
                    "Your reference number is {ref_no}. Is there anything else?"
                ),
                is_terminal=False,
                next_node="BIZ-OPN-03",
            ),
        ]
        for n in nodes:
            self.register(n)

    # ── Service Query nodes (multi-step entity collection per domain) ─────────

    def _build_service_query_nodes(self):
        """
        Lightweight entity-collection nodes that mirror the 4-step flows
        in Service_Query_Handling_Flow.xlsx (N1→N6) for each domain.
        These sit between SERVICE_HANDOFF and CONFIRMATION.
        """
        # Permit entity-collection (permit_type → dzongkhag → info delivery)
        self.register(FlowNode(
            node_id="PRM-SVC-N2", name="PERMIT_TYPE_COLLECT", node_type=NodeType.INPUT,
            domain=Domain.PERMIT, step="Step 1 of 4",
            bot_text="What type of permit do you need? For example: construction, business, or trade permit.",
            voice_text="What type of permit do you need? For example, say construction, business, or trade.",
            entity_captured="permit_type", next_node="PRM-SVC-N3",
        ))
        self.register(FlowNode(
            node_id="PRM-SVC-N3", name="DZONGKHAG_COLLECT", node_type=NodeType.INPUT,
            domain=Domain.PERMIT, step="Step 2 of 4",
            bot_text="Which dzongkhag are you in? For example: Thimphu, Paro, Bumthang.",
            voice_text="Which dzongkhag are you in? For example, say Thimphu, Paro, or Bumthang.",
            entity_captured="dzongkhag", next_node="PRM-SVC-N4",
        ))
        self.register(FlowNode(
            node_id="PRM-SVC-N4", name="INFO_DELIVERY", node_type=NodeType.INFO,
            domain=Domain.PERMIT, step="Step 3 of 4", triggers_rag=True,
            bot_text=(
                "For a {permit_type} permit in {dzongkhag}, you will need:\n"
                "{rag_answer}\n"
                "Would you like me to check if you have everything ready?"
            ),
            voice_text=(
                "For a {permit_type} permit in {dzongkhag}, you will need the following. "
                "{rag_answer} Would you like me to check your readiness?"
            ),
            next_node="PRM-SVC-N5",
        ))
        self.register(FlowNode(
            node_id="PRM-SVC-N5", name="READINESS_CHECK", node_type=NodeType.INPUT,
            domain=Domain.PERMIT, step="Step 4 of 4",
            bot_text="Which of these documents do you already have? Tell me what you have and I'll tell you what is missing.",
            voice_text="Which of these documents do you already have? Tell me and I will say what is missing.",
            entity_captured="document_list", next_node="PRM-SVC-CONFIRM",
        ))

        # Health entity-collection
        self.register(FlowNode(
            node_id="HLT-SVC-N2", name="DZONGKHAG_COLLECT", node_type=NodeType.INPUT,
            domain=Domain.HEALTH, step="Step 1 of 4",
            bot_text="Which dzongkhag are you in? This will help me find the right information for you.",
            voice_text="Which dzongkhag are you in? Say the name and I will find the right information.",
            entity_captured="dzongkhag", next_node="HLT-SVC-N3",
        ))
        self.register(FlowNode(
            node_id="HLT-SVC-N3", name="SERVICE_TYPE_COLLECT", node_type=NodeType.INPUT,
            domain=Domain.HEALTH, step="Step 2 of 4",
            bot_text="What kind of health service do you need? For example: general check-up, vaccination, specialist, or emergency.",
            voice_text="What kind of health service do you need? For example, say check-up, vaccination, specialist, or emergency.",
            entity_captured="service_type", next_node="HLT-SVC-N4",
        ))
        self.register(FlowNode(
            node_id="HLT-SVC-N4", name="INFO_DELIVERY", node_type=NodeType.INFO,
            domain=Domain.HEALTH, step="Step 3 of 4", triggers_rag=True,
            bot_text=(
                "For {service_type} in {dzongkhag}:\n"
                "{rag_answer}"
            ),
            voice_text="Here is the information for {service_type} in {dzongkhag}. {rag_answer}",
            next_node="HLT-SVC-CONFIRM",
        ))

        # Business entity-collection (Path A: Register)
        self.register(FlowNode(
            node_id="BIZ-SVC-N3a", name="BUSINESS_TYPE_COLLECT", node_type=NodeType.INPUT,
            domain=Domain.BUSINESS, step="Step 1 of 5",
            bot_text="What type of business do you want to register? For example: sole proprietorship, partnership, or private company.",
            voice_text="What type of business do you want to register? For example, say sole proprietorship, partnership, or private company.",
            entity_captured="business_type", next_node="BIZ-SVC-N4a",
        ))
        self.register(FlowNode(
            node_id="BIZ-SVC-N4a", name="DZONGKHAG_COLLECT", node_type=NodeType.INPUT,
            domain=Domain.BUSINESS, step="Step 2 of 5",
            bot_text="Which dzongkhag will the business operate in? For example: Thimphu, Paro, Bumthang.",
            voice_text="Which dzongkhag will the business operate in? Say the name.",
            entity_captured="dzongkhag", next_node="BIZ-SVC-N5a",
        ))
        self.register(FlowNode(
            node_id="BIZ-SVC-N5a", name="CITIZENSHIP_CHECK", node_type=NodeType.INPUT,
            domain=Domain.BUSINESS, step="Step 3 of 5",
            bot_text="Are you a Bhutanese citizen or a foreign national? This determines your registration path.",
            voice_text="Are you a Bhutanese citizen or a foreign national? This determines your path.",
            entity_captured="citizenship_status", next_node="BIZ-SVC-N6a",
        ))
        self.register(FlowNode(
            node_id="BIZ-SVC-N6a", name="CHECKLIST_DELIVERY", node_type=NodeType.INFO,
            domain=Domain.BUSINESS, step="Step 4 of 5", triggers_rag=True,
            bot_text=(
                "To register a {business_type} in {dzongkhag}, you will need:\n"
                "{rag_answer}\n"
                "Would you like me to check if you have everything ready?"
            ),
            voice_text=(
                "To register a {business_type} in {dzongkhag}, you will need the following. "
                "{rag_answer} Would you like to check your readiness?"
            ),
            next_node="BIZ-SVC-CONFIRM",
        ))

    # ── Fallback & Error Recovery nodes (T2-06) ───────────────────────────────

    def _build_fallback_nodes(self):
        """
        8 fallback scenarios from T2-06_Fallback_Flow.xlsx.
        These are domain-agnostic — attached to sessions as needed.
        """
        fallbacks = [
            ("FB-01", "UNRECOGNISED_INTENT",
             "I am not sure which service you need yet. "
             "Please tell me in one short sentence what you want to do. "
             "I can also help you check documents, find an office, check status, or talk to a person.",
             "I am not sure which service you need yet. "
             "In one short sentence, tell me what you want to do."),
            ("FB-02", "INVALID_ENTITY_FORMAT",
             "I could not read that one detail clearly. "
             "Please share just that one item again. You can type it or say it.",
             "I could not read that one detail clearly. Please say or type just that one item again."),
            ("FB-03", "OUT_OF_SCOPE",
             "I can help only with the public services in this assistant. "
             "I can still help you find the right office, start again, or talk to a person.",
             "I can help only with the public services in this assistant. "
             "I can still help you find the right office, start again, or talk to a person."),
            ("FB-04", "REPEATED_FAILURE",
             "I do not want to keep you stuck. I will move you to human help now. "
             "Please use the phone number or email shown for this service. "
             "Keep this reference number: {ref_no}.",
             "I do not want to keep you stuck. I will move you to human help now. "
             "Please use the phone number or email shown. Keep your reference number: {ref_no}."),
            ("FB-05", "SERVICE_UNAVAILABLE",
             "I cannot complete this service right now. Please try again a little later. "
             "I can also help you start again, find the office, or talk to a person.",
             "I cannot complete this service right now. Please try again later."),
            ("FB-06", "MISSING_ENTITY",
             "I still need one detail to continue: {missing_entity_name}. "
             "Please share just that one item. "
             "If you do not have it now, I can help you start again or talk to a person.",
             "I still need one detail to continue: {missing_entity_name}. "
             "Please share just that one item."),
            ("FB-07", "OFFENSIVE_INPUT",
             "I am here to help with public service questions. "
             "We can continue with your service request, start again, or talk to a person.",
             "I am here to help with public service questions. "
             "We can continue, start again, or talk to a person."),
            ("FB-08", "INACTIVITY_TIMEOUT",
             "I am still here if you need help. Reply when you are ready.",
             "I am still here if you need help. Reply when you are ready."),
        ]
        for fb_id, name, bot_text, voice_text in fallbacks:
            self.register(FlowNode(
                node_id=fb_id, name=name, node_type=NodeType.FALLBACK,
                domain=Domain.GENERAL, bot_text=bot_text, voice_text=voice_text,
            ))


# =============================================================================
# SECTION 3 — Entity Extractor
# =============================================================================

class EntityExtractor:
    """
    Rule-based slot filler for Bhutan-specific entities.
    Extracts dzongkhag, permit_type, business_type, service_type, etc.
    from free text user inputs.
    """

    DZONGKHAGS = [
        "thimphu", "paro", "bumthang", "punakha", "wangdue", "haa", "dagana",
        "tsirang", "sarpang", "zhemgang", "trongsa", "mongar", "lhuentse",
        "trashigang", "pemagatshel", "samdrupjongkhar", "trashiyangste", "gasa",
        "chhukha", "samtse",
    ]

    PERMIT_TYPES = [
        "construction", "trade", "business", "building", "demolition",
        "land use", "forest", "tourism", "vehicle", "import", "export",
    ]

    BUSINESS_TYPES = [
        "sole proprietorship", "partnership", "private company", "corporation",
        "restaurant", "shop", "hotel", "farm", "manufacturing", "retail",
        "service", "agriculture", "tourism", "it", "technology",
    ]

    HEALTH_SERVICES = [
        "check-up", "checkup", "general", "vaccination", "immunization",
        "specialist", "emergency", "dental", "maternity", "mental health",
        "surgery", "outpatient", "referral",
    ]

    CITIZENSHIP = {
        "citizen": "bhutanese_citizen",
        "bhutanese": "bhutanese_citizen",
        "national": "bhutanese_citizen",
        "foreign": "foreign_national",
        "foreigner": "foreign_national",
        "expat": "foreign_national",
        "nri": "foreign_national",
        "joint venture": "joint_venture",
    }

    LICENCE_TYPES = [
        "trade licence", "trade license", "food licence", "food license",
        "professional licence", "professional license", "operating licence",
        "operating license",
    ]

    INTENT_MENU_MAP = {
        # Permit menu (OPN-03)
        "1": {"permits": "INT-001", "health": "INT-H-001", "business": "INT-B-001"},
        "2": {"permits": "INT-003", "health": "INT-H-002", "business": "INT-B-002"},
        "3": {"permits": "INT-004", "health": "INT-H-003", "business": "INT-B-003"},
        "4": {"permits": "INT-005", "health": "INT-H-004", "business": "INT-B-004"},
    }

    INTENT_KEYWORDS = {
        # Permit intents
        "INT-001": ["document", "requirement", "need", "what do i need", "checklist"],
        "INT-002": ["apply", "start", "new application", "submit"],
        "INT-003": ["ready", "readiness", "have everything", "check my documents"],
        "INT-004": ["status", "track", "reference", "application status"],
        "INT-005": ["office", "location", "where", "find", "address"],
        # Health intents
        "INT-H-001": ["facility", "hospital", "clinic", "nearest", "where", "bhu"],
        "INT-H-002": ["eligible", "eligibility", "free", "covered", "qualify"],
        "INT-H-003": ["understand", "explain", "simplify", "what does", "meaning"],
        "INT-H-004": ["appointment", "register", "book", "next step", "how to"],
        "INT-H-005": ["diagnose", "diagnosis", "treat", "medicine", "prescription", "cure"],
        # Business intents
        "INT-B-001": ["register", "registration", "how to register", "steps"],
        "INT-B-002": ["document", "checklist", "papers", "what do i need"],
        "INT-B-003": ["first", "sequence", "order", "licence first", "register first"],
        "INT-B-004": ["eligible", "eligibility", "can i", "allowed", "foreign"],
        "INT-B-005": ["office", "location", "where", "address", "find office"],
    }

    def extract(self, text: str, domain: Optional[Domain] = None) -> Dict[str, Any]:
        t = text.lower().strip()
        entities: Dict[str, Any] = {}

        # Dzongkhag
        for dz in self.DZONGKHAGS:
            if dz in t:
                entities["dzongkhag"] = dz.title()
                break

        # Permit type
        for pt in self.PERMIT_TYPES:
            if pt in t:
                entities["permit_type"] = pt
                break

        # Business type
        for bt in self.BUSINESS_TYPES:
            if bt in t:
                entities["business_type"] = bt
                break

        # Health service type
        for hs in self.HEALTH_SERVICES:
            if hs in t:
                entities["service_type"] = hs
                break

        # Citizenship
        for kw, val in self.CITIZENSHIP.items():
            if kw in t:
                entities["citizenship_status"] = val
                break

        # Licence type
        for lt in self.LICENCE_TYPES:
            if lt in t:
                entities["licence_type"] = lt
                break

        # Language preference — only set at LANGUAGE_SELECT node context
        # We check for "english"/"dzongkha" keywords to avoid "1" falsely triggering this
        if any(w in t for w in ["english", "one for english", "eng"]) and "dzongkha" not in t:
            entities["language_pref"] = "EN"
        elif any(w in t for w in ["dzongkha", "two for dzongkha", "dz"]):
            entities["language_pref"] = "DZ"
        elif t.strip() == "1":
            entities["language_pref"] = "EN"   # bare "1" at LANGUAGE_SELECT = English
        elif t.strip() == "2":
            entities["language_pref"] = "DZ"

        # Yes / No confirmation
        if any(w in t for w in ["yes", "yeah", "correct", "right", "yep"]):
            entities["user_confirmation"] = True
        elif any(w in t for w in ["no", "nope", "wrong", "rephrase"]):
            entities["user_confirmation"] = False

        # Intent from menu number
        menu_num = re.search(r"\b([1-4])\b", t)
        if menu_num and domain:
            num = menu_num.group(1)
            domain_key = domain.value if domain else "general"
            # Map domain value to short key used in INTENT_MENU_MAP
            short = {"permits": "permits", "health": "health", "business": "business"}.get(domain_key)
            if short and num in self.INTENT_MENU_MAP:
                entities["intent_id"] = self.INTENT_MENU_MAP[num].get(short)

        return entities

    def classify_intent(self, text: str, domain: Optional[Domain] = None) -> Tuple[Optional[str], float]:
        """Return (intent_id, confidence) from user free text."""
        t = text.lower()
        scores: Dict[str, int] = {}

        # Filter by domain prefix
        domain_prefix: Optional[str] = None
        if domain == Domain.PERMIT:
            domain_prefix = "INT-"
        elif domain == Domain.HEALTH:
            domain_prefix = "INT-H-"
        elif domain == Domain.BUSINESS:
            domain_prefix = "INT-B-"

        for intent_id, keywords in self.INTENT_KEYWORDS.items():
            # Only consider intents matching current domain
            if domain_prefix:
                if domain == Domain.PERMIT and ("INT-H-" in intent_id or "INT-B-" in intent_id):
                    continue
                if domain == Domain.HEALTH and ("INT-B-" in intent_id or (intent_id.startswith("INT-") and not intent_id.startswith("INT-H-"))):
                    continue
                if domain == Domain.BUSINESS and ("INT-H-" in intent_id or (intent_id.startswith("INT-") and not intent_id.startswith("INT-B-"))):
                    continue
            score = sum(1 for kw in keywords if kw in t)
            if score > 0:
                scores[intent_id] = score

        # Check menu number → intent mapping
        menu_match = re.search(r"\b([1-4])\b", t)
        if menu_match and domain:
            num = menu_match.group(1)
            mapping = {"1": 0, "2": 1, "3": 2, "4": 3}
            intent_lists = {
                Domain.PERMIT:   ["INT-001",   "INT-003",   "INT-004",   "INT-005"],
                Domain.HEALTH:   ["INT-H-001", "INT-H-002", "INT-H-003", "INT-H-004"],
                Domain.BUSINESS: ["INT-B-001", "INT-B-002", "INT-B-003", "INT-B-005"],
            }
            idx = mapping.get(num)
            ilist = intent_lists.get(domain, [])
            if idx is not None and idx < len(ilist):
                scores[ilist[idx]] = scores.get(ilist[idx], 0) + 3  # strong signal

        if not scores:
            return None, 0.0

        best = max(scores, key=scores.get)
        conf = min(1.0, scores[best] / 2.0)
        return best, conf

    @staticmethod
    def intent_summary(intent_id: Optional[str], entities: Dict[str, Any]) -> str:
        """Human-readable intent summary for INTENT_CONFIRM node."""
        summaries = {
            "INT-001": "permit requirements & documents",
            "INT-002": "starting a new permit application",
            "INT-003": "checking if your documents are ready",
            "INT-004": "checking your application status",
            "INT-005": "finding the permit office",
            "INT-H-001": "finding a health facility",
            "INT-H-002": "health service eligibility",
            "INT-H-003": "understanding health information in simpler terms",
            "INT-H-004": "appointment & registration steps",
            "INT-H-005": "a health question (general information only)",
            "INT-B-001": "business registration steps",
            "INT-B-002": "required documents & checklist",
            "INT-B-003": "what to do first — licensing or registration",
            "INT-B-004": "eligibility for business registration",
            "INT-B-005": "finding the business registration office",
        }
        base = summaries.get(intent_id or "", "your request")

        # Enrich with collected entities
        parts = []
        if "permit_type" in entities:
            parts.append(f"{entities['permit_type']} permit")
        if "business_type" in entities:
            parts.append(f"{entities['business_type']}")
        if "dzongkhag" in entities:
            parts.append(f"in {entities['dzongkhag']}")
        if parts:
            return f"{base} ({', '.join(parts)})"
        return base


# =============================================================================
# SECTION 4 — Domain Guardrails
# =============================================================================

class DomainGuardrail:
    """
    SAFETY_DEC  — Health domain: block medical advice / diagnosis queries.
    SCOPE_DEC   — Business domain: flag complex foreign/multi-sector queries.
    """

    UNSAFE_HEALTH_KEYWORDS = [
        "diagnose", "diagnosis", "treat", "treatment", "prescription",
        "medicine", "drug", "cure", "symptom", "do i have", "am i sick",
        "what disease", "is it cancer", "should i take",
    ]

    COMPLEX_BIZ_KEYWORDS = [
        "foreign investment", "joint venture", "multi-sector", "fdi",
        "100%", "fully foreign", "foreign company", "international",
        "manufacturing licence", "special economic zone", "sez",
    ]

    def safety_check(self, text: str, intent_id: Optional[str]) -> bool:
        """Returns True if SAFE to answer (no medical advice territory)."""
        t = text.lower()
        # INT-H-005 (open health query) always requires check
        if intent_id == "INT-H-005":
            return not any(kw in t for kw in self.UNSAFE_HEALTH_KEYWORDS)
        # All other health intents are safe by design
        return True

    def scope_check(self, text: str, intent_id: Optional[str], entities: Dict) -> bool:
        """Returns True if IN SCOPE for standard self-service."""
        t = text.lower()
        if any(kw in t for kw in self.COMPLEX_BIZ_KEYWORDS):
            return False
        if entities.get("citizenship_status") == "foreign_national":
            # Foreign nationals need officer consultation
            return False
        return True


# =============================================================================
# SECTION 5 — RAG Document Knowledge Base (domain knowledge for retrieval)
# =============================================================================

def build_domain_knowledge_chunks() -> List[DocumentChunk]:
    """
    Build DocumentChunk knowledge base from the conversational flow content.
    These chunks are retrieved at SERVICE_HANDOFF nodes to populate
    {rag_answer} in bot responses.
    """
    chunks: List[DocumentChunk] = []

    # ── Permit domain chunks ──────────────────────────────────────────────────
    permit_chunks = [
        ("prm_doc_construction", "permits",
         "Construction permit in Thimphu requires: 1) Your CID card "
         "2) Land Thram (land ownership document) "
         "3) No Objection Certificate (NOC) from the landlord "
         "4) Completed application form from the dzongkhag office. "
         "Visit the Department of Urban Development and Engineering Services.",
         ("construction", "permit", "documents", "thimphu"),
         {"title": "Construction Permit — Thimphu"}),

        ("prm_doc_trade", "permits",
         "Trade permit requires: 1) CID card 2) Business registration certificate "
         "3) Proof of address 4) Completed trade licence form. "
         "Submit at your dzongkhag office, Monday to Friday, 9am to 5pm.",
         ("trade", "permit", "documents", "licence"),
         {"title": "Trade Permit Requirements"}),

        ("prm_status_001", "permits",
         "To check your permit application status, share your application reference number. "
         "Status can also be checked at the dzongkhag administration office or "
         "online via the Bhutan e-Government portal.",
         ("status", "application", "track", "reference"),
         {"title": "Permit Application Status"}),

        ("prm_office_thimphu", "permits",
         "Permit office in Thimphu: Department of Urban Development, "
         "Thadrak, Thimphu. Phone: +975-2-xxx-xxxx. "
         "Open Monday to Friday, 9am to 5pm. "
         "Bring original documents and two photocopies.",
         ("office", "thimphu", "location", "address"),
         {"title": "Permit Office — Thimphu"}),

        ("prm_office_paro", "permits",
         "Permit office in Paro: Paro Dzongkhag Administration office, "
         "Paro town centre. Phone: +975-8-xxx-xxxx. "
         "Open Monday to Friday, 9am to 5pm.",
         ("office", "paro", "location"),
         {"title": "Permit Office — Paro"}),

        ("prm_readiness", "permits",
         "Before visiting the permit office, ensure you have: "
         "1) Valid CID card (original + photocopy) "
         "2) Land Thram or proof of ownership "
         "3) NOC from landlord (if applicable) "
         "4) Completed application form (available at dzongkhag office or e-Gov portal) "
         "5) Application fee payment receipt.",
         ("readiness", "checklist", "office visit", "documents"),
         {"title": "Permit Readiness Checklist"}),
    ]

    # ── Health domain chunks ──────────────────────────────────────────────────
    health_chunks = [
        ("hlt_facility_thimphu", "health",
         "Nearest health facility in Thimphu: Jigme Dorji Wangchuck National Referral Hospital (JDWNRH), "
         "Gongphel Lam, Thimphu. Phone: 112 or +975-2-xxx-xxxx. "
         "Open Monday to Saturday, 8am to 4pm. "
         "Emergency: 24 hours.",
         ("facility", "hospital", "thimphu", "referral"),
         {"title": "JDWNRH — Thimphu"}),

        ("hlt_facility_paro", "health",
         "Nearest health facility in Paro: Paro District Hospital, "
         "Paro town. Phone: +975-8-xxx-xxxx. "
         "Open Monday to Saturday, 8am to 4pm.",
         ("facility", "hospital", "paro"),
         {"title": "Paro District Hospital"}),

        ("hlt_bhu_general", "health",
         "Basic Health Units (BHUs) are available in every block across Bhutan. "
         "Services include: outpatient consultation, immunization, antenatal care, "
         "family planning, and referral to district hospitals. "
         "BHU services are free for Bhutanese citizens.",
         ("bhu", "basic health unit", "free", "eligibility"),
         {"title": "BHU Services — General"}),

        ("hlt_certificate", "health",
         "To get a medical certificate: "
         "1) Visit your nearest hospital or BHU with your CID card "
         "2) See the doctor on duty "
         "3) Request a certificate for your specific purpose (employment, travel, school) "
         "Fee: Nu. 50–200 depending on type. Usually ready the same day.",
         ("certificate", "medical certificate", "documents", "cid"),
         {"title": "Medical Certificate Process"}),

        ("hlt_vaccination", "health",
         "Vaccination / immunization services are available at all BHUs and district hospitals. "
         "For children: bring the child health card. "
         "For adults: bring your CID card. "
         "National vaccination programme is free for all citizens.",
         ("vaccination", "immunization", "child", "free"),
         {"title": "Vaccination Services"}),

        ("hlt_safety_guardrail", "health",
         "The Bhutan Health Assistant provides navigation and information only. "
         "It does not provide medical advice, diagnosis, or treatment recommendations. "
         "For medical guidance, please visit your nearest BHU or call 112.",
         ("safety", "guardrail", "medical advice"),
         {"title": "Health Safety Guardrail"}),
    ]

    # ── Business Registration domain chunks ───────────────────────────────────
    business_chunks = [
        ("biz_reg_sole", "business",
         "To register a sole proprietorship in Bhutan: "
         "1) Valid CID card 2) Completed application form (from MOEA or online) "
         "3) Proof of address 4) Initial capital declaration "
         "5) No-objection letter (if operating from rented premises). "
         "Submit at your dzongkhag MOEA office or online at www.moea.gov.bt.",
         ("register", "sole proprietorship", "documents", "checklist"),
         {"title": "Sole Proprietorship Registration"}),

        ("biz_reg_partnership", "business",
         "To register a partnership in Bhutan: "
         "1) CID cards of all partners 2) Partnership agreement (notarized) "
         "3) Completed registration form 4) Proof of business address "
         "5) Capital declaration signed by all partners.",
         ("register", "partnership", "documents"),
         {"title": "Partnership Registration"}),

        ("biz_licence_trade", "business",
         "For a trade licence in Bhutan: "
         "1) Business registration certificate 2) Valid CID card "
         "3) Completed licence application form 4) Premises inspection certificate "
         "5) Tax clearance certificate. "
         "Apply at the dzongkhag administration office.",
         ("trade licence", "licence", "documents", "checklist"),
         {"title": "Trade Licence Requirements"}),

        ("biz_eligibility_citizen", "business",
         "Bhutanese citizens are eligible to register as: "
         "sole proprietor, partnership, or private limited company. "
         "No minimum capital requirement for sole proprietorship. "
         "Partnership requires at least 2 partners (all Bhutanese). "
         "Private company requires a board of directors.",
         ("eligibility", "citizen", "bhutanese", "register"),
         {"title": "Eligibility — Bhutanese Citizen"}),

        ("biz_office_thimphu", "business",
         "Business registration office in Thimphu: "
         "Ministry of Economic Affairs (MoEA), "
         "Tashichhodzong vicinity, Thimphu. "
         "Phone: +975-2-xxx-xxxx. "
         "Open Monday to Friday, 9am to 5pm. "
         "Online: www.moea.gov.bt",
         ("office", "thimphu", "moea", "location"),
         {"title": "MoEA Office — Thimphu"}),

        ("biz_sequencing", "business",
         "Process sequencing for business in Bhutan: "
         "Step 1: Register your business first with MOEA. "
         "Step 2: Obtain your trade or operating licence AFTER registration. "
         "Step 3: Register with the Department of Revenue and Customs for tax. "
         "Registration usually takes 3–5 working days.",
         ("sequence", "first", "order", "steps", "process"),
         {"title": "Business Registration Sequence"}),

        # ── Industry FAQ chunks from industry.gov.bt FAQ ─────────────────────────

        ("biz_faq_industry_scales", "business",
         "Industry scale in Bhutan is defined by fixed capital investment and employment. "
         "Cottage-scale industries have initial fixed capital investment below Nu. 1 million "
         "and employ up to 4 people. Small-scale industries have fixed capital investment "
         "from Nu. 1 million to Nu. 10 million and employ up to 19 people. "
         "Medium industries have fixed capital investment between Nu. 10 million and "
         "Nu. 100 million. Large industries have fixed capital investment above "
         "Nu. 100 million.",
         ("industry scale", "cottage", "small", "medium", "large", "capital investment"),
         {"title": "FAQ - Industry Scale Classification"}),

        ("biz_faq_industry_sectors", "business",
         "There are three industry sectors in Bhutan: "
         "1) Production and Manufacturing sector, "
         "2) Service sector, "
         "3) Contract business.",
         ("industry sectors", "manufacturing", "service", "contract business"),
         {"title": "FAQ - Industry Sectors in Bhutan"}),

        ("biz_faq_eligibility", "business",
         "A Bhutanese citizen who is 18 years of age and above can legally own and operate "
         "an industry in Bhutan. An industry can produce or manufacture goods or provide "
         "essential services. Retail and wholesale of goods are not considered industry "
         "activities and fall under the Trading Sector.",
         ("eligibility", "own industry", "operate industry", "bhutanese citizen", "age"),
         {"title": "FAQ - Eligibility to Own and Operate an Industry"}),

        ("biz_faq_start_business", "business",
         "The first step towards starting a business in Bhutan is to have a good business idea. "
         "Ideally, the business should be in an industry the person understands and should "
         "reflect their strengths, goals, and an opportunity gap they have identified. "
         "After that, the person should prepare a proper business plan.",
         ("start business", "first step", "business idea", "business plan", "industry"),
         {"title": "FAQ - First Step to Start a Business"}),

        ("biz_faq_business_plan", "business",
         "A business plan is needed to explain business ideas and models to lenders, investors, "
         "and potential partners. If support is needed, applicants may seek help from friends, "
         "consultants, or relevant government agencies such as the Department of Industry or "
         "the Regional Office of Industry, Commerce and Employment.",
         ("business plan", "lenders", "investors", "department of industry", "regional office"),
         {"title": "FAQ - Business Plan"}),

        ("biz_faq_license_required", "business",
         "To start an industry in Bhutan, an industrial license or registration certificate "
         "is required from the Department of Industry or the Regional Office of Industry, "
         "Commerce and Employment. Entertainment-related businesses are an exception and "
         "should approach the Bhutan InfoCom and Media Authority, BICMA.",
         ("license", "registration certificate", "industrial license", "department of industry", "bicma"),
         {"title": "FAQ - Industry License Requirement"}),

        ("biz_faq_documents_clearances", "business",
         "To obtain a license or registration certificate, applicants should have a valid "
         "security clearance certificate. Depending on the business activity, soft copies "
         "of sector clearances may also be required. Some industries may need clearances "
         "from agencies such as the Department of Environment and Climate Change, Bhutan "
         "Food and Drug Authority, Thromde offices, or Dzongkhag offices. A business plan "
         "may also be required where applicable.",
         ("documents", "clearances", "security clearance", "sector clearance", "business plan"),
         {"title": "FAQ - Required Documents and Clearances"}),

        ("biz_faq_processing_time", "business",
         "If all required information, documents, and clearances are submitted, the industry "
         "license or registration certificate will be issued within one or two working days.",
         ("processing time", "license", "registration certificate", "working days"),
         {"title": "FAQ - License Processing Time"}),

        ("biz_faq_license_cost", "business",
         "The cost of an industry license or registration certificate depends on the scale "
         "of the business. Cottage-scale businesses have no registration or renewal cost, "
         "but a late renewal penalty of Nu. 20 applies if not renewed on time. "
         "Small-scale businesses pay Nu. 3,000. Medium-scale businesses pay Nu. 6,000. "
         "Large-scale businesses pay Nu. 12,000. Restaurant, contract tour operator, "
         "and entertainment license fees depend on the specific activity.",
         ("cost", "fees", "license cost", "cottage", "small", "medium", "large"),
         {"title": "FAQ - License Cost"}),

        ("biz_faq_allowed_activities", "business",
         "Most business activities are allowed in Bhutan, except activities that violate "
         "relevant laws, threaten national security or public order, harm public health, "
         "environment, morals, or culture, involve arms, ammunition, explosives, hazardous "
         "chemicals, imported waste, pornographic materials, gambling and betting, or "
         "tobacco and tobacco-based products.",
         ("allowed activities", "restricted activities", "prohibited", "industry"),
         {"title": "FAQ - Allowed and Restricted Industrial Activities"}),

        ("biz_faq_incentives", "business",
         "The government provides fiscal and non-fiscal incentives to industries, including "
         "tax holidays, import duty exemption on machines, equipment, primary raw materials "
         "and packaging materials, preferential buying by the government, and focused "
         "capacity development support.",
         ("incentives", "tax holiday", "import duty exemption", "government support"),
         {"title": "FAQ - Government Incentives for Industries"}),

        ("biz_faq_land_support", "business",
         "If an entrepreneur has a good industry idea but does not have land, support may "
         "include leasing State Forest Reserve Land or government land for a fixed period, "
         "depending on industry priority, viability, and location. Applicants may also apply "
         "for industrial plots in existing industrial estates and service centers, depending "
         "on availability.",
         ("land", "government land", "industrial plot", "industrial estate", "support"),
         {"title": "FAQ - Land Support for Industry"}),

        ("biz_faq_medium_large_requirement", "business",
         "The first requirement for starting a medium or large industry is to prepare a "
         "comprehensive business plan. The business plan acts as the foundation for the "
         "industry and serves as a reference point for approval and licensing.",
         ("medium industry", "large industry", "business plan", "first requirement"),
         {"title": "FAQ - Medium and Large Industry Requirement"}),

        ("biz_faq_business_plan_guidelines", "business",
         "Business plan guidelines depend on the nature of the industry activity. The "
         "Department has published guidelines for production and manufacturing industries "
         "and service industries. These guidelines can be downloaded from the Legislations "
         "section of the Department website.",
         ("business plan guidelines", "manufacturing", "service industry", "legislations"),
         {"title": "FAQ - Business Plan Guidelines"}),

        ("biz_faq_additional_documents_medium_large", "business",
         "For medium and large industries, apart from a business plan, applicants must obtain "
         "and submit sectoral clearances and documents from relevant agencies depending on "
         "the industry activity. Examples include environmental clearance, location clearance, "
         "technical clearances, currency clearances, and conditional food safety clearances. "
         "Applicants may contact the Large Industry Promotion Division of the Department of "
         "Industry for advice on relevant clearances and documents.",
         ("medium industry", "large industry", "sectoral clearances", "environmental clearance", "food safety"),
         {"title": "FAQ - Additional Documents for Medium and Large Industry"}),
    ]

    # ── Fallback / generic chunks ─────────────────────────────────────────────
    fallback_chunks = [
        ("fb_human_contact", "general",
         "For human assistance: "
         "General helpline: 17 (toll-free) "
         "Health emergencies: 112 "
         "Business registration: +975-2-xxx-xxxx | biz@gov.bt "
         "Health services: +975-2-xxx-xxxx | health@gov.bt "
         "Permits: +975-2-xxx-xxxx | help@gov.bt "
         "Operating hours: Monday to Friday, 9am to 5pm.",
         ("human", "contact", "escalation", "phone", "email"),
         {"title": "Human Contact Directory"}),

        ("fb_out_of_scope", "general",
         "The Bhutan Voice-First Public Service Assistant covers: "
         "permit services, health information, and business registration & licensing. "
         "For other services, please visit www.gov.bt or call the general helpline at 17.",
         ("out of scope", "other services", "general"),
         {"title": "Scope of Services"}),
    ]

    all_raw = [
        *[(id_, "permits", *rest) for id_, _, *rest in permit_chunks],
        *[(id_, "health", *rest) for id_, _, *rest in health_chunks],
        *[(id_, "business", *rest) for id_, _, *rest in business_chunks],
        *[(id_, "general", *rest) for id_, _, *rest in fallback_chunks],
    ]

    combined = permit_chunks + health_chunks + business_chunks + fallback_chunks
    for chunk_id, domain, text, tags, metadata in combined:
        priority = 4 if domain != "general" else 1
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=text,
            source=f"bhutan_flow_docs/{domain}",
            domain=domain,
            doc_type="conversational_flow",
            priority=priority,
            phase="live",
            tags=tuple(tags),
            metadata=metadata,
        ))

    return chunks


# =============================================================================
# SECTION 6 — DPR Retriever (fixed from original pipeline)
# =============================================================================

class DPRRetriever:
    """DPR dense retrieval — uses proper DPRQuestionEncoder / DPRContextEncoder."""

    def __init__(
        self,
        query_model_name: str  = "facebook/dpr-question_encoder-single-nq-base",
        passage_model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
    ):
        if DPRQuestionEncoder is None or torch is None:
            raise ImportError("transformers and torch are required")

        self.query_tokenizer   = DPRQuestionEncoderTokenizer.from_pretrained(query_model_name)
        self.query_model       = DPRQuestionEncoder.from_pretrained(query_model_name)
        self.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_model_name)
        self.passage_model     = DPRContextEncoder.from_pretrained(passage_model_name)

        self.chunks:     List[DocumentChunk]   = []
        self.embeddings: Optional[np.ndarray]  = None
        self.index = None

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        vecs = self._embed_passages([c.text for c in chunks])
        self.embeddings = vecs if self.embeddings is None \
                          else np.vstack([self.embeddings, vecs])
        self.chunks.extend(list(chunks))
        self._rebuild_faiss()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or self.embeddings is None:
            return []
        q_emb = self._embed_query([query])
        idxs  = self._filter_indices(filters)
        if not idxs:
            return []

        # ── Step 1: Dense semantic retrieval — fetch expanded candidate pool ──
        # Retrieve top_k * 3 candidates so the reranker has headroom to promote
        # the correct chunk even if DPR's embedding doesn't rank it perfectly.
        vecs       = self.embeddings[idxs]
        cos_scores = vecs @ q_emb[0]
        fetch_k    = min(len(idxs), top_k * 3)
        top_raw    = np.argsort(cos_scores)[::-1][:fetch_k]

        # ── Step 2: Keyword-aware reranking layer ─────────────────────────────
        # Mirrors your teammate's FallbackRetriever fix, now applied post-DPR.
        # Priority weighting (+0.02 * chunk.priority) is also added here —
        # it was missing from DPRRetriever entirely.
        results: List[Tuple[DocumentChunk, float]] = []
        for i in top_raw:
            chunk = self.chunks[idxs[i]]
            score = float(cos_scores[i])
            score += 0.02 * chunk.priority          # domain-priority signal
            score += _keyword_boost(query, chunk)   # FAQ-intent reranking
            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _embed_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_tokenizer, self.query_model)

    def _embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.passage_tokenizer, self.passage_model)

    @staticmethod
    def _encode(texts: Sequence[str], tokenizer, model) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device).eval()
        with torch.no_grad():
            enc = tokenizer(
                list(texts), padding=True, truncation=True,
                return_tensors="pt", max_length=256,
            )
            enc  = {k: v.to(device) for k, v in enc.items()}
            out  = model(**enc)
            # DPR encoders expose pooler_output — NOT last_hidden_state
            emb  = out.pooler_output
            emb  = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def _rebuild_faiss(self) -> None:
        if faiss is None or self.embeddings is None:
            self.index = None
            return
        d     = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(self.embeddings.astype(np.float32))
        self.index = index

    def _filter_indices(self, filters: Optional[Dict[str, Any]]) -> List[int]:
        if not filters:
            return list(range(len(self.chunks)))
        return [i for i, c in enumerate(self.chunks) if self._match(c, filters)]

    @staticmethod
    def _match(chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        for k, v in filters.items():
            if v is None:
                continue
            if k == "tags":
                wanted = set(v if isinstance(v, (list, tuple, set)) else [v])
                if not wanted.intersection(set(chunk.tags)):
                    return False
                continue
            if getattr(chunk, k, None) != v:
                return False
        return True

# =============================================================================
# SECTION 5b — Shared Keyword-Boost Reranking Layer
# =============================================================================

def _keyword_boost(query: str, chunk: "DocumentChunk") -> float:
    """
    Keyword-aware score adjustment for high-intent FAQ queries.

    Applied as a post-retrieval reranking layer on BOTH DPRRetriever and
    FallbackRetriever so that short, exact-intent queries (cost, processing
    time, documents, eligibility, first step) always surface the correct
    FAQ chunk regardless of which retriever is active.

    Boost magnitudes are calibrated to override DPR cosine-score differences
    (~0.05–0.15 range) without completely suppressing semantic ranking for
    ambiguous queries.
    """
    q = query.lower()

    title    = str(chunk.metadata.get("title", "")).lower()
    tags     = " ".join(chunk.tags).lower()
    text     = chunk.text.lower()
    chunk_id = chunk.id.lower()

    haystack = f"{chunk_id} {title} {tags} {text}"

    boost = 0.0

    # ── Cost / fee queries ────────────────────────────────────────────────────
    if any(term in q for term in ["cost", "fee", "fees", "charge", "charges", "price"]):
        if "license_cost" in chunk_id or "license cost" in title:
            boost += 1.5
        elif any(term in haystack for term in
                 ["cost", "fee", "fees", "nu.", "small-scale businesses pay",
                  "medium-scale businesses pay"]):
            boost += 0.5

    # ── Processing time queries ───────────────────────────────────────────────
    if any(term in q for term in
           ["how long", "processing time", "take", "takes", "working days", "issued"]):
        if "processing_time" in chunk_id or "processing time" in title:
            boost += 1.5
        elif any(term in haystack for term in
                 ["working days", "issued within one or two working days"]):
            boost += 0.5

    # ── Documents / clearances queries ───────────────────────────────────────
    if any(term in q for term in
           ["document", "documents", "clearance", "clearances", "required"]):
        if ("documents_clearances" in chunk_id
                or "required documents" in title
                or "clearances" in title):
            boost += 1.6
        elif any(term in haystack for term in
                 ["security clearance", "sector clearances", "required documents"]):
            boost += 0.5

        # Prevent processing-time chunk from hijacking document-intent queries
        if "processing_time" in chunk_id or "processing time" in title:
            boost -= 0.8

    # ── Eligibility queries ───────────────────────────────────────────────────
    if any(term in q for term in
           ["eligible", "eligibility", "can i", "allowed", "own", "operate"]):
        if "eligibility" in chunk_id or "eligibility" in title:
            boost += 1.2
        elif any(term in haystack for term in ["bhutanese citizen", "own and operate"]):
            boost += 0.5

    # ── First step / start business queries ──────────────────────────────────
    if any(term in q for term in
           ["first step", "start a business", "starting a business", "business idea"]):
        if "start_business" in chunk_id or "first step" in title:
            boost += 1.2
        elif any(term in haystack for term in ["business idea", "business plan"]):
            boost += 0.5

    # ── License requirement queries ───────────────────────────────────────────
    # "Do I need a license?" / "Is a license required?" — must not be confused
    # with cost or processing-time queries which also mention "license".
    if any(term in q for term in
           ["need a license", "need license", "require a license", "do i need",
            "license required", "is a license"]):
        if "license_required" in chunk_id or "license requirement" in title:
            boost += 1.2
        elif any(term in haystack for term in
                 ["industrial license", "registration certificate",
                  "department of industry"]):
            boost += 0.4

    return boost


#---------------------------
class FallbackRetriever:
    """
    Simple TF-IDF-style retriever — used when torch/DPR is not available,
    or for fast prototyping without model downloads.
    Adds lightweight keyword-aware boosting for FAQ-specific intents.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks.extend(list(chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        results = []

        for chunk in self.chunks:
            if filters:
                skip = False
                for k, v in filters.items():
                    if v is None:
                        continue
                    if k == "tags":
                        wanted = set(v if isinstance(v, (list, tuple, set)) else [v])
                        if not wanted.intersection(set(chunk.tags)):
                            skip = True
                            break
                    elif getattr(chunk, k, None) != v:
                        skip = True
                        break
                if skip:
                    continue

            title = str(chunk.metadata.get("title", "")).lower()
            tags = " ".join(chunk.tags).lower()
            text = chunk.text.lower()
            chunk_id = chunk.id.lower()

            searchable_text = f"{chunk_id} {title} {tags} {text}"
            c_terms = set(re.findall(r"[a-z0-9]+", searchable_text))

            overlap = len(q_terms.intersection(c_terms))
            score = overlap / (len(q_terms) + 1e-9)
            score += 0.02 * chunk.priority

            score += self._keyword_boost(query, chunk)

            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _keyword_boost(query: str, chunk: DocumentChunk) -> float:
        """Delegates to the shared module-level _keyword_boost function."""
        return _keyword_boost(query, chunk)
    
# =============================================================================
# SECTION 7 — Main Conversational RAG Engine
# =============================================================================

class ConversationalRAGEngine:
    """
    Integrates the conversational flow graphs with DPR-RAG retrieval.

    Turn loop:
      1. Detect / advance node from session state
      2. Extract entities from user input
      3. Apply domain guardrails (safety / scope)
      4. If triggers_rag: retrieve relevant DocumentChunks and inject into response
      5. Apply fallback logic (FB-01 to FB-08) when needed
      6. Return TurnResponse with bot_text, voice_text, next_node
    """

    INACTIVITY_WARN_SECS  = 180   # 3 minutes → FB-08 first nudge
    INACTIVITY_CLOSE_SECS = 480   # 8 minutes → session close

    def __init__(self, retriever=None):
        self.flow      = FlowRegistry()
        self.extractor = EntityExtractor()
        self.guardrail = DomainGuardrail()
        self.retriever = retriever  # DPRRetriever or FallbackRetriever

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, use_dpr: bool = True) -> "ConversationalRAGEngine":
        """Build engine and index all domain knowledge chunks."""
        chunks = build_domain_knowledge_chunks()

        if use_dpr and torch is not None:
            try:
                retriever = DPRRetriever()
                retriever.add(chunks)
            except Exception as e:
                print(f"[WARN] DPR init failed ({e}), falling back to TF-IDF retriever.")
                retriever = FallbackRetriever()
                retriever.add(chunks)
        else:
            retriever = FallbackRetriever()
            retriever.add(chunks)

        return cls(retriever=retriever)

    # ── Session management ────────────────────────────────────────────────────

    def new_session(
        self,
        domain: Optional[Domain] = None,
        channel: str = "chat",
    ) -> ConversationSession:
        domain_prefix = {
            Domain.PERMIT:   "PRM",
            Domain.HEALTH:   "HLT",
            Domain.BUSINESS: "BIZ",
        }
        prefix = domain_prefix.get(domain, "PRM") if domain else "PRM"
        session = ConversationSession(
            domain=domain,
            current_node_id=f"{prefix}-OPN-01",
            channel=Channel(channel),
        )
        return session

    # ── Main turn ─────────────────────────────────────────────────────────────

    def turn(self, session: ConversationSession, user_input: str) -> TurnResponse:
        """Process one user turn and return the bot response."""

        now = time.time()

        # ── Auto-advance past non-interactive (GREETING/INFO) nodes ──────────
        # These nodes serve text when bot speaks but don't consume user input.
        # When the user responds, we must be at the first interactive node.
        _advance_types = {NodeType.GREETING, NodeType.INFO}
        _node = self.flow.get(session.current_node_id)
        while _node and _node.node_type in _advance_types and not _node.triggers_rag:
            _nxt = _node.next_node
            if not _nxt:
                break
            session.current_node_id = _nxt
            _node = self.flow.get(_nxt)

        # ── Inactivity check (FB-08) ──────────────────────────────────────────
        idle = now - session.last_activity
        if idle > self.INACTIVITY_CLOSE_SECS:
            session.is_terminal = True
            return self._fb_response("FB-08", session,
                                     override_text=(
                                         f"I will close this chat for now. "
                                         f"Keep this reference number: {session.session_id}. "
                                         f"You can come back any time or use the help contact shown here."
                                     ))
        if idle > self.INACTIVITY_WARN_SECS and session.turn_count > 0:
            # Gentle nudge — don't advance node
            return self._fb_response("FB-08", session)

        session.last_activity = now
        session.turn_count   += 1

        # ── Offensive input guard (FB-07) ─────────────────────────────────────
        if self._is_offensive(user_input):
            return self._fb_response("FB-07", session)

        # ── Out-of-scope check (FB-03) ────────────────────────────────────────
        if self._is_out_of_scope(user_input) and session.domain is not None:
            return self._fb_response("FB-03", session)

        # ── Get current node ──────────────────────────────────────────────────
        node = self.flow.get(session.current_node_id)
        if node is None:
            # Recover: go back to capability intro
            node = self.flow.get(self._capability_node(session.domain))

        # ── Extract entities from user input ──────────────────────────────────
        new_entities = self.extractor.extract(user_input, domain=session.domain)
        session.entities.update({k: v for k, v in new_entities.items() if v is not None})

        # ── Classify intent (at INTENT_PROMPT nodes) ──────────────────────────
        if node.node_type == NodeType.INPUT and node.name == "INTENT_PROMPT":
            return self._handle_intent_prompt(session, node, user_input)

        # ── Handle CONFIRM nodes (yes/no routing) ─────────────────────────────
        if node.node_type == NodeType.CONFIRM and node.name == "INTENT_CONFIRM":
            return self._handle_intent_confirm(session, node, user_input)

        # ── Handle entity-collection INPUT nodes (service query) ──────────────
        if node.node_type == NodeType.INPUT and node.entity_captured:
            return self._handle_entity_collect(session, node, user_input, new_entities)

        # ── Handle LANGUAGE_SELECT ────────────────────────────────────────────
        if node.name == "LANGUAGE_SELECT":
            return self._handle_language_select(session, node, user_input)

        # ── Handle SERVICE_HANDOFF / INFO nodes with RAG ──────────────────────
        if node.triggers_rag:
            return self._handle_rag_node(session, node)

        # ── Default: serve node response and advance ──────────────────────────
        next_id  = node.next_node or session.current_node_id
        bot_text  = self._render(node.bot_text, session)
        voice_text = self._render(node.voice_text, session)
        session.current_node_id = next_id
        session.is_terminal     = node.is_terminal

        return TurnResponse(
            bot_text=bot_text,
            voice_text=voice_text,
            next_node_id=next_id,
            entities_collected=dict(session.entities),
            is_terminal=node.is_terminal,
            session_ref=session.session_id,
            step_label=node.step or "",
        )

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_language_select(
        self, session: ConversationSession, node: FlowNode, user_input: str
    ) -> TurnResponse:
        lang = session.entities.get("language_pref", "EN")

        if lang == "DZ":
            dz_node_id = node.alt_next.get("DZ", "")
            dz_node    = self.flow.get(dz_node_id)
            if dz_node:
                session.current_node_id = dz_node_id
                session.is_terminal     = True
                return TurnResponse(
                    bot_text    = dz_node.bot_text,
                    voice_text  = dz_node.voice_text,
                    next_node_id= dz_node_id,
                    entities_collected=dict(session.entities),
                    is_terminal = True,
                    session_ref = session.session_id,
                )

        # English — serve CAPABILITY_INTRO text, advance session to INTENT_PROMPT
        cap_id   = node.next_node or self._capability_node(session.domain)
        cap_node = self.flow.get(cap_id)
        # Advance past INFO node so next user turn goes to INTENT_PROMPT
        intent_id   = cap_node.next_node if cap_node else cap_id
        session.current_node_id = intent_id
        # Combine CAPABILITY_INTRO + INTENT_PROMPT in single message
        intent_node = self.flow.get(intent_id)
        combined_text  = (cap_node.bot_text if cap_node else "") 
        combined_voice = (cap_node.voice_text if cap_node else "")
        return TurnResponse(
            bot_text    = combined_text,
            voice_text  = combined_voice,
            next_node_id= intent_id,
            entities_collected=dict(session.entities),
            session_ref = session.session_id,
        )

    def _handle_intent_prompt(
        self, session: ConversationSession, node: FlowNode, user_input: str
    ) -> TurnResponse:
        intent_id, conf = self.extractor.classify_intent(user_input, domain=session.domain)

        # Also check if an entity already gave us an intent_id from menu
        if not intent_id:
            intent_id = session.entities.get("intent_id")

        if intent_id and conf >= 0.2:
            # Intent parsed ✓
            session.intent_id    = intent_id
            session.fallback_count = 0
            summary = self.extractor.intent_summary(intent_id, session.entities)

            # Go to INTENT_CONFIRM
            confirm_node_id = node.next_node  # e.g. PRM-OPN-05
            confirm_node    = self.flow.get(confirm_node_id)
            session.current_node_id = confirm_node_id

            bot_text   = self._render(confirm_node.bot_text, session,
                                      extra={"intent_summary": summary})
            voice_text = self._render(confirm_node.voice_text, session,
                                      extra={"intent_summary": summary})
            return TurnResponse(
                bot_text=bot_text, voice_text=voice_text,
                next_node_id=confirm_node_id,
                entities_collected=dict(session.entities),
                session_ref=session.session_id,
            )

        # Intent NOT parsed → fallback
        session.fallback_count += 1
        if session.fallback_count >= 2:
            return self._escalate(session)

        fb_node_id = node.alt_next.get("fail", "")
        fb_node    = self.flow.get(fb_node_id)
        session.current_node_id = fb_node_id or session.current_node_id
        return TurnResponse(
            bot_text    = fb_node.bot_text if fb_node else self.flow.get("FB-01").bot_text,
            voice_text  = fb_node.voice_text if fb_node else self.flow.get("FB-01").voice_text,
            next_node_id= fb_node_id or session.current_node_id,
            entities_collected=dict(session.entities),
            fallback_id = "FB-01",
            session_ref = session.session_id,
        )

    def _handle_intent_confirm(
        self, session: ConversationSession, node: FlowNode, user_input: str
    ) -> TurnResponse:
        confirmed = session.entities.get("user_confirmation")

        if confirmed is False:
            # User said "No — let me rephrase" → back to INTENT_PROMPT
            intent_node_id = node.alt_next.get("no", "")
            intent_node    = self.flow.get(intent_node_id)
            session.current_node_id = intent_node_id
            return TurnResponse(
                bot_text    = intent_node.bot_text if intent_node else "",
                voice_text  = intent_node.voice_text if intent_node else "",
                next_node_id= intent_node_id,
                entities_collected=dict(session.entities),
                session_ref = session.session_id,
            )

        # ── Domain guardrails ─────────────────────────────────────────────────
        if node.safety_check and session.domain == Domain.HEALTH:
            is_safe = self.guardrail.safety_check(
                session.entities.get("user_intent", ""), session.intent_id
            )
            if not is_safe:
                safe_node_id = node.alt_next.get("unsafe", "HLT-OPN-SAFE")
                safe_node    = self.flow.get(safe_node_id)
                session.current_node_id = safe_node_id
                session.is_terminal     = True
                return TurnResponse(
                    bot_text    = safe_node.bot_text if safe_node else "",
                    voice_text  = safe_node.voice_text if safe_node else "",
                    next_node_id= safe_node_id,
                    entities_collected=dict(session.entities),
                    is_terminal = True,
                    session_ref = session.session_id,
                )

        if node.scope_check and session.domain == Domain.BUSINESS:
            in_scope = self.guardrail.scope_check(
                session.entities.get("user_intent", ""),
                session.intent_id,
                session.entities,
            )
            if not in_scope:
                cx_node_id = node.alt_next.get("complex", "BIZ-OPN-COMPLEX")
                cx_node    = self.flow.get(cx_node_id)
                session.current_node_id = cx_node_id
                session.is_terminal     = True
                return TurnResponse(
                    bot_text    = cx_node.bot_text if cx_node else "",
                    voice_text  = cx_node.voice_text if cx_node else "",
                    next_node_id= cx_node_id,
                    entities_collected=dict(session.entities),
                    is_terminal = True,
                    session_ref = session.session_id,
                )

        # ── Intent confirmed ✓ → route to service entity-collection or RAG ───
        session.confirmed_intent = session.intent_id
        next_id   = self._route_to_service_node(session)
        next_node = self.flow.get(next_id)
        session.current_node_id = next_id

        if next_node and next_node.triggers_rag:
            return self._handle_rag_node(session, next_node)

        return TurnResponse(
            bot_text    = self._render(next_node.bot_text, session) if next_node else "",
            voice_text  = self._render(next_node.voice_text, session) if next_node else "",
            next_node_id= next_id,
            entities_collected=dict(session.entities),
            session_ref = session.session_id,
            step_label  = next_node.step or "" if next_node else "",
        )

    def _handle_entity_collect(
        self, session: ConversationSession, node: FlowNode,
        user_input: str, new_entities: Dict
    ) -> TurnResponse:
        """Handle slot-filling nodes in the service query flow."""
        entity_key = node.entity_captured
        if not entity_key:
            return self._advance_node(session, node)

        # Check if entity was successfully extracted
        if entity_key in session.entities:
            # Got it — advance
            session.field_fail_count[entity_key] = 0
        else:
            # Missing entity (FB-02 / FB-06)
            fail_count = session.field_fail_count.get(entity_key, 0) + 1
            session.field_fail_count[entity_key] = fail_count

            if fail_count >= 2:
                # FB-06: still missing after 2 prompts
                if session.fallback_count >= 2:
                    return self._escalate(session)
                fb06 = self.flow.get("FB-06")
                bt = fb06.bot_text.format(missing_entity_name=entity_key.replace("_", " "))
                vt = fb06.voice_text.format(missing_entity_name=entity_key.replace("_", " "))
                return TurnResponse(
                    bot_text=bt, voice_text=vt,
                    next_node_id=session.current_node_id,
                    entities_collected=dict(session.entities),
                    fallback_id="FB-06",
                    session_ref=session.session_id,
                    step_label=node.step or "",
                )
            else:
                # FB-02: invalid format — re-ask
                fb02 = self.flow.get("FB-02")
                return TurnResponse(
                    bot_text    = fb02.bot_text,
                    voice_text  = fb02.voice_text,
                    next_node_id= session.current_node_id,
                    entities_collected=dict(session.entities),
                    fallback_id = "FB-02",
                    session_ref = session.session_id,
                    step_label  = node.step or "",
                )

        return self._advance_node(session, node)

    def _advance_node(self, session: ConversationSession, node: FlowNode) -> TurnResponse:
        """Advance to next node after successful entity collection."""
        next_id   = node.next_node or session.current_node_id
        next_node = self.flow.get(next_id)
        session.current_node_id = next_id

        if next_node and next_node.triggers_rag:
            return self._handle_rag_node(session, next_node)

        bot_text   = self._render(next_node.bot_text, session) if next_node else ""
        voice_text = self._render(next_node.voice_text, session) if next_node else ""
        session.is_terminal = next_node.is_terminal if next_node else False

        return TurnResponse(
            bot_text=bot_text, voice_text=voice_text,
            next_node_id=next_id,
            entities_collected=dict(session.entities),
            is_terminal=session.is_terminal,
            session_ref=session.session_id,
            step_label=next_node.step or "" if next_node else "",
        )

    def _handle_rag_node(
        self, session: ConversationSession, node: FlowNode
    ) -> TurnResponse:
        """Retrieve relevant content and inject into the node template."""
        query  = self._build_rag_query(session)
        domain = session.domain.value if session.domain else "general"

        hits = []
        if self.retriever:
            try:
                hits = self.retriever.search(
                    query, top_k=3,
                    filters={"domain": domain},
                )
            except Exception as e:
                # FB-05: service unavailable
                fb05 = self.flow.get("FB-05")
                return TurnResponse(
                    bot_text    = fb05.bot_text,
                    voice_text  = fb05.voice_text,
                    next_node_id= session.current_node_id,
                    entities_collected=dict(session.entities),
                    fallback_id = "FB-05",
                    session_ref = session.session_id,
                )

        # Build RAG answer from retrieved chunks
        if hits:
            rag_answer = "\n".join(
                f"• {chunk.text.strip()}"
                for chunk, score in hits
                if score > 0.05
            )
            citations = [
                {"rank": i + 1, "source": c.source, "score": round(s, 4)}
                for i, (c, s) in enumerate(hits)
            ]
        else:
            rag_answer = (
                "Please visit the relevant dzongkhag office for specific details. "
                f"Your reference: {session.session_id}"
            )
            citations = []

        # Advance session to confirmation node
        next_id   = node.next_node or session.current_node_id
        session.current_node_id = next_id

        bot_text   = self._render(node.bot_text,   session, extra={"rag_answer": rag_answer})
        voice_text = self._render(node.voice_text,  session, extra={"rag_answer": rag_answer})

        return TurnResponse(
            bot_text    = bot_text,
            voice_text  = voice_text,
            next_node_id= next_id,
            entities_collected=dict(session.entities),
            triggers_rag = True,
            rag_context  = rag_answer,
            rag_citations= citations,
            session_ref  = session.session_id,
            step_label   = node.step or "",
        )

    # ── Routing helpers ───────────────────────────────────────────────────────

    def _route_to_service_node(self, session: ConversationSession) -> str:
        """
        Map confirmed intent + domain → first entity-collection node,
        or directly to SERVICE_HANDOFF if entities are already present.
        """
        d = session.domain
        i = session.intent_id or ""

        # Permit
        if d == Domain.PERMIT:
            if "permit_type" not in session.entities:
                return "PRM-SVC-N2"          # collect permit_type first
            if "dzongkhag" not in session.entities:
                return "PRM-SVC-N3"
            return "PRM-SVC-N4"              # trigger RAG delivery

        # Health
        if d == Domain.HEALTH:
            if i == "INT-H-003":             # explain document (Path B — 3 steps)
                return "HLT-SVC-N4"
            if "dzongkhag" not in session.entities:
                return "HLT-SVC-N2"
            if "service_type" not in session.entities:
                return "HLT-SVC-N3"
            return "HLT-SVC-N4"

        # Business
        if d == Domain.BUSINESS:
            if i in ("INT-B-005",):          # find office — just need dzongkhag
                if "dzongkhag" not in session.entities:
                    return "BIZ-SVC-N4a"
                return "BIZ-SVC-N6a"
            if "business_type" not in session.entities:
                return "BIZ-SVC-N3a"
            if "dzongkhag" not in session.entities:
                return "BIZ-SVC-N4a"
            if i in ("INT-B-001", "INT-B-002") and "citizenship_status" not in session.entities:
                return "BIZ-SVC-N5a"
            return "BIZ-SVC-N6a"

        return self._handoff_node(d)

    def _handoff_node(self, domain: Optional[Domain]) -> str:
        return {
            Domain.PERMIT:   "PRM-OPN-06",
            Domain.HEALTH:   "HLT-OPN-06",
            Domain.BUSINESS: "BIZ-OPN-06",
        }.get(domain, "PRM-OPN-06")

    def _capability_node(self, domain: Optional[Domain]) -> str:
        return {
            Domain.PERMIT:   "PRM-OPN-03",
            Domain.HEALTH:   "HLT-OPN-03",
            Domain.BUSINESS: "BIZ-OPN-03",
        }.get(domain, "PRM-OPN-03")

    def _escalate(self, session: ConversationSession) -> TurnResponse:
        esc_ids = {
            Domain.PERMIT:   "PRM-OPN-ESC",
            Domain.HEALTH:   "HLT-OPN-ESC",
            Domain.BUSINESS: "BIZ-OPN-ESC",
        }
        esc_id   = esc_ids.get(session.domain, "PRM-OPN-ESC")
        esc_node = self.flow.get(esc_id)
        session.current_node_id = esc_id
        session.is_terminal     = True
        bot_text   = self._render(esc_node.bot_text,   session) if esc_node else ""
        voice_text = self._render(esc_node.voice_text, session) if esc_node else ""
        return TurnResponse(
            bot_text=bot_text, voice_text=voice_text,
            next_node_id=esc_id,
            entities_collected=dict(session.entities),
            is_terminal=True,
            fallback_id="FB-04",
            session_ref=session.session_id,
        )

    # ── Fallback helpers ──────────────────────────────────────────────────────

    def _fb_response(
        self, fb_id: str, session: ConversationSession,
        override_text: Optional[str] = None,
    ) -> TurnResponse:
        fb_node = self.flow.get(fb_id)
        if fb_node:
            bt = override_text or self._render(fb_node.bot_text, session)
            vt = self._render(fb_node.voice_text, session)
        else:
            bt = vt = override_text or "I'm having trouble. Please try again or call 17."
        return TurnResponse(
            bot_text=bt, voice_text=vt,
            next_node_id=session.current_node_id,
            entities_collected=dict(session.entities),
            fallback_id=fb_id,
            session_ref=session.session_id,
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _build_rag_query(self, session: ConversationSession) -> str:
        """Build a retrieval query from session context."""
        parts = []
        if session.confirmed_intent:
            intent_summary = self.extractor.intent_summary(
                session.confirmed_intent, session.entities
            )
            parts.append(intent_summary)
        for key in ("permit_type", "business_type", "service_type", "dzongkhag", "health_topic"):
            val = session.entities.get(key)
            if val:
                parts.append(str(val))
        if session.domain:
            parts.append(session.domain.value)
        return " ".join(parts) if parts else "bhutan public service"

    @staticmethod
    def _render(template: str, session: ConversationSession,
                extra: Optional[Dict] = None) -> str:
        """Fill {placeholders} in bot script templates."""
        ctx = {
            "ref_no":          session.session_id,
            "office_name":     "the relevant dzongkhag office",
            "office_address":  "please call for address details",
            "next_step":       "visit the office shown above",
            "health_office_number": "112",
            "document_list":   session.entities.get("document_list", "the checklist above"),
            "business_type":   session.entities.get("business_type", "[business type]"),
            "dzongkhag":       session.entities.get("dzongkhag", "[your dzongkhag]"),
            "permit_type":     session.entities.get("permit_type", "[permit type]"),
            "service_type":    session.entities.get("service_type", "[service type]"),
            "intent_summary":  "[your request]",
            "rag_answer":      "",
        }
        if extra:
            ctx.update(extra)
        try:
            return template.format(**ctx)
        except KeyError:
            return template

    @staticmethod
    def _is_offensive(text: str) -> bool:
        offensive = ["idiot", "stupid", "fuck", "shit", "asshole", "useless bot"]
        t = text.lower()
        return any(w in t for w in offensive)

    @staticmethod
    def _is_out_of_scope(text: str) -> bool:
        out_of_scope = [
            "cricket", "football", "movie", "recipe", "joke",
            "stock price", "forex", "weather", "election",
        ]
        t = text.lower()
        return any(w in t for w in out_of_scope)


# =============================================================================
# SECTION 8 — Demo
# =============================================================================

def run_demo():
    """
    Demo: walks through a complete permit + health + business session
    using the FallbackRetriever (no model download needed).
    """
    print("=" * 70)
    print(" Bhutan Voice-First Public Service Assistant — RAG + Flow Demo")
    print("=" * 70)

    engine = ConversationalRAGEngine.build(use_dpr=False)  # fast mode

    scenarios = [
        # (domain, channel, turns)
        (Domain.PERMIT, "chat", [
            "Hello",                                          # WELCOME
            "1",                                              # LANGUAGE_SELECT → English
            "I want to know what documents I need for a construction permit in Thimphu",
            "Yes",                                            # INTENT_CONFIRM
        ]),
        (Domain.HEALTH, "voice", [
            "Hi",
            "1",                                              # English
            "Where is the nearest hospital in Paro",
            "yes",
        ]),
        (Domain.BUSINESS, "chat", [
            "Hello",
            "1",
            "I want to register a sole proprietorship restaurant in Thimphu",
            "yes",
            "bhutanese citizen",                              # citizenship_status
        ]),
    ]

    for domain, channel, turns in scenarios:
        session = engine.new_session(domain=domain, channel=channel)
        print(f"\n{'─'*70}")
        print(f"DOMAIN: {domain.value.upper()}  |  CHANNEL: {channel}  |  REF: {session.session_id}")
        print(f"{'─'*70}")

        # Serve the WELCOME node first
        welcome_node = engine.flow.get(session.current_node_id)
        if welcome_node:
            print(f"\n[BOT] {welcome_node.bot_text}\n")

        for user_turn in turns:
            print(f"[USER] {user_turn}")
            resp = engine.turn(session, user_turn)
            print(f"[BOT]  {resp.bot_text}")
            if resp.step_label:
                print(f"       {resp.step_label}")
            if resp.rag_citations:
                print(f"       📚 Sources: {[c['source'] for c in resp.rag_citations]}")
            if resp.fallback_id:
                print(f"       ⚠  Fallback: {resp.fallback_id}")
            print()
            if resp.is_terminal:
                print(f"[SESSION CLOSED — Ref: {session.session_id}]")
                break


#if __name__ == "__main__":
    #run_demo()

if __name__ == "__main__":
    engine = ConversationalRAGEngine.build(use_dpr=False)

    test_questions = [
        "What is the first step to start a business in Bhutan?",
        "Do I need a license to start an industry in Bhutan?",
        "What documents are required to get an industry license?",
        "How long does it take to get an industry license?",
        "What is the cost of an industry license in Bhutan?",
    ]

    for q in test_questions:
        print("=" * 80)
        print("Q:", q)

        hits = engine.retriever.search(
            q,
            top_k=5,
            filters={"domain": "business"}
        )

        for i, (chunk, score) in enumerate(hits, start=1):
            print(f"\nRank {i} | Score: {score:.4f}")
            print("ID:", chunk.id)
            print("Title:", chunk.metadata.get("title"))
            print("Source:", chunk.source)
            print("Text:", chunk.text[:500])
