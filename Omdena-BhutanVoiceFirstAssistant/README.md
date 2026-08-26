# Bhutan Voice-First Public Service Assistant

Omdena sprint (Team 4) — a voice-first assistant that helps Bhutanese citizens navigate public services (permits, health, business registration) in Dzongkha and English. Contribution here: request classification and the RAG-backed conversational flow engine.

## Components

**`classifier.py`** — multi-backend intent classifier. Classifies each user turn along 4 dimensions (in-scope, safety, request type, service) with confidence scores, and escalates to a human when confidence is low.

- Backends: Gemini API, local LM Studio (OpenAI-compatible, no API key), or pure keyword matching
- Fixes over the original single-backend version: local-model JSON parsing (malformed/truncated output), few-shot examples, `facility_lookup` request type, expanded service labels, confidence-gated escalation, KB relevance threshold

**`data_retrieval.py`** — full conversational flow engine: scripted dialog graphs (`FlowRegistry`) per domain (permits/health/business), entity extraction, safety/scope guardrails, and DPR-style dense retrieval (`DPRRetriever`, falls back to `FallbackRetriever` TF-IDF-style matching when torch/faiss aren't available) over a domain knowledge base.

## Setup

```bash
uv sync
cp .env.example .env   # fill in GEMINI_API_KEY (only needed for the "gemini" backend)
```

## Run

Classifier — interactive chat or demo scenarios:

```bash
python classifier.py --backend keyword --mode demo
python classifier.py --backend lmstudio --mode interactive
python classifier.py --backend gemini --mode demo
```

Flow engine demo (dense retrieval on domain knowledge base, no model download required):

```bash
python data_retrieval.py
```

## Project layout

```
classifier.py       intent classifier (Gemini / LM Studio / keyword)
data_retrieval.py    conversational flow graphs + RAG retrieval engine
data/                classifier test set
docs/                sprint updates, task lists, presentation, phase docs
notebooks/           earlier Colab snapshot of the classifier (superseded by classifier.py)
```

## Status

Sprint deliverable. `classifier.py` and `data_retrieval.py` are self-contained and not yet wired together into a single entrypoint — that integration (classifier feeding into the flow engine's RAG node) is the next step.
