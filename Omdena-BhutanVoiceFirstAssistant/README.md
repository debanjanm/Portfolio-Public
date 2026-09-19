# Bhutan Voice-First Public Service Assistant

Omdena sprint (Team 4) — a voice-first assistant that helps Bhutanese citizens navigate public services (permits, health, business registration) in Dzongkha and English. Contribution here: the request classifier.

## `classifier.py`

Multi-backend intent classifier. Classifies each user turn along 4 dimensions (in-scope, safety, request type, service) with confidence scores, and escalates to a human when confidence is low.

- Backends: Gemini API, local LM Studio (OpenAI-compatible, no API key), or pure keyword matching
- Fixes over the original single-backend version: local-model JSON parsing (malformed/truncated output), few-shot examples, `facility_lookup` request type, expanded service labels, confidence-gated escalation, KB relevance threshold

## Setup

```bash
uv sync
cp .env.example .env   # fill in GEMINI_API_KEY (only needed for the "gemini" backend)
```

## Run

```bash
python classifier.py --backend keyword --mode demo
python classifier.py --backend lmstudio --mode interactive
python classifier.py --backend gemini --mode demo
```

## Project layout

```
classifier.py       intent classifier (Gemini / LM Studio / keyword)
data/                classifier test set
docs/                sprint updates, task lists, presentation, phase docs
notebooks/           earlier Colab snapshot of the classifier (superseded by classifier.py)
```

## Status

Sprint deliverable — classifier is complete and self-contained.
