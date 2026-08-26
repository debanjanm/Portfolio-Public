# Mental Health Chatbot

RAG-based mental health support chatbot built during an Omdena sprint (Team 5). Retrieves grounded context from mental-health reference PDFs and answers via Gemini, so responses stay tied to source material instead of hallucinating.

## Architecture

```
User query
    │
    ▼
FAISS retriever (src/core/retriever.py)   ← embeds query, pulls top-k chunks
    │
    ▼
RAG prompt (src/core/prompter.py)         ← injects context + chat history
    │
    ▼
Gemini generator (src/core/generator.py)  ← per-user/conversation message history
    │
    ▼
Response
```

`src/pipeline/flow.py` (`MentalChatbot`) wires retriever → prompt → generator together. `src/core/indexer.py` builds the FAISS index from source PDFs; `src/core/categorizer.py` is a standalone DistilBERT sentiment classifier not yet wired into the main pipeline.

## Setup

```bash
uv sync
cp .env.example .env   # fill in GOOGLE_API_KEY
```

Build the vector index once (from inside `src/`):

```bash
cd src
python core/indexer.py
```

## Run

CLI:

```bash
cd src
python main.py --query "I've been feeling anxious lately" --user-id alice --conversation-id session1
```

Streamlit UI:

```bash
cd src
streamlit run frontend.py
```

## Project layout

```
src/
  common/    config + model loading
  core/      retriever, generator, prompter, indexer, categorizer, summarizer
  pipeline/  MentalChatbot orchestration
  utils/     chat history helpers
docs/        sprint presentations and background reading
notebooks/   Colab R&D notebooks from the sprint (exploratory, not production code)
artifacts/   PDF source data, FAISS index, prompt templates, finetuned model
```

## Status

Sprint deliverable — retrieval + generation pipeline works end to end; sentiment classification (`categorizer.py`) is standalone and not yet integrated into the conversation flow.
