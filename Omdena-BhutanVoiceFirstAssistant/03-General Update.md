## 🔹 Team 1 — Research & Problem Understanding

* Identified 3 initial use cases:
  * Transportation
  * Healthcare
  * Starting a Business

* Created Git PR covering:
  * Problem understanding
  * Use case shortlist
  * Feasibility hypotheses

* Conducted interviews with Bhutanese users; awaiting feedback
* Will finalize 2 use cases post interview insights

### 📌 Updated Direction:

* Primary focus: Work Permit use case
* Transportation & Healthcare → parked for now (can be explored later)

### 📌 Ongoing Work:

* Deep research on:
  * Work permit process
  * Required documents
  * Departments involved
  * Step-by-step user journey

* Bhutanese participants requested to support with ground-level insights

### 💡 Key Pain Points Identified:

* Complex and confusing process
* Multiple forms and departments
* Errors → restart entire process

## 🔹 Team 2 — Conversation Design

* Defined:
  * User queries
  * Intents
  * Conversation flows

* Uploaded work to GitHub

### 📌 Next Steps:

* Refine flows based on:
  * Work permit research from Team 1

* Build structured conversational journeys for prototype

## 🔹 Team 3 — Voice & Chat Prototyping

### 📌 Key Feedback from Project Lead (Arjun):

* Focus on Voice Agent (not just chatbot)

* System should support:
  * Voice input (Dzongkha)
  * Speech → Text → Processing
  * Output in Dzongkha (voice)

### 📌 Core Questions Raised:

* Model approach: Train vs API-based?
* Can we build multiple AI agents (per department)?
* Can agents:
  * Share context
  * Redirect users across departments

### 💡 Example:

* Business agent → routes to Tax agent when needed

### 📌 Suggested Tech Direction:

* Agent-based architecture
* Use tools like LangChain and n8n

## 🔹 Existing System Context

* Bhutan’s G2C portal has a chatbot
* Limitation:
  * Text-only interaction
* Opportunity:
  * Build voice-first system in Dzongkha

## 🔹 Proposed System Architecture

* Dzongkha Voice Input
  → Speech-to-Text
  → English Text Processing
  → Intent Detection
  → Response Generation
  → Translate to Dzongkha
  → Voice Output

* Govt may provide Dzongkha ↔️ English translation API

## 🔹 Additional Technical Suggestions

* Ayush:

  * Use AssemblyAI (speech, summaries, sentiment, multilingual support)
  * Consider:
    * Network latency
    * Voice transcription challenges

* Mark:

  * Suggested RAG (Retrieval-Augmented Generation)
  * Use government documents/policies as knowledge base

## 🔹 Team 4 — Agents & Workflow

* Focus: Intent classification & routing
* Proposed flow:

  1. Check if query is in scope
  2. Check if safe to answer
  3. Process further

(More details to be collected from their presentation)

## 🔹 Key Use Case Insight (from Sonam)

* Highlighted timber permit use case of rural Bhutan
* Key challenges:
  * Low literacy in rural areas
  * Lack of awareness of process
* Even though info exists on portals, users need:
  * Step-by-step voice guidance
  * Help with document preparation

## 🔹 Key Direction from Project Lead

* Focus on Work Permit (priority use case)
* Build separate prototypes for different departments
* Explore multi-agent system with shared context

## 🔹 Week 2 Plan

* First 2–3 days:
  * Team 1 → Deep research on work permit
  * Team 2 → Refine flows based on research
  * Team 3 → Start early prototypes using available flows

* Goal:
  * Build agent-based voice prototype for work permit

## 🔹 Important Considerations

* Estimated usage: 22,000–30,000 users/week
* Need to evaluate:

  * Cost of models
  * Scalability

## 🔹 Key Dependencies Across Teams

* Team 3 depends on:
  * Team 2 → conversation flows
  * Team 4 → intent classification logic
* Team 2 depends on:
  * Team 1 → research insights

## 🔹 Action Points (especially for Team 3)

* Align on:
  * Voice-first architecture
  * Multi-agent design (department-wise)
* Start:
  * Early prototype (even basic version)
  * Cost analysis of models