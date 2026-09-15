# 🚀 AutoStream AI Agent

A production-style conversational AI agent built using **LangGraph + LangChain** that demonstrates:

* Intent classification
* RAG-based question answering
* Deterministic lead capture (no extractor / regex / guessing)
* Tool invocation with strict state validation

---

## 📌 Problem Statement

AutoStream is a fictional AI-powered video automation platform. The goal of this project is to build a conversational AI agent that:

1. Answers user questions about AutoStream (pricing, features, etc.) using **RAG**
2. Detects **high purchase intent**
3. Collects lead information step-by-step (name, email, platform)
4. Captures the lead using a mock backend tool

The agent must behave naturally, avoid incorrect data extraction, and ensure tool calls only happen when all required data is present.

---

## 🧠 Architecture Explanation

This project uses LangGraph instead of prompt-only agents or AutoGen because the problem is best modeled as a state-driven workflow. The agent must move through well-defined stages such as intent classification, information retrieval (RAG), lead qualification, and tool execution. LangGraph allows explicit modeling of these steps as a graph, making agent behavior deterministic, debuggable, and safe for production use.

State management is the core of the architecture. A shared AgentState object tracks conversation messages, detected intent, lead details (name, email, platform), and whether the lead has already been captured. The state is updated before each graph execution, ensuring that every node in the graph operates on the latest user input. This prevents premature tool execution and avoids accidental overwriting of previously collected data.

Unlike extractor-based approaches, lead information is collected deterministically: the agent only assigns a value when it explicitly asks for that field. Tool execution is guarded by strict state checks, ensuring it runs exactly once and only when all required fields are present. This design prioritizes correctness, transparency, and production reliability over prompt-only flexibility.

## 🏗️ Architecture Overview

```
User Input 
   ↓
Message State
   ↓
Intent Classifier ──┐
   │                │
   │ INFO           │ HIGH_INTENT
   ↓                ↓
RAG Answer      Lead Collection
                    ↓
              Tool Execution
              
```

---

## 📂 Project Structure

```
autostream_agent/
│
├── agent/
│   ├── graph.py              # LangGraph state machine
│   ├── intent_classifier.py  # Intent classification logic
│   ├── rag.py                # RAG pipeline (FAISS / Chroma)
│   └── prompts.py            # Centralized prompts
|   |__ tools.py            
|   |__ state.py
|   |__ __init__.py           
│
├── data/
│   └── knowledge_base.txt    # AutoStream documents
│
├── main.py                   # CLI entry point
├── requirements.txt          # stores dependencies  
└── README.md
|__ .env                      # Store API's
```

---

## 🧾 State Definition

The agent state tracks:

* Conversation messages
* Detected intent
* Lead fields (name, email, platform)
* Lead captured flag

This ensures:

* Idempotent execution
* No duplicate tool calls
* Predictable behavior

---

## 🔍 Intent Classification

The agent classifies user intent into:

* `INFO` → Answer using RAG
* `HIGH_INTENT` → Start lead capture flow

Classification is LLM-driven but constrained to known labels.

---

## 📚 RAG Implementation

* Knowledge base is chunked and embedded
* Stored in FAISS
* Retriever uses top-k semantic search
* Context injected into a strict prompt

If the answer is not found in context, the agent explicitly responds with uncertainty.

---

## 🧑‍💼 Lead Capture Flow

The agent collects:

1. Name
2. Email
3. Platform (YouTube, Instagram, etc.)

### Important Rules

* Fields are assigned **before** graph execution
* Tool executes **only when all fields are present**
* Tool fires exactly once

---
## ▶️ How to Run the Project Locally 

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY=your_api_key
```

### 4. Run the Agent

```bash
python main.py
```

---

## 💬 Sample Conversation

```
You: Tell me about pricing
Agent: AutoStream offers two pricing plans...

You: I want pro
Agent: Great! May I have your name?

You: Rahul
Agent: Thanks! Could you share your email address?

You: rahul.patil@gmail.com
Agent: Which platform do you create content on?

You: Instagram
Lead captured successfully: Rahul, rahul.patil@gmail.com, Instagram
Agent: You’re all set! Your lead has been captured successfully.
```

## 📲 WhatsApp Deployment (Webhook-Based)

To integrate this agent with WhatsApp, the CLI interface can be replaced with a webhook-based API using FastAPI or Flask.

1. WhatsApp Provider
Use WhatsApp Cloud API or Twilio WhatsApp. Incoming user messages are sent to a webhook en

2. Webhook Endpoint
The webhook receives the message payload (user phone number and message text). The phone number is used as a session identifier.

3. State Persistence
The agent state is stored in Redis or a database, keyed by the user’s phone number, allowing multi-turn conversations.

4. Agent Invocation
The incoming message is appended to the stored state and passed to graph.invoke(state).

5. Response Delivery
The agent’s response is extracted from the updated state and sent back to WhatsApp using the provider’s API.

This architecture cleanly separates messaging infrastructure from agent logic, allowing the same LangGraph-based agent to run across CLI, WhatsApp, or web chat with minimal changes.
---

---

## 🧪 Error Handling & Guards

* Prevents premature tool execution
* Avoids overwriting already-captured fields
* Handles unknown or irrelevant queries gracefully

---

## 🚀 Possible Extensions

* Email validation
* Platform normalization
* Lead stage tracking enum
* FastAPI / WhatsApp integration
* Persistent memory (Redis / DB)
* Unit tests for graph nodes

---

## 🚀 Tech Stack

* Python 3.9+
* LangChain
* LangGraph
* OpenAI LLMs
* ChromaDB (local vector store)


---

