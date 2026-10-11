# NovaTech Solutions — AI Support & Lead Qualification Agent (Local Setup)

An AI support and lead qualification system built with **LangGraph**, **Google Gemini 3.8 Flash**, **Gemini Text Embedding**, **Pinecone Vector Database**, and **MongoDB Atlas**.

---

## 🏛️ Local Architecture

```
                    ┌─────────────────────────┐
                    │      Incoming User      │
                    │   (Web Widget / CLI)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │       LangGraph       │
                     │     Orchestrator      │
                     └───────┬───────┬───────┘
                             │       │
             ┌───────────────┘       └────────────────┐
             ▼                                        ▼
   ┌────────────────────┐                   ┌────────────────────┐
   │     QnA Agent      │                   │     Lead Agent     │
   │ (Gemini 3.8 Flash) │                   │ (Gemini 3.8 Flash) │
   └─────────┬──────────┘                   └─────────┬──────────┘
             │                                        │
             ▼                                        ▼
   ┌────────────────────┐                   ┌────────────────────┐
   │ SearchRanker Tool  │                   │  LeadSaver Tool    │
   └─────────┬──────────┘                   └─────────┬──────────┘
             │                                        │
             ▼                                        ▼
   ┌────────────────────┐                   ┌────────────────────┐
   │  Pinecone Index    │                   │   MongoDB Atlas    │
   │(novetech-knowledge)│                   │  (novatech_db)     │
   └────────────────────┘                   └────────────────────┘
```

---

## 📁 Project Structure (Local)

```
AI-SUPPORT-AGENT/
├── .env                              # Environment credentials & keys
├── .env.example                      # Template for configuration
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── ChatBot/                          # Core Agent Service
│   ├── .env                          # Service environment configuration
│   ├── add_temp_records.py           # Ingestion script: indexes PDF KB into Pinecone
│   ├── agents.py                     # LangGraph Orchestrator, QnA Agent & Lead Agent
│   ├── chatbot_inference.py          # FastAPI application & Interactive CLI chat runner
│   ├── tools.py                      # search_ranker (Pinecone) & lead_saver (MongoDB)
│   ├── test_tools.py                 # Standalone tests for both tools
│   ├── logger_utils.py               # Structured logger utility
│   └── requirements.txt              # Service dependencies
│
├── frontend/                         # Modern Glassmorphic Web Widget
│   ├── index.html                    # Chat interface & MongoDB leads viewer modal
│   ├── style.css                     # Premium dark-mode glassmorphic styling
│   └── chatbot.js                    # Chat client & real-time API connector
│
└── knowledge_base/
    └── novatech_knowledge_base.pdf   # Official NovaTech Solutions Knowledge Base
```

---

## ⚙️ Environment Configuration (`.env`)

```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-text-embedding-001

# Pinecone Vector Store Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=novetech-knowledge

# MongoDB Atlas Configuration
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.9dumvdg.mongodb.net
MONGODB_DATABASE=novatech_db
MONGODB_COLLECTION=leads

# Application Settings
LOG_LEVEL=INFO
```

---

## 🚀 How to Run Locally

### 1. Ingest Knowledge Base into Pinecone
Indexes `novatech_knowledge_base.pdf` into Pinecone using `gemini-text-embedding-001`:

```bash
python ChatBot/add_temp_records.py
```

### 2. Verify Tools Standalone
Tests `search_ranker` (Pinecone query) and `lead_saver` (MongoDB Atlas write):

```bash
python ChatBot/test_tools.py
```

### 3. Interactive CLI Chat Mode
Chat with the agent directly in your terminal:

```bash
python ChatBot/chatbot_inference.py
```

### 4. Run the Local Web Server & UI
Starts the FastAPI backend and serves the frontend widget:

```bash
python ChatBot/chatbot_inference.py server
# or:
uvicorn ChatBot.chatbot_inference:app --reload --port 8000
```

- Open **`http://127.0.0.1:8000`** in your browser to interact with the full web UI.
- Click the **"Leads"** button in the header to view saved leads from MongoDB Atlas in real time.
- View interactive Swagger API documentation at **`http://127.0.0.1:8000/docs`**.
