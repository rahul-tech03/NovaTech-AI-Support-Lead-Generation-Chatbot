"""
chatbot_inference.py: Inference engine and FastAPI server for NovaTech AI Support Agent.
Supports:
1. FastAPI REST endpoints (/api/chat, /api/leads, /api/health)
2. Interactive Terminal CLI chat mode for rapid local testing
3. Embedded static hosting for frontend/ widget
"""

import os
import sys
import uuid
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from logger_utils import setup_logger
from agents import compiled_graph
from tools import get_mongo_collection

# Load environment configuration
load_dotenv(find_dotenv(usecwd=True))
logger = setup_logger("Inference")

# ==========================================
# 1. FASTAPI APPLICATION SETUP
# ==========================================

app = FastAPI(
    title="NovaTech AI Support Agent API",
    description="Multi-agent customer support & lead qualification system built with LangGraph, Gemini, Pinecone, and MongoDB Atlas.",
    version="1.0.0"
)

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. REQUEST & RESPONSE MODELS
# ==========================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's input query or message", min_length=1)
    session_id: Optional[str] = Field(default=None, description="Unique session/thread identifier for conversation persistence")

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    agent: str
    lead_saved: bool
    lead_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    pinecone_configured: bool
    mongodb_configured: bool
    details: Dict[str, Any]


# ==========================================
# 3. CORE INFERENCE LOGIC
# ==========================================

def run_agent_turn(message: str, session_id: str) -> Dict[str, Any]:
    """
    Executes a single turn through the LangGraph multi-agent system.
    Maintains conversational memory using thread_id checkpointer.
    """
    config = {"configurable": {"thread_id": session_id}}
    
    # Input state update: append the human message
    inputs = {
        "messages": [HumanMessage(content=message)]
    }

    try:
        final_state = compiled_graph.invoke(inputs, config=config)
        messages = final_state.get("messages", [])

        # Find the last AIMessage content
        reply_content = "I'm sorry, I could not process your request at this moment."
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                reply_content = m.content
                break

        agent = final_state.get("current_agent", "qna_agent")
        lead_saved = final_state.get("lead_saved", False)
        lead_info = final_state.get("lead_info", {})

        return {
            "reply": reply_content,
            "session_id": session_id,
            "agent": agent,
            "lead_saved": lead_saved,
            "lead_info": lead_info if lead_saved else None
        }

    except Exception as e:
        logger.error(f"Inference error during turn for session '{session_id}': {e}", exc_info=True)
        raise e


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Primary chat endpoint for web frontend and microservices."""
    session_id = request.session_id or str(uuid.uuid4())
    try:
        result = run_agent_turn(message=request.message, session_id=session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent inference failed: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check validating external API connections."""
    gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    pinecone_key = bool(os.getenv("PINECONE_API_KEY"))
    mongo_uri = bool(os.getenv("MONGODB_URI"))

    mongo_status = "unconfigured"
    if mongo_uri:
        try:
            col = get_mongo_collection()
            mongo_status = "connected" if col is not None else "connection_failed"
        except Exception:
            mongo_status = "error"

    return HealthResponse(
        status="healthy",
        gemini_configured=gemini_key,
        pinecone_configured=pinecone_key,
        mongodb_configured=mongo_uri,
        details={
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
            "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "novetech-knowledge"),
            "mongodb_database": os.getenv("MONGODB_DATABASE", "novatech_db"),
            "mongodb_status": mongo_status
        }
    )


@app.get("/api/leads")
async def list_recent_leads(limit: int = 20):
    """Admin utility endpoint: returns recent leads saved in MongoDB Atlas."""
    col = get_mongo_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="MongoDB Atlas connection not available.")

    try:
        leads_cursor = col.find().sort("created_at", -1).limit(limit)
        leads = []
        for doc in leads_cursor:
            doc["_id"] = str(doc["_id"])
            leads.append(doc)
        return {"total": len(leads), "leads": leads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch leads: {str(e)}")


# ==========================================
# 5. STATIC FILES / FRONTEND MOUNTING
# ==========================================

# Look for frontend or Chatbot Widget folder
frontend_candidates = [
    os.path.join(os.path.dirname(__file__), "..", "frontend"),
    os.path.join(os.path.dirname(__file__), "frontend"),
    os.path.join(os.path.dirname(__file__), "..", "Chatbot Widget"),
    os.path.join(os.path.dirname(__file__), "Chatbot Widget")
]

for candidate in frontend_candidates:
    if os.path.isdir(candidate):
        app.mount("/static", StaticFiles(directory=candidate), name="static")

        @app.get("/")
        async def serve_index():
            index_path = os.path.join(candidate, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "NovaTech AI Support Agent API is running. See /docs for Swagger UI."}
        break


# ==========================================
# 6. INTERACTIVE CLI CHAT RUNNER
# ==========================================

def run_cli_chat():
    """Terminal chat loop for quick interactive testing."""
    session_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    print("\n" + "=" * 65)
    print("🤖 NOVATECH SOLUTIONS AI SUPPORT & LEAD QUALIFICATION AGENT")
    print("=" * 65)
    print(f"Session ID: {session_id}")
    print("Type your message below. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! Thank you for contacting NovaTech Solutions.")
                break

            print("\n[Thinking...]")
            res = run_agent_turn(message=user_input, session_id=session_id)
            
            agent_badge = f"[{res['agent'].upper()}]"
            print(f"\nAI {agent_badge} > {res['reply']}")

            if res.get("lead_saved"):
                print("\n🎉 [LEAD RECORDED IN MONGODB ATLAS]")
                print(f"Details: {res.get('lead_info')}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        import uvicorn
        uvicorn.run("chatbot_inference:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Default: run interactive CLI
        run_cli_chat()

