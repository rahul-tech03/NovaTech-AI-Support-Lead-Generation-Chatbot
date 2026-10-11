"""
agents.py: LangGraph Orchestrator & Multi-Agent Architecture
Architecture:
- Orchestrator: Supervises and routes between QnA and Lead agents.
- QnA Agent: Answers user questions using the 'search_ranker' Pinecone tool.
- Lead Agent: Detects customer purchase/service interest, collects lead details conversationally,
              and saves records to MongoDB Atlas using the 'lead_saver' tool.
"""

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from logger_utils import setup_logger
from tools import search_ranker, lead_saver, search_ranker_tool, lead_saver_tool

load_dotenv(find_dotenv(usecwd=True))
logger = setup_logger("Agents")

# Model Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

def get_llm(temperature: float = 0.2):
    """Factory to initialize Gemini model."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature
    )


# ==========================================
# 1. STATE DEFINITION
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_agent: str
    intent: str
    lead_info: Dict[str, Any]
    lead_saved: bool


# ==========================================
# 2. SYSTEM PROMPTS
# ==========================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are the NovaTech Solutions Orchestrator (Supervisor Router).
NovaTech is an AI consulting firm providing AI Chatbot Development (Starter ₹25,000, Business ₹60,000, Enterprise ₹1,50,000+), Data Analytics, Cloud & Data Engineering, and Custom Software.

Your sole duty is to inspect the latest user message and conversation history to classify intent and select the appropriate specialist:

1. 'lead_agent':
   Route here if the user:
   - Expresses interest in buying, purchasing, hiring, getting started, or implementing an AI chatbot / service.
   - Asks for a custom quote, consultation, meeting, or says "I want to build a chatbot for my company", "I am interested", "contact me", etc.
   - Is in the middle of providing or replying with their contact details (name, email, phone number, company name, requirements).

2. 'qna_agent':
   Route here if the user:
   - Asks general informational questions about NovaTech Solutions, what the company does, technologies used (RAG, LangGraph, etc.), office hours, headquarters (Pune), industries served, or indicative service plans.
   - Greets ("hello", "hi", "who are you").

Respond ONLY in valid JSON format:
{
  "next_agent": "qna_agent" | "lead_agent",
  "reasoning": "Brief explanation of routing decision"
}
"""

QNA_SYSTEM_PROMPT = """You are the NovaTech Solutions QnA Support Agent.
Your role is to answer user inquiries accurately and politely, strictly adhering to the official NovaTech Knowledge Base.

RULES:
1. Always consult your `search_ranker` tool to retrieve knowledge base context about NovaTech products, plans, technologies, and company information.
2. Ground your answers in the retrieved context:
   - Indicative Pricing: Starter AI Chatbot is ₹25,000; Business AI Assistant is ₹60,000; Enterprise AI Solution starts at ₹1,50,000.
   - State clearly that these are indicative demonstration prices; final quotations require human confirmation. Do not promise discounts.
   - Mention key capabilities when asked: Natural-language conversations, RAG, business knowledge base integration, lead capture, and human handoff.
   - Headquartered in Pune, Maharashtra, India. Hours: Mon-Fri, 9:00 AM-6:00 PM IST.
3. If the user expresses interest in hiring NovaTech or acquiring a chatbot/service, provide a concise helpful answer and warmly invite them to connect with our solutions team so we can qualify their requirements.
4. Keep answers friendly, professional, clear, and concise.
"""

LEAD_SYSTEM_PROMPT = """You are the NovaTech Solutions Lead Qualification Specialist.
Your objective is to warmly assist prospective customers who are interested in NovaTech's products or services (such as Starter AI Chatbot, Business AI Assistant, Enterprise AI Solution, Data Analytics, or Custom Software) and capture their details for our sales engineering team.

LEAD QUALIFICATION PROTOCOL:
1. When a user expresses interest in services or solutions:
   - Acknowledge their interest enthusiastically.
   - Explain why you are collecting details: "To connect you with our solutions consultant and provide a tailored proposal, could you share a few details?"
2. Check the conversation to see which of the following required details are still missing:
   - Full Name
   - Business Email
   - Contact / Phone Number
   - Service / Product of Interest (e.g. Starter Chatbot, Business AI Assistant, Enterprise AI, etc.)
   - Company Name or Brief Project Requirements (optional but recommended)
3. Ask for any missing information politely. You may ask in a single friendly prompt.
4. As soon as you have the user's essential contact details (Name, Email/Contact, and Service of Interest), immediately call the `lead_saver` tool to save the lead to MongoDB Atlas!
5. After `lead_saver` is executed:
   - Confirm to the customer that their request has been successfully recorded.
   - Mention the assigned Lead ID and reassure them that our solutions team will reach out shortly.
6. Privacy & Safety: Never ask for passwords, payment card numbers, or sensitive financial information.
"""


# ==========================================
# 3. GRAPH NODES
# ==========================================

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates conversation and routes to the appropriate specialist agent."""
    messages = state.get("messages", [])
    if not messages:
        return {"current_agent": "qna_agent", "intent": "qna"}

    # If lead was already saved in previous step, keep polite closing in qna
    if state.get("lead_saved", False):
        return {"current_agent": "qna_agent", "intent": "post_lead"}

    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    logger.info(f"Orchestrator evaluating user message: '{last_user_msg[:60]}...'")

    llm = get_llm(temperature=0.0)
    prompt = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Latest user message: '{last_user_msg}'\nConversation context length: {len(messages)} messages.")
    ]

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Clean markdown fences if any
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        next_agent = data.get("next_agent", "qna_agent")
        reasoning = data.get("reasoning", "")
        logger.info(f"Orchestrator decision: next_agent='{next_agent}', reason='{reasoning}'")
    except Exception as e:
        logger.warning(f"Orchestrator JSON parsing fallback ({e}). Using heuristic routing.")
        text = last_user_msg.lower()
        lead_triggers = ["interested", "hire", "buy", "purchase", "quote", "pricing for my", "contact me", "my name is", "my email", "phone", "call me", "starter plan", "business plan", "enterprise plan"]
        if any(trig in text for trig in lead_triggers):
            next_agent = "lead_agent"
        else:
            next_agent = "qna_agent"

    return {"current_agent": next_agent, "intent": next_agent}


def qna_agent_node(state: AgentState) -> Dict[str, Any]:
    """QnA Agent node: queries knowledge base and formulates answers."""
    messages = state.get("messages", [])
    llm = get_llm(temperature=0.2).bind_tools([search_ranker_tool])

    system_message = SystemMessage(content=QNA_SYSTEM_PROMPT)
    conversation = [system_message] + list(messages)

    response = llm.invoke(conversation)
    return {"messages": [response], "current_agent": "qna_agent"}


def lead_agent_node(state: AgentState) -> Dict[str, Any]:
    """Lead Agent node: collects user info and invokes lead_saver tool."""
    messages = state.get("messages", [])
    llm = get_llm(temperature=0.1).bind_tools([lead_saver_tool])

    system_message = SystemMessage(content=LEAD_SYSTEM_PROMPT)
    conversation = [system_message] + list(messages)

    response = llm.invoke(conversation)
    return {"messages": [response], "current_agent": "lead_agent"}


def execute_qna_tools(state: AgentState) -> Dict[str, Any]:
    """Executes tools requested by the QnA agent (search_ranker)."""
    messages = state.get("messages", [])
    last_message = messages[-1]

    tool_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            call_id = tool_call.get("id")

            logger.info(f"Executing QnA tool '{tool_name}' with args: {tool_args}")
            if tool_name == "search_ranker":
                query = tool_args.get("query", "")
                result = search_ranker(query=query)
            else:
                result = f"Unknown tool: {tool_name}"

            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id,
                    name=tool_name
                )
            )

    return {"messages": tool_messages}


def execute_lead_tools(state: AgentState) -> Dict[str, Any]:
    """Executes tools requested by the Lead agent (lead_saver)."""
    messages = state.get("messages", [])
    last_message = messages[-1]

    tool_messages = []
    lead_saved_flag = False
    lead_info = dict(state.get("lead_info", {}))

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            call_id = tool_call.get("id")

            logger.info(f"Executing Lead tool '{tool_name}' with args: {tool_args}")
            if tool_name == "lead_saver":
                result_dict = lead_saver(**tool_args)
                result_str = json.dumps(result_dict)
                if result_dict.get("status") == "success":
                    lead_saved_flag = True
                    lead_info.update(tool_args)
                    lead_info["lead_id"] = result_dict.get("lead_id")
            else:
                result_str = f"Unknown tool: {tool_name}"

            tool_messages.append(
                ToolMessage(
                    content=result_str,
                    tool_call_id=call_id,
                    name=tool_name
                )
            )

    return {
        "messages": tool_messages,
        "lead_saved": lead_saved_flag or state.get("lead_saved", False),
        "lead_info": lead_info
    }


# ==========================================
# 4. CONDITIONAL ROUTING EDGES
# ==========================================

def route_orchestrator(state: AgentState) -> str:
    """Selects which agent node to invoke based on state['current_agent']."""
    agent = state.get("current_agent", "qna_agent")
    if agent == "lead_agent":
        return "lead_agent"
    return "qna_agent"


def route_qna_tools(state: AgentState) -> str:
    """Checks if QnA Agent requested tool execution or finished."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "qna_tools"
    return END


def route_lead_tools(state: AgentState) -> str:
    """Checks if Lead Agent requested tool execution or finished."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "lead_tools"
    return END


# ==========================================
# 5. GRAPH CONSTRUCTION & COMPILATION
# ==========================================

def create_support_graph():
    """Builds and compiles the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("qna_agent", qna_agent_node)
    workflow.add_node("qna_tools", execute_qna_tools)
    workflow.add_node("lead_agent", lead_agent_node)
    workflow.add_node("lead_tools", execute_lead_tools)

    # Edge from START to orchestrator
    workflow.add_edge(START, "orchestrator")

    # Conditional edge from orchestrator to selected agent
    workflow.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "qna_agent": "qna_agent",
            "lead_agent": "lead_agent"
        }
    )

    # QnA Agent loop: agent -> tools -> agent -> END
    workflow.add_conditional_edges(
        "qna_agent",
        route_qna_tools,
        {
            "qna_tools": "qna_tools",
            END: END
        }
    )
    workflow.add_edge("qna_tools", "qna_agent")

    # Lead Agent loop: agent -> tools -> agent -> END
    workflow.add_conditional_edges(
        "lead_agent",
        route_lead_tools,
        {
            "lead_tools": "lead_tools",
            END: END
        }
    )
    workflow.add_edge("lead_tools", "lead_agent")

    # In-memory checkpointer for session state persistence across turns
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph Support Agent successfully compiled with checkpointing.")
    return app


# Singleton compiled graph instance
compiled_graph = create_support_graph()
