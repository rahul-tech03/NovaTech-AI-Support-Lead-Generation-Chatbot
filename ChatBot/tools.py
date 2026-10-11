"""
Tools for NovaTech AI Support Agent:
1. search_ranker: Queries the Pinecone vector index using Gemini embeddings to retrieve ranked knowledge base context.
2. lead_saver: Saves qualified sales lead records into MongoDB Atlas.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv

from logger_utils import setup_logger

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))
logger = setup_logger("Tools")

# Global clients (lazy loaded)
_pinecone_client = None
_pinecone_index = None
_embedding_model = None
_mongo_client = None
_mongo_collection = None

# Fallback in-memory knowledge chunks from NovaTech Knowledge Base
# Ensures the QnA agent remains functional even before indexing or during Pinecone connectivity issues
FALLBACK_KNOWLEDGE_BASE = [
    {
        "id": "kb_overview",
        "title": "Company Overview",
        "content": (
            "NovaTech Solutions is a technology consulting company that helps small and medium-sized "
            "businesses modernize their operations using software development, cloud computing, "
            "data analytics, and artificial intelligence.\n"
            "Website: www.novatech.example\n"
            "Email: contact@novatech.example\n"
            "Business hours: Monday-Friday, 9:00 AM-6:00 PM IST\n"
            "Headquarters: Pune, Maharashtra, India."
        )
    },
    {
        "id": "kb_services",
        "title": "Products and Services",
        "content": (
            "1. AI Chatbot Development: NovaTech builds AI-powered chatbots for customer support, "
            "frequently asked questions, lead qualification, and internal knowledge search. Capabilities include "
            "natural-language conversations, Retrieval-Augmented Generation (RAG), business knowledge-base "
            "integration, lead information collection, and human handoff for unresolved issues.\n"
            "2. Data Analytics: Cleaning, analysis, dashboards, KPI tracking, and reporting automation.\n"
            "3. Cloud and Data Engineering: Data pipeline development, ETL/ELT workflows, cloud storage, "
            "data integration, and workflow automation.\n"
            "4. Custom Software Development: Web applications, REST APIs, and internal business tools."
        )
    },
    {
        "id": "kb_pricing",
        "title": "Indicative Service Plans & Pricing",
        "content": (
            "Indicative service plans for demonstration:\n"
            "• Starter AI Chatbot: Indicative price ₹25,000 one-time. Includes basic FAQ chatbot, "
            "one knowledge source, and simple website integration.\n"
            "• Business AI Assistant: Indicative price ₹60,000 one-time. Includes RAG-based knowledge "
            "assistant, lead capture, and basic conversation analytics.\n"
            "• Enterprise AI Solution: Starting at ₹1,50,000. Includes customized workflows, multiple "
            "knowledge sources, access-control planning, and advanced integrations.\n"
            "Note: These are indicative demonstration prices; final quotations require human confirmation. "
            "The chatbot must not promise discounts or custom quotations without human representative approval."
        )
    },
    {
        "id": "kb_industries",
        "title": "Industries Served",
        "content": (
            "NovaTech's demonstration services are designed for:\n"
            "• Retail and e-commerce\n"
            "• Healthcare administration\n"
            "• Education and training\n"
            "• Financial services operations\n"
            "• Professional services\n"
            "• Small and medium-sized businesses (SMBs)."
        )
    },
    {
        "id": "kb_faqs",
        "title": "Frequently Asked Questions",
        "content": (
            "Q: Can NovaTech build a chatbot using company documents?\n"
            "A: Yes, using Retrieval-Augmented Generation (RAG), the chatbot retrieves relevant "
            "information from approved company documents and formulates answers.\n"
            "Q: Can the chatbot collect customer leads?\n"
            "A: Yes. With the visitor's consent, it collects details such as name, business email, "
            "company name, and service interest for follow-up.\n"
            "Q: How long does implementation take?\n"
            "A: Timelines depend on scope, data readiness, integrations, and approval process. "
            "Direct visitors to the sales team for a project-specific estimate.\n"
            "Q: Does NovaTech provide customer support?\n"
            "A: Support arrangements depend on service agreement. Contact sales or support for project details."
        )
    },
    {
        "id": "kb_lead_process",
        "title": "Lead Qualification & Contact Process",
        "content": (
            "Visitors interested in a service may be asked for:\n"
            "1. Full name\n"
            "2. Business email\n"
            "3. Contact / phone number\n"
            "4. Service of interest (e.g. Starter AI Chatbot, Business AI Assistant, Enterprise AI Solution, Custom Software)\n"
            "5. Company name and brief project requirements.\n"
            "The chatbot must explain why it collects these details and obtain consent before saving them. "
            "Never request passwords, payment card details, or sensitive personal data."
        )
    }
]


# ==========================================
# 1. EMBEDDINGS & PINECONE INITIALIZATION
# ==========================================

def get_embedding_model():
    """Lazy initialize and return Google Gemini Embeddings instance."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment.")
        return None

    model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()

    candidate_models = [
        model_name,
        f"models/{model_name}" if not model_name.startswith("models/") else model_name,
        "models/embedding-001",
        "embedding-001",
        "models/text-embedding-004",
        "text-embedding-004"
    ]
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    # Try LangChain Google GenAI embeddings with candidate models
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        class ResilientLangChainEmbedder:
            def __init__(self, key: str, models: list, target_dim: int = 768):
                self.key = key
                self.models = models
                self.target_dim = target_dim
                self.active_embedder = None
                self._init_first_working()

            def _init_first_working(self):
                for m in self.models:
                    try:
                        emb = GoogleGenerativeAIEmbeddings(
                            model=m,
                            google_api_key=self.key,
                            output_dimensionality=self.target_dim
                        )
                        res = emb.embed_query("ping")
                        self.active_embedder = emb
                        logger.info(f"Initialized active embedding model: '{m}' (dim={len(res)})")
                        break
                    except Exception:
                        try:
                            emb = GoogleGenerativeAIEmbeddings(model=m, google_api_key=self.key)
                            res = emb.embed_query("ping")
                            self.active_embedder = emb
                            logger.info(f"Initialized fallback embedding model: '{m}' (dim={len(res)})")
                            break
                        except Exception:
                            continue

            def embed_query(self, text: str) -> List[float]:
                if self.active_embedder:
                    try:
                        vec = self.active_embedder.embed_query(text)
                        if len(vec) != self.target_dim:
                            vec = vec[:self.target_dim]
                        return vec
                    except Exception:
                        pass
                for m in self.models:
                    try:
                        emb = GoogleGenerativeAIEmbeddings(
                            model=m,
                            google_api_key=self.key,
                            output_dimensionality=self.target_dim
                        )
                        vec = emb.embed_query(text)
                        if len(vec) != self.target_dim:
                            vec = vec[:self.target_dim]
                        self.active_embedder = emb
                        return vec
                    except Exception:
                        continue
                raise RuntimeError("Failed to generate embedding with all candidate models")

        target_dim = 3072
        try:
            from pinecone import Pinecone
            p_key = os.getenv("PINECONE_API_KEY", "").strip()
            p_idx = os.getenv("PINECONE_INDEX_NAME", "novetech-knowledge").strip()
            if p_key:
                _pc = Pinecone(api_key=p_key)
                _desc = _pc.describe_index(p_idx)
                target_dim = getattr(_desc, "dimension", 3072)
                logger.info(f"Target embedding dimension detected from Pinecone: {target_dim}")
        except Exception:
            target_dim = 3072

        _embedding_model = ResilientLangChainEmbedder(api_key, models_to_try, target_dim=target_dim)
        return _embedding_model
    except ImportError:
        logger.warning("langchain_google_genai not installed. Falling back to google.generativeai.")

    # Fallback to direct google.generativeai
    try:
        # pyrefly: ignore [missing-import]
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        class DirectGeminiEmbedder:
            def __init__(self, models: list, target_dim: int = 3072):
                self.models = models
                self.target_dim = target_dim

            def embed_query(self, text: str) -> List[float]:
                for m in self.models:
                    clean_model = m if m.startswith("models/") else f"models/{m}"
                    try:
                        result = genai.embed_content(
                            model=clean_model,
                            content=text,
                            task_type="retrieval_query",
                            output_dimensionality=self.target_dim
                        )
                        vec = result["embedding"]
                        if len(vec) != self.target_dim:
                            vec = vec[:self.target_dim]
                        return vec
                    except Exception:
                        try:
                            result = genai.embed_content(
                                model=clean_model,
                                content=text,
                                task_type="retrieval_query"
                            )
                            vec = result["embedding"]
                            if len(vec) != self.target_dim:
                                vec = vec[:self.target_dim]
                            return vec
                        except Exception:
                            continue
                raise RuntimeError("Failed to generate query embedding with google.generativeai")

        _embedding_model = DirectGeminiEmbedder(models_to_try, target_dim=target_dim)
        return _embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return None


def get_pinecone_index():
    """Lazy initialize and return Pinecone index connection."""
    global _pinecone_client, _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "novetech-knowledge").strip()

    if not api_key:
        logger.warning("PINECONE_API_KEY is not set.")
        return None

    try:
        from pinecone import Pinecone
        _pinecone_client = Pinecone(api_key=api_key)
        _pinecone_index = _pinecone_client.Index(index_name)
        logger.info(f"Connected to Pinecone index: '{index_name}'")
        return _pinecone_index
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
        return None


# ==========================================
# 2. TOOL 1: SEARCH_RANKER
# ==========================================

def search_ranker(query: str, top_k: int = 3) -> str:
    """
    Search and retrieve ranked knowledge context from Pinecone vector store using Gemini embeddings.
    
    Args:
        query: The user's query or question to search the knowledge base for.
        top_k: Number of relevant context chunks to retrieve (default: 3).

    Returns:
        A formatted string of relevant knowledge base excerpts, or fallback knowledge if vector search is unavailable.
    """
    logger.info(f"Executing search_ranker tool for query: '{query}'")
    
    # 1. Attempt Pinecone Vector Search
    index = get_pinecone_index()
    embedder = get_embedding_model()

    if index and embedder:
        try:
            # Generate query embedding
            if hasattr(embedder, "embed_query"):
                query_vector = embedder.embed_query(query)
            else:
                query_vector = embedder(query)

            # Query Pinecone
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            matches = response.get("matches", [])
            if matches:
                formatted_results = []
                for i, match in enumerate(matches, 1):
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    title = meta.get("title", meta.get("section", f"Result {i}"))
                    text = meta.get("text", meta.get("content", ""))
                    if text:
                        formatted_results.append(
                            f"--- [Knowledge Match #{i}] (Relevance Score: {score:.3f}) ---\n"
                            f"Section: {title}\n"
                            f"{text.strip()}"
                        )
                if formatted_results:
                    logger.info(f"Retrieved {len(formatted_results)} matches from Pinecone.")
                    return "\n\n".join(formatted_results)
                else:
                    logger.info("Pinecone returned matches, but metadata text was empty.")
            else:
                logger.info("Pinecone query returned 0 matches; using local fallback.")
        except Exception as e:
            logger.warning(f"Error querying Pinecone index: {e}. Falling back to internal knowledge base.")
    else:
        logger.info("Pinecone/Embedder not configured or unavailable. Using internal knowledge base.")

    # 2. Local Fallback Matching (Keyword & Semantic relevance from NovaTech KB)
    query_lower = query.lower()
    scored_chunks = []
    for chunk in FALLBACK_KNOWLEDGE_BASE:
        text = chunk["content"].lower()
        title = chunk["title"].lower()
        score = 0
        words = query_lower.split()
        for word in words:
            if len(word) > 2:
                if word in title:
                    score += 3
                if word in text:
                    score += 1
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    selected = [c for s, c in scored_chunks[:top_k] if s > 0]
    
    # If no specific keyword matched, provide the primary overview & services
    if not selected:
        selected = FALLBACK_KNOWLEDGE_BASE[:2]

    results = []
    for i, chunk in enumerate(selected, 1):
        results.append(
            f"--- [Knowledge Match #{i}] ---\n"
            f"Section: {chunk['title']}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(results)


# ==========================================
# 3. MONGODB ATLAS INITIALIZATION
# ==========================================

def get_mongo_collection():
    """Lazy initialize and return MongoDB Atlas leads collection."""
    global _mongo_client, _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection

    mongo_uri = os.getenv("MONGODB_URI", "").strip()
    db_name = os.getenv("MONGODB_DATABASE", "novatech_db").strip()
    col_name = os.getenv("MONGODB_COLLECTION", "leads").strip()

    if not mongo_uri:
        logger.error("MONGODB_URI is not set in environment.")
        return None

    try:
        from pymongo import MongoClient
        import certifi

        try:
            # Try connecting with certifi ca bundle for robust TLS/SSL verification
            _mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        except Exception:
            # Fallback if certifi is not installed
            _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

        db = _mongo_client[db_name]
        _mongo_collection = db[col_name]
        logger.info(f"Connected to MongoDB Atlas: Database='{db_name}', Collection='{col_name}'")
        return _mongo_collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        return None


# ==========================================
# 4. TOOL 2: LEAD_SAVER
# ==========================================

def lead_saver(
    name: str,
    email: str,
    contact: str,
    service_of_interest: str,
    company_name: str = "",
    requirements: str = ""
) -> Dict[str, Any]:
    """
    Saves a qualified lead's contact information and project requirements into MongoDB Atlas.
    
    Args:
        name: Full name of the prospect.
        email: Business email address of the prospect.
        contact: Phone / mobile contact number.
        service_of_interest: The specific service or product they are interested in (e.g. 'Starter AI Chatbot', 'Business AI Assistant', 'Enterprise AI Solution', 'Data Analytics', 'Custom Software').
        company_name: Name of the prospect's company (optional).
        requirements: Summary of project needs, timeline, or special requirements (optional).

    Returns:
        Dict containing success status, confirmation lead_id, and response message.
    """
    logger.info(f"Executing lead_saver tool for lead: name='{name}', email='{email}', service='{service_of_interest}'")

    # Validate essential fields
    if not name or not name.strip():
        return {
            "status": "error",
            "message": "Lead name is required to save the contact information."
        }
    
    if not email and not contact:
        return {
            "status": "error",
            "message": "At least one contact method (email or phone) is required to save the lead."
        }

    lead_doc = {
        "name": name.strip(),
        "email": email.strip() if email else "",
        "contact": contact.strip() if contact else "",
        "service_of_interest": service_of_interest.strip() if service_of_interest else "General Inquiry",
        "company_name": company_name.strip() if company_name else "Not provided",
        "requirements": requirements.strip() if requirements else "Not provided",
        "status": "NEW_QUALIFIED_LEAD",
        "source": "NovaTech AI Support Agent",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    collection = get_mongo_collection()
    if collection is None:
        error_msg = "MongoDB Atlas collection is currently unavailable. Please check MONGODB_URI and network access."
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "lead_data": lead_doc
        }

    try:
        result = collection.insert_one(lead_doc)
        lead_id = str(result.inserted_id)
        logger.info(f"Successfully saved lead into MongoDB Atlas with ID: {lead_id}")
        return {
            "status": "success",
            "lead_id": lead_id,
            "message": (
                f"Lead successfully recorded in NovaTech CRM (MongoDB Atlas) with Lead ID: {lead_id}. "
                f"Our solutions team will reach out to {name.strip()} at {email.strip() or contact.strip()} shortly."
            ),
            "lead_data": {
                "id": lead_id,
                "name": lead_doc["name"],
                "email": lead_doc["email"],
                "contact": lead_doc["contact"],
                "service": lead_doc["service_of_interest"]
            }
        }
    except Exception as e:
        logger.error(f"Error saving lead to MongoDB Atlas: {e}")
        return {
            "status": "error",
            "message": f"Failed to save lead in database: {str(e)}",
            "lead_data": lead_doc
        }


# ==========================================
# 5. LANGCHAIN TOOL WRAPPERS
# ==========================================

try:
    from langchain_core.tools import tool

    @tool("search_ranker")
    def search_ranker_tool(query: str) -> str:
        """Search the NovaTech Knowledge Base in Pinecone using Gemini embeddings to find official company information, indicative pricing, services, and policies."""
        return search_ranker(query=query)

    @tool("lead_saver")
    def lead_saver_tool(
        name: str,
        email: str,
        contact: str,
        service_of_interest: str,
        company_name: str = "",
        requirements: str = ""
    ) -> str:
        """Save a prospective customer's qualified lead information into MongoDB Atlas once their consent and details have been gathered."""
        res = lead_saver(
            name=name,
            email=email,
            contact=contact,
            service_of_interest=service_of_interest,
            company_name=company_name,
            requirements=requirements
        )
        return json.dumps(res, indent=2)

except ImportError:
    # If langchain_core is not present yet, direct callables remain available
    search_ranker_tool = search_ranker
    lead_saver_tool = lead_saver
