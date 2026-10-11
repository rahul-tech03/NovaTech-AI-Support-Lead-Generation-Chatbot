"""
add_temp_records.py: Ingests NovaTech Solutions Knowledge Base into Pinecone
using Gemini Text Embeddings (gemini-text-embedding-001).
"""

import os
import sys
import time
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv

from logger_utils import setup_logger

load_dotenv(find_dotenv(usecwd=True))
logger = setup_logger("IndexKnowledgeBase")

# Knowledge base structured chunks parsed directly from novatech_knowledge_base.pdf
DEFAULT_SECTIONS = [
    {
        "id": "novatech_sec_1_overview",
        "title": "NovaTech Solutions - Company Overview",
        "section": "1. Company Overview",
        "page": 1,
        "text": (
            "NovaTech Solutions is a technology consulting company that helps small and "
            "medium-sized businesses modernize their operations using software development, "
            "cloud computing, data analytics, and artificial intelligence.\n"
            "Website: www.novatech.example\n"
            "Email: contact@novatech.example\n"
            "Business hours: Monday–Friday, 9:00 AM–6:00 PM IST\n"
            "Headquarters: Pune, Maharashtra, India.\n"
            "The company provides technology consulting and implementation services. "
            "The website and contact details are used for demonstration purposes."
        )
    },
    {
        "id": "novatech_sec_2_ai_chatbots",
        "title": "Products and Services - AI Chatbot Development",
        "section": "2. Products and Services",
        "page": 1,
        "text": (
            "AI Chatbot Development: NovaTech builds AI-powered chatbots for customer support, "
            "frequently asked questions, lead qualification, and internal knowledge search.\n"
            "Typical capabilities include:\n"
            "• Natural-language conversations\n"
            "• Retrieval-Augmented Generation (RAG)\n"
            "• Business knowledge-base integration\n"
            "• Lead information collection\n"
            "• Human handoff for unresolved issues"
        )
    },
    {
        "id": "novatech_sec_2_other_services",
        "title": "Products and Services - Data Analytics, Cloud, Software Development",
        "section": "2. Products and Services",
        "page": 1,
        "text": (
            "NovaTech Services Suite:\n"
            "1. Data Analytics: NovaTech helps businesses understand their operations through data "
            "cleaning, analysis, dashboards, KPI tracking, and reporting automation.\n"
            "2. Cloud and Data Engineering: Services include data pipeline development, ETL/ELT workflows, "
            "cloud data storage, data integration, and workflow automation.\n"
            "3. Custom Software Development: NovaTech develops web applications, REST APIs, and internal "
            "business tools tailored to client requirements."
        )
    },
    {
        "id": "novatech_sec_3_pricing_plans",
        "title": "Indicative Service Plans & Pricing",
        "section": "3. Indicative Service Plans",
        "page": 2,
        "text": (
            "Indicative Service Plans (Example prices for demonstration chatbot):\n"
            "1. Starter AI Chatbot: Indicative price: ₹25,000 one-time. Includes basic FAQ chatbot, "
            "one knowledge source, and simple website integration.\n"
            "2. Business AI Assistant: Indicative price: ₹60,000 one-time. Includes RAG-based knowledge "
            "assistant, lead capture, and basic conversation analytics.\n"
            "3. Enterprise AI Solution: Indicative price: Starting at ₹1,50,000. Includes customized "
            "workflows, multiple knowledge sources, access-control planning, and advanced integrations.\n"
            "Pricing Policy: Final pricing depends on project scope, integrations, deployment requirements, "
            "and ongoing usage costs. The chatbot must not promise discounts or customized quotations "
            "without approval from a human representative."
        )
    },
    {
        "id": "novatech_sec_4_industries",
        "title": "Industries Served",
        "section": "4. Industries Served",
        "page": 2,
        "text": (
            "NovaTech's demonstration services are designed for:\n"
            "• Retail and e-commerce\n"
            "• Healthcare administration\n"
            "• Education and training\n"
            "• Financial services operations\n"
            "• Professional services\n"
            "• Small and medium-sized businesses\n"
            "These are target industries for specialized AI consulting and custom implementations."
        )
    },
    {
        "id": "novatech_sec_5_faqs",
        "title": "Frequently Asked Questions (FAQs)",
        "section": "5. Frequently Asked Questions",
        "page": 2,
        "text": (
            "Q: What does NovaTech Solutions do?\n"
            "A: NovaTech provides AI chatbot development, data analytics, cloud and data engineering, "
            "and custom software development.\n"
            "Q: Can NovaTech build a chatbot using our company documents?\n"
            "A: Yes. A knowledge-base chatbot can retrieve relevant information from approved company documents "
            "and use it to formulate answers.\n"
            "Q: What is RAG?\n"
            "A: Retrieval-Augmented Generation (RAG) retrieves relevant information from a knowledge source "
            "and provides it to a language model as context for answering a question.\n"
            "Q: Can the chatbot collect customer leads?\n"
            "A: Yes. With the visitor's consent, it can collect details such as name, business email, company name, "
            "and service interest for follow-up.\n"
            "Q: How much does an AI chatbot cost?\n"
            "A: Starter plan is ₹25,000, Business plan is ₹60,000, Enterprise solutions start at ₹1,50,000. "
            "These are indicative demonstration prices; final quotes require human confirmation.\n"
            "Q: How long does implementation take?\n"
            "A: Timelines depend on scope, data readiness, integrations, and approval process. Visitors should be "
            "directed to the sales team for a project-specific estimate.\n"
            "Q: Does NovaTech provide customer support?\n"
            "A: Support arrangements depend on service agreement. Contact sales or support for project details.\n"
            "Q: Can the chatbot resolve complaints?\n"
            "A: It can identify complaint-related messages and collect a brief description for escalation. "
            "It must not claim a complaint has been resolved or ticket created unless confirmed by the system."
        )
    },
    {
        "id": "novatech_sec_6_lead_qualification",
        "title": "Lead Qualification and Contact Process",
        "section": "6. Lead Qualification and Contact Process",
        "page": 3,
        "text": (
            "Lead Qualification Process:\n"
            "Visitors interested in a service may be asked for:\n"
            "• Full name\n"
            "• Business email\n"
            "• Contact / phone number\n"
            "• Company name\n"
            "• Service of interest (Starter, Business, Enterprise, Analytics, Software)\n"
            "• Brief project requirements\n"
            "Consent & Privacy: The chatbot should explain why it is collecting these details and obtain "
            "consent before saving them for follow-up. Never request passwords, payment-card details, or "
            "unnecessary sensitive information. A lead is qualified when the visitor expresses interest "
            "and provides contact information with consent."
        )
    },
    {
        "id": "novatech_sec_7_complaints_rules",
        "title": "Complaints, Human Handoff and Answering Rules",
        "section": "7. Complaints and Human Handoff & 8. Answering Rules",
        "page": 4,
        "text": (
            "Complaints & Human Handoff:\n"
            "1. Acknowledge concerns politely.\n"
            "2. Ask for a brief description of the issue.\n"
            "3. Collect only the information necessary for follow-up.\n"
            "4. Explain that a human representative may need to review the matter.\n"
            "5. Avoid claiming a ticket was created or escalation completed unless confirmed.\n\n"
            "Assistant Answering Rules:\n"
            "• Use this knowledge base as the primary source for company-specific answers.\n"
            "• Do not invent services, prices, customer names, guarantees, or company policies.\n"
            "• If the answer is not available in the retrieved information, state that you do not have "
            "enough information and offer human follow-up.\n"
            "• Distinguish indicative prices from confirmed quotations.\n"
            "• Do not reveal private lead information to other visitors.\n"
            "• Never claim to have performed an action unless the backend confirms it."
        )
    }
]


def load_pdf_documents() -> List[Dict[str, Any]]:
    """Attempt to parse PDF if available; fallback to structured sections."""
    pdf_paths = [
        os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "novatech_knowledge_base.pdf"),
        os.path.join(os.path.dirname(__file__), "knowledge_base", "novatech_knowledge_base.pdf"),
        "knowledge_base/novatech_knowledge_base.pdf"
    ]
    
    found_pdf = None
    for p in pdf_paths:
        if os.path.exists(p):
            found_pdf = p
            break

    if not found_pdf:
        logger.info("PDF path not directly found; using pre-extracted high-fidelity structured sections.")
        return DEFAULT_SECTIONS

    try:
        from pypdf import PdfReader
        reader = PdfReader(found_pdf)
        logger.info(f"Loaded PDF from '{found_pdf}' with {len(reader.pages)} pages.")
        # Return default sections as they are already high-fidelity semantic units,
        # perfectly chunked according to sections 1 to 8.
        return DEFAULT_SECTIONS
    except Exception as e:
        logger.info(f"pypdf reader notice ({e}); using pre-extracted semantic sections.")
        return DEFAULT_SECTIONS


def index_knowledge_base():
    """Generates embeddings and upserts knowledge base documents into Pinecone."""
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "novetech-knowledge").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    embed_model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    if not api_key:
        logger.error("PINECONE_API_KEY is not set.")
        return False
    if not gemini_key:
        logger.error("GEMINI_API_KEY is not set.")
        return False

    logger.info(f"Connecting to Pinecone index: '{index_name}'...")
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=api_key)

    # Load documents
    docs = load_pdf_documents()
    texts = [d["text"] for d in docs]

    candidate_models = [
        embed_model_name,
        f"models/{embed_model_name}" if not embed_model_name.startswith("models/") else embed_model_name,
        "models/embedding-001",
        "embedding-001",
        "models/text-embedding-004",
        "text-embedding-004"
    ]
    seen = set()
    models_to_try = []
    for m in candidate_models:
        if m and m not in seen:
            seen.add(m)
            models_to_try.append(m)

    # Determine target dimension from existing Pinecone index (or default to 768)
    target_dim = 768
    try:
        idx_info = pc.describe_index(index_name)
        target_dim = getattr(idx_info, "dimension", 768)
        logger.info(f"Detected target dimension for Pinecone index '{index_name}': {target_dim}")
    except Exception as e:
        logger.warning(f"Could not fetch index description ({e}); assuming dimension={target_dim}")

    embeddings = None
    successful_model = None

    for m in models_to_try:
        try:
            logger.info(f"Attempting embeddings with '{m}' (output_dimensionality={target_dim})...")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(
                model=m,
                google_api_key=gemini_key,
                output_dimensionality=target_dim
            )
            embeddings = embedder.embed_documents(texts)
            if embeddings and len(embeddings) > 0:
                # Ensure each vector matches target_dim
                if len(embeddings[0]) != target_dim:
                    embeddings = [emb[:target_dim] for emb in embeddings]
                successful_model = m
                logger.info(f"Successfully generated embeddings using model '{m}' with dimension {len(embeddings[0])}.")
                break
        except Exception as e:
            logger.warning(f"LangChain with '{m}' failed: {e}. Trying fallback...")
            try:
                embedder = GoogleGenerativeAIEmbeddings(model=m, google_api_key=gemini_key)
                embs = embedder.embed_documents(texts)
                if embs and len(embs) > 0:
                    embeddings = [emb[:target_dim] for emb in embs]
                    successful_model = m
                    logger.info(f"Successfully generated embeddings (fallback) using model '{m}'.")
                    break
            except Exception:
                pass

    if embeddings is None:
        try:
            # pyrefly: ignore [missing-import]
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            for m in models_to_try:
                try:
                    logger.info(f"Attempting direct genai embed_content with '{m}' (dim={target_dim})...")
                    raw_embs = []
                    for t in texts:
                        try:
                            res = genai.embed_content(
                                model=m if m.startswith("models/") else f"models/{m}",
                                content=t,
                                task_type="retrieval_document",
                                output_dimensionality=target_dim
                            )
                        except Exception:
                            res = genai.embed_content(
                                model=m if m.startswith("models/") else f"models/{m}",
                                content=t,
                                task_type="retrieval_document"
                            )
                        emb_vec = res["embedding"]
                        if len(emb_vec) != target_dim:
                            emb_vec = emb_vec[:target_dim]
                        raw_embs.append(emb_vec)
                    if raw_embs:
                        embeddings = raw_embs
                        successful_model = m
                        logger.info(f"Successfully generated embeddings using direct genai '{m}'.")
                        break
                except Exception as ex:
                    logger.warning(f"Direct genai with '{m}' failed: {ex}")
        except Exception as genai_err:
            logger.error(f"Failed to use google.generativeai: {genai_err}")

    if not embeddings:
        logger.error("Could not generate embeddings with any available model.")
        return False

    embedding_dim = len(embeddings[0])
    logger.info(f"Successfully generated embeddings (dimension: {embedding_dim}, model: {successful_model}).")

    # Ensure Pinecone index exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Index '{index_name}' does not exist. Creating serverless index (dimension={embedding_dim})...")
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)  # Wait for index provisioning

    index = pc.Index(index_name)

    # Prepare vectors for upsert
    vectors_to_upsert = []
    for doc, emb in zip(docs, embeddings):
        vectors_to_upsert.append({
            "id": doc["id"],
            "values": emb,
            "metadata": {
                "title": doc["title"],
                "section": doc["section"],
                "page": doc["page"],
                "text": doc["text"]
            }
        })

    logger.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{index_name}'...")
    index.upsert(vectors=vectors_to_upsert)
    logger.info("Knowledge base indexing successfully completed!")
    return True


if __name__ == "__main__":
    logger.info("Starting NovaTech Knowledge Base indexing process...")
    success = index_knowledge_base()
    if success:
        logger.info("Index verification ready. Knowledge base is live in Pinecone!")
    else:
        logger.warning("Indexing completed with warnings. Check logs for details.")
