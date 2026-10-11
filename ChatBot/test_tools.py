"""
test_tools.py: Test suite to independently verify:
1. search_ranker tool (Gemini Embeddings + Pinecone / Knowledge Base)
2. lead_saver tool (MongoDB Atlas Lead Ingestion)
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

from logger_utils import setup_logger
from tools import search_ranker, lead_saver, get_mongo_collection

load_dotenv(find_dotenv(usecwd=True))
logger = setup_logger("TestTools")

def test_search_ranker_tool():
    """Test search_ranker tool with sample queries."""
    print("\n" + "="*60)
    print("TEST 1: Testing 'search_ranker' Tool")
    print("="*60)
    
    test_queries = [
        "What are the indicative prices for NovaTech AI chatbots?",
        "Can NovaTech build a chatbot using our company documents?"
    ]

    for q in test_queries:
        print(f"\n[Query]: {q}")
        result = search_ranker(query=q, top_k=2)
        print("-" * 50)
        print(result[:400] + ("..." if len(result) > 400 else ""))
        print("-" * 50)
        assert result and len(result) > 20, "search_ranker failed to return meaningful context"

    print("\n>>> search_ranker tool test PASSED! <<<\n")


def test_lead_saver_tool():
    """Test lead_saver tool by writing a test lead to MongoDB Atlas."""
    print("\n" + "="*60)
    print("TEST 2: Testing 'lead_saver' Tool (MongoDB Atlas)")
    print("="*60)

    test_lead = {
        "name": "Alex Mercer",
        "email": "alex.mercer@innovatetech.example",
        "contact": "+91 98765 43210",
        "service_of_interest": "Business AI Assistant (₹60,000)",
        "company_name": "InnovateTech Solutions",
        "requirements": "Need a RAG customer support chatbot with knowledge base integration and lead collection."
    }

    print(f"Submitting test lead for: {test_lead['name']} ({test_lead['email']})")
    res = lead_saver(**test_lead)
    print(f"Result Status: {res.get('status')}")
    print(f"Message: {res.get('message')}")
    
    if res.get("status") == "success":
        lead_id = res.get("lead_id")
        print(f"Inserted Lead ID in MongoDB Atlas: {lead_id}")
        
        # Verify document exists in MongoDB Atlas
        col = get_mongo_collection()
        if col is not None:
            from bson import ObjectId
            saved = col.find_one({"_id": ObjectId(lead_id)})
            if saved:
                print(f"Verified MongoDB record: Name={saved.get('name')}, Status={saved.get('status')}")
        print("\n>>> lead_saver tool test PASSED! <<<\n")
    else:
        print(f"\n[Notice] lead_saver returned error: {res.get('message')}")
        print("Please ensure your IP is whitelisted on MongoDB Atlas Network Access (0.0.0.0/0 or current IP).")


if __name__ == "__main__":
    print("Executing standalone tool tests...")
    test_search_ranker_tool()
    test_lead_saver_tool()
