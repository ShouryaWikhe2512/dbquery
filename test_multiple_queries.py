#!/usr/bin/env python3
"""
Test multiple queries to demonstrate RAG agent capabilities
"""

import requests
import json

def test_multiple_queries():
    """Test various types of queries"""
    
    base_url = "http://localhost:8002"
    
    print("🧪 Testing Multiple Query Types...")
    print("=" * 60)
    
    # Test queries covering different domains
    test_queries = [
        {
            "query": "How do I investigate an assault case?",
            "description": "Crime investigation query"
        },
        {
            "query": "What is homomorphic encryption?",
            "description": "Privacy/encryption query"
        },
        {
            "query": "How to analyze financial risk in criminal cases?",
            "description": "Financial investigation query"
        },
        {
            "query": "What are the steps for cyber crime investigation?",
            "description": "Cyber crime query"
        },
        {
            "query": "How does differential privacy work?",
            "description": "Privacy technique query"
        },
        {
            "query": "What is IPC 351?",
            "description": "Legal code query"
        },
        {
            "query": "How to identify suspicious financial transactions?",
            "description": "Financial forensics query"
        },
        {
            "query": "What are the best practices for evidence preservation?",
            "description": "Evidence handling query"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 50)
        
        try:
            query_data = {
                "query": test_case['query'],
                "include_api_calls": True
            }
            
            response = requests.post(
                f"{base_url}/query",
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Response: {data['response'][:150]}...")
                print(f"📊 Confidence: {data['confidence_score']:.2f}")
                print(f"🔗 Documents: {len(data['relevant_documents'])} relevant")
                print(f"🌐 API Calls: {data['api_calls']}")
                
                # Show top relevant document if available
                if data['relevant_documents']:
                    top_doc = data['relevant_documents'][0]
                    print(f"📄 Top Doc: {top_doc['document'][:80]}...")
                    print(f"🎯 Similarity: {top_doc['similarity']:.3f}")
                
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 60)
    print("🏁 Multiple query testing completed!")

if __name__ == "__main__":
    test_multiple_queries()
