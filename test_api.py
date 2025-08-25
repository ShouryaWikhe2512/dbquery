#!/usr/bin/env python3
"""
Simple API test script for the RAG Agent
"""

import requests
import json

def test_rag_api():
    """Test the RAG agent API endpoints"""
    
    base_url = "http://localhost:8002"
    
    print("🧪 Testing RAG Agent API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['status']}")
            print(f"   Agent: {data['agent']}")
            print(f"   Documents: {data['documents_count']}")
            print(f"   Algorithm: {data['algorithm']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test query endpoint
    try:
        query_data = {
            "query": "How do I investigate an assault case?",
            "include_api_calls": True
        }
        
        response = requests.post(
            f"{base_url}/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query processed successfully!")
            print(f"   Query: {data['query']}")
            print(f"   Response: {data['response'][:100]}...")
            print(f"   Confidence: {data['confidence_score']:.2f}")
            print(f"   Relevant Documents: {len(data['relevant_documents'])}")
            print(f"   API Calls: {data['api_calls']}")
            
            if data['external_results']:
                print(f"   External Results: {list(data['external_results'].keys())}")
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Query error: {e}")
    
    print()
    
    # Test documents endpoint
    try:
        response = requests.get(f"{base_url}/documents")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Documents retrieved: {data['total_count']} total")
        else:
            print(f"❌ Documents failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Documents error: {e}")
    
    print()
    
    # Test suggestions endpoint
    try:
        response = requests.get(f"{base_url}/suggestions?q=assault")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Suggestions for 'assault': {len(data['suggestions'])} found")
            for i, suggestion in enumerate(data['suggestions'][:3], 1):
                print(f"   {i}. {suggestion[:80]}...")
        else:
            print(f"❌ Suggestions failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Suggestions error: {e}")

if __name__ == "__main__":
    test_rag_api()
