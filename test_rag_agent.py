#!/usr/bin/env python3
"""
Test script for the RAG Agent
This script demonstrates the RAG agent functionality without requiring the FastAPI server.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import RAGAgent, RAGConfig

def test_rag_agent():
    """Test the RAG agent with various queries"""
    
    print("🚀 Initializing RAG Agent for Criminal Investigation...")
    print("=" * 60)
    
    # Initialize RAG agent
    config = RAGConfig()
    agent = RAGAgent(config)
    
    print(f"✅ RAG Agent initialized successfully!")
    print(f"📚 Knowledge base contains {len(agent.documents)} documents")
    print(f"🔍 Index built: {agent.index is not None}")
    print()
    
    # Test queries
    test_queries = [
        "How do I investigate an assault case?",
        "What is homomorphic encryption?",
        "How to analyze financial risk in criminal cases?",
        "What are the steps for cyber crime investigation?",
        "How does differential privacy work?",
        "What is IPC 351?",
        "How to identify suspicious financial transactions?",
        "What are the best practices for evidence preservation?"
    ]
    
    print("🧪 Testing RAG Agent with various queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process query
            result = agent.process_query(query)
            
            # Display results
            print(f"📝 Response: {result['response'][:200]}...")
            print(f"📊 Confidence Score: {result['confidence_score']:.2f}")
            print(f"🔗 Relevant Documents: {len(result['relevant_documents'])}")
            print(f"🌐 API Calls Suggested: {result['api_calls']}")
            
            # Show top relevant document
            if result['relevant_documents']:
                top_doc = result['relevant_documents'][0]
                print(f"📄 Top Document: {top_doc['document'][:100]}...")
                print(f"🎯 Similarity: {top_doc['similarity']:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print()
    
    print("=" * 60)
    print("🏁 RAG Agent testing completed!")
    
    # Test document retrieval
    print("\n📚 Testing document retrieval...")
    test_query = "assault investigation"
    relevant_docs = agent.retrieve_relevant_documents(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Found {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  {i}. {doc['document'][:80]}... (similarity: {doc['similarity']:.3f})")
    
    # Test knowledge base expansion
    print("\n📖 Testing knowledge base expansion...")
    new_documents = [
        "Forensic accounting examines financial records for evidence of fraud or embezzlement.",
        "Chain of custody documentation ensures evidence integrity throughout investigation.",
        "Digital evidence must be collected using forensically sound methods."
    ]
    
    print(f"Adding {len(new_documents)} new documents...")
    agent.documents.extend(new_documents)
    agent._build_index()
    
    print(f"✅ Knowledge base expanded to {len(agent.documents)} documents")
    
    # Test with new knowledge
    test_query = "forensic accounting"
    result = agent.process_query(test_query)
    print(f"\nQuery: '{test_query}'")
    print(f"Response: {result['response'][:150]}...")
    print(f"Confidence: {result['confidence_score']:.2f}")

def test_api_integration():
    """Test API integration capabilities"""
    print("\n🌐 Testing API Integration Capabilities...")
    print("=" * 60)
    
    config = RAGConfig()
    agent = RAGAgent(config)
    
    # Test API call determination
    test_queries = [
        "Find suspects for assault in Delhi",
        "Encrypt financial transaction data",
        "How to investigate cyber crime",
        "Add differential privacy to risk scores"
    ]
    
    for query in test_queries:
        api_calls = agent._determine_api_calls(query)
        print(f"Query: '{query}'")
        print(f"Suggested API calls: {api_calls}")
        print()

if __name__ == "__main__":
    try:
        test_rag_agent()
        test_api_integration()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo run the full FastAPI server:")
        print("python rag_agent.py")
        print("\nThe RAG agent will be available at:")
        print("- Main API: http://localhost:8002")
        print("- Documentation: http://localhost:8002/docs")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements_rag.txt")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
