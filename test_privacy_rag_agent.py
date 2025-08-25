#!/usr/bin/env python3
"""
Test script for the Privacy-Preserving RAG Agent
This script demonstrates the agent's ability to process CSV data and find suspects while maintaining privacy.
"""

import sys
import os
import requests
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_privacy_preserving_rag_agent():
    """Test the privacy-preserving RAG agent with various suspect search queries"""
    
    base_url = "http://localhost:8003"
    
    print("🔒 Testing Privacy-Preserving RAG Agent...")
    print("=" * 70)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['status']}")
            print(f"   Agent: {data['agent']}")
            print(f"   Documents: {data['documents_count']}")
            print(f"   CSV Files: {data['csv_files_loaded']}")
            print(f"   Privacy Techniques: {data['privacy_techniques']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    print()
    
    # Test CSV data summary
    try:
        response = requests.get(f"{base_url}/csv-data-summary")
        if response.status_code == 200:
            data = response.json()
            print("📊 CSV Data Summary:")
            for filename, summary in data.items():
                print(f"   📁 {filename}: {summary['records']} records, {len(summary['columns'])} columns")
                print(f"      Columns: {', '.join(summary['columns'][:5])}{'...' if len(summary['columns']) > 5 else ''}")
        else:
            print(f"❌ CSV summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ CSV summary error: {e}")
    
    print()
    
    # Test suspect search queries
    test_queries = [
        {
            "crime_type": "assault",
            "location": "Delhi",
            "description": "Assault cases in Delhi"
        },
        {
            "crime_type": "theft", 
            "location": "Mumbai",
            "description": "Theft cases in Mumbai"
        },
        {
            "crime_type": "cyber",
            "location": "Bangalore", 
            "description": "Cyber crime in Bangalore"
        },
        {
            "crime_type": "fraud",
            "location": "Chennai",
            "description": "Fraud cases in Chennai"
        }
    ]
    
    print("🔍 Testing Suspect Search with Privacy Protection...")
    print("=" * 70)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"Crime Type: {test_case['crime_type']}, Location: {test_case['location']}")
        print("-" * 60)
        
        try:
            # Test direct suspect search
            suspect_data = {
                "crime_type": test_case['crime_type'],
                "location": test_case['location'],
                "include_risk_analysis": True
            }
            
            response = requests.post(
                f"{base_url}/find-suspects",
                json=suspect_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Suspects found: {data['suspects_found']}")
                print(f"🔒 Privacy techniques: {', '.join(data['privacy_techniques_used'])}")
                
                # Show privacy-protected results
                if data['privacy_protected_results']:
                    print(f"📋 Privacy-protected results:")
                    for j, suspect in enumerate(data['privacy_protected_results'][:3], 1):
                        print(f"   {j}. Crime: {suspect['crime_type']}, City: {suspect['city']}")
                        print(f"      Year: {suspect['year']}, Privacy Level: {suspect['privacy_level']}")
                        print(f"      ZK Proof: {suspect['proof_of_existence'][:20]}...")
                        print(f"      Risk Score: {suspect['encrypted_risk_score'][:20]}...")
                else:
                    print("   No suspects found in this category")
                
            else:
                print(f"❌ Suspect search failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 70)
    
    # Test general query processing
    print("🧠 Testing General Query Processing...")
    print("=" * 70)
    
    general_queries = [
        "Find suspects for assault in Delhi",
        "How to investigate cyber crime using privacy-preserving techniques?",
        "What is the risk assessment for theft cases in Mumbai?",
        "Explain how homomorphic encryption protects suspect identities"
    ]
    
    for i, query in enumerate(general_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        try:
            query_data = {
                "query": query,
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
                print(f"🔒 Privacy Guarantees:")
                for guarantee, status in data['privacy_guarantees'].items():
                    print(f"   {guarantee}: {'✅' if status else '❌'}")
                
                # Show suspect identification results
                if 'suspect_identification' in data and 'suspects_found' in data['suspect_identification']:
                    suspects_found = data['suspect_identification']['suspects_found']
                    print(f"🔍 Suspects identified: {suspects_found}")
                
            else:
                print(f"❌ Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 70)
    print("🏁 Privacy-Preserving RAG Agent testing completed!")

def test_csv_processing_capabilities():
    """Test the CSV processing and data encryption capabilities"""
    
    base_url = "http://localhost:8003"
    
    print("\n📊 Testing CSV Processing Capabilities...")
    print("=" * 70)
    
    # Test different crime types and locations
    test_cases = [
        {"crime_type": "assault", "location": "Delhi"},
        {"crime_type": "theft", "location": "Mumbai"},
        {"crime_type": "cyber", "location": "Bangalore"},
        {"crime_type": "fraud", "location": "Chennai"},
        {"crime_type": "drugs", "location": "Kolkata"}
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['crime_type']} in {test_case['location']}")
        print("-" * 50)
        
        try:
            response = requests.post(
                f"{base_url}/find-suspects",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Suspects found: {data['suspects_found']}")
                
                if data['suspects_found'] > 0:
                    print(f"🔒 Privacy protection applied:")
                    for technique in data['privacy_techniques_used']:
                        print(f"   - {technique}")
                    
                    # Show sample of privacy-protected results
                    sample_result = data['privacy_protected_results'][0]
                    print(f"📋 Sample protected result:")
                    print(f"   Crime Type: {sample_result['crime_type']}")
                    print(f"   City: {sample_result['city']}")
                    print(f"   Privacy Level: {sample_result['privacy_level']}")
                    print(f"   ZK Proof: {sample_result['proof_of_existence'][:30]}...")
                else:
                    print("   No suspects found in this category")
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        test_privacy_preserving_rag_agent()
        test_csv_processing_capabilities()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo run the full Privacy-Preserving RAG Agent server:")
        print("python privacy_preserving_rag_agent.py")
        print("\nThe agent will be available at:")
        print("- Main API: http://localhost:8003")
        print("- Documentation: http://localhost:8003/docs")
        print("- Interactive docs: http://localhost:8003/redoc")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check that all required modules are available.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
