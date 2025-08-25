# privacy_preserving_client.py
import requests
import json
from client_processor import ClientDataProcessor
import base64
import tenseal as ts

class PrivacyPreservingClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.client_processor = ClientDataProcessor()
    
    def test_financial_processing(self):
        """Test the financial data processing endpoint"""
        # Sample financial transactions
        transactions = [
            {"amount": 1500.50, "type": "transfer", "timestamp": "2023-01-01"},
            {"amount": 2500.75, "type": "withdrawal", "timestamp": "2023-01-02"},
            {"amount": 800.25, "type": "deposit", "timestamp": "2023-01-03"}
        ]
        
        try:
            # Process data on client side
            processed_financials = self.client_processor.process_financial_data(transactions)
            
            # Send to server
            response = requests.post(
                f"{self.server_url}/process-financials",
                json=processed_financials
            )
            
            if response.status_code == 200:
                encrypted_result_b64 = response.json()
                encrypted_result_bytes = base64.b64decode(encrypted_result_b64)
                decrypted = self.client_processor.privacy_engine.decrypt_result(
                    ts.bfv_vector_from(self.client_processor.privacy_engine.context, encrypted_result_bytes)
                )
                print("✅ Financial processing test successful!")
                print(f"Private total: {decrypted}")
            else:
                print(f"❌ Financial processing test failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during financial processing test: {e}")
    
    def test_suspect_finding(self):
        """Test the suspect finding endpoint"""
        # Sample query parameters
        query_params = {
            "city": "Delhi",
            "crime_type": "Assault",
            "year": 2022
        }
        
        try:
            # Process query with privacy protections
            private_query = self.client_processor.process_criminal_query(query_params)
            
            # Send to server
            response = requests.post(
                f"{self.server_url}/find-suspects",
                json=private_query
            )
            
            if response.status_code == 200:
                print("✅ Suspect finding test successful!")
                print(f"Response: {response.json()}")
            else:
                print(f"❌ Suspect finding test failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during suspect finding test: {e}")
    
    def run_all_tests(self):
        """Run all available tests"""
        print("🚀 Starting Privacy-Preserving Criminal Finder Tests...")
        print("=" * 60)
        
        self.test_financial_processing()
        print()
        self.test_suspect_finding()
        
        print("\n" + "=" * 60)
        print("🏁 All tests completed!")

if __name__ == "__main__":
    client = PrivacyPreservingClient()
    client.run_all_tests()