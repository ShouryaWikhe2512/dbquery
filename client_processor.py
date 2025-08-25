# client_processor.py
import json
import base64
from privacy import PrivacyEngine

class ClientDataProcessor:
    def __init__(self):
        self.privacy_engine = PrivacyEngine()
    
    def process_financial_data(self, transactions):
        """Process financial data on the client side before sending to server"""
        amounts = [t['amount'] for t in transactions]
        encrypted_amounts = self.privacy_engine.encrypt_values(amounts)
        total_amount = sum(amounts)
        proof = self.privacy_engine.generate_zk_proof(
            total_amount, 
            "total_amount > 10000 AND total_amount < 1000000"
        )
        processed_data = {
            "encrypted_amounts": [base64.b64encode(enc.serialize()).decode('utf-8') for enc in encrypted_amounts],
            "zk_proof": proof,
            "transaction_count": len(transactions),
        }
        return processed_data
    
    def process_criminal_query(self, query_params):
        """Process a criminal query with privacy protections"""
        # Generate ZK proof that the query meets certain criteria
        # (e.g., user is authorized to make this query)
        proof = self.privacy_engine.generate_zk_proof(
            query_params, 
            "user_is_authorized AND query_is_valid"
        )
        
        # Add differential privacy to any sensitive parameters
        private_params = {}
        for key, value in query_params.items():
            if key in ['age', 'income']:  # Sensitive numerical attributes
                private_params[key] = self.privacy_engine.add_noise(value)
            else:
                private_params[key] = value
        
        return {
            "private_params": private_params,
            "zk_proof": proof
        }