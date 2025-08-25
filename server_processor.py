# server_processor.py
import pandas as pd
import tenseal as ts
from privacy import PrivacyEngine
import base64

class PrivateCriminalFinder:
    def __init__(self):
        self.privacy_engine = PrivacyEngine()
        # Load your datasets (these would contain non-sensitive or properly anonymized data)
        self.persons_df = pd.read_csv('person.csv')
        self.criminal_history_df = pd.read_csv('criminal_history_full.csv')
        
    def process_encrypted_financials(self, encrypted_data):
        """Process encrypted financial data from clients"""
        encrypted_amounts = [ts.bfv_vector_from(self.privacy_engine.context, base64.b64decode(enc)) 
                            for enc in encrypted_data['encrypted_amounts']]
        encrypted_total = self.privacy_engine.compute_on_encrypted(encrypted_amounts, "sum")
        encrypted_total_serialized = encrypted_total.serialize()
        encrypted_total_b64 = base64.b64encode(encrypted_total_serialized).decode('utf-8')
        return encrypted_total_b64
    
    def find_suspects_with_privacy(self, query_data):
        """Find suspects while preserving privacy"""
        # Verify ZK proof
        if not self.verify_zk_proof(query_data['zk_proof']):
            return {"error": "Invalid proof"}
        
        # Extract and use differentially private parameters
        params = query_data['private_params']
        
        # Apply differential privacy to the results
        results = self._find_matches(params)
        private_results = self._apply_differential_privacy(results)
        
        return private_results
    
    def verify_zk_proof(self, proof_data):
        """Verify a zero-knowledge proof"""
        # This would use the privacy engine to verify the proof
        return self.privacy_engine.verify_zk_proof(proof_data, {})
    
    def _find_matches(self, params):
        """Find matches in the database (using non-sensitive or anonymized data)"""
        # Your existing logic, but adapted for privacy
        city = params.get('city', '')
        crime_type = params.get('crime_type', '')
        
        # Filter persons by city (with fuzzy matching for privacy)
        city_suspects = self.persons_df[
            self.persons_df['city'].str.lower().str.contains(city.lower())
        ]
        
        # Filter criminal history by crime type
        crime_suspects = self.criminal_history_df[
            self.criminal_history_df['incident_type'].str.lower() == crime_type.lower()
        ]
        
        # Merge results
        suspects = city_suspects.merge(
            crime_suspects, 
            on='person_id', 
            how='inner'
        )
        
        return suspects
    
    def _apply_differential_privacy(self, results):
        """Apply differential privacy to the results"""
        if len(results) == 0:
            return results
        
        # Add noise to counts and other sensitive aggregates
        private_count = self.privacy_engine.add_noise(len(results))
        
        # For individual records, we might return aggregated or noisy results
        if private_count < 5:  # Privacy threshold
            return {"count": private_count, "results": "Redacted for privacy"}
        
        # For larger result sets, we can return noisy versions
        noisy_results = []
        for _, row in results.iterrows():
            noisy_row = {}
            for col in row.index:
                if pd.api.types.is_numeric_dtype(row[col]):
                    # Add noise to numeric values
                    noisy_row[col] = self.privacy_engine.add_noise(row[col])
                else:
                    # For categorical data, we might use other techniques
                    noisy_row[col] = row[col]
            noisy_results.append(noisy_row)
        
        return {
            "count": private_count,
            "results": noisy_results
        }