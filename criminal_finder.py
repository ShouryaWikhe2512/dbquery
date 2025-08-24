import pandas as pd
import re
from typing import List, Dict, Any, Optional
import numpy as np

class CriminalFinder:
    def __init__(self):
        self.persons_df = None
        self.criminal_history_df = None
        self.financial_transactions_df = None
        self.load_data()
        
    def load_data(self):
        """Load all necessary CSV files into DataFrames"""
        try:
            self.persons_df = pd.read_csv('person.csv')
            self.criminal_history_df = pd.read_csv('criminal_history_full.csv')
            self.financial_transactions_df = pd.read_csv('financial_transactions.csv')
            print("Data loaded successfully!")
            print(f"Persons: {len(self.persons_df)} records")
            print(f"Criminal History: {len(self.criminal_history_df)} records")
            print(f"Financial Transactions: {len(self.financial_transactions_df)} records")
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please ensure you have persons.csv, criminal_history_full.csv, and financial_transactions.csv in your directory")

    def parse_query(self, query: str) -> Dict[str, str]:
        """Extract key information from natural language query"""
        query = query.lower()
        
        # Define crime patterns to look for - updated to match your CSV
        crime_patterns = {
            'Cyber Crime': r'cyber|hack|online|digital|internet|computer|66c',
            'Assault': r'assault|attack|hurt|harm|violence|beat|351',
            'Cheating': r'cheat|fraud|scam|dupe|swindle|420',
            'Theft': r'theft|rob|steal|stole|thief|burglary|379'
        }
        
        # Define city patterns to look for
        city_patterns = {
            'mumbai': r'mumbai|bombay',
            'delhi': r'delhi',
            'bangalore': r'bangalore|bengaluru',
            'chennai': r'chennai|madras',
            'kolkata': r'kolkata|calcutta',
            'pune': r'pune'
        }
        
        # Find matching crime type
        crime_type = None
        for crime, pattern in crime_patterns.items():
            if re.search(pattern, query):
                crime_type = crime
                break
        
        # Find matching city
        city = None
        for city_name, pattern in city_patterns.items():
            if re.search(pattern, query):
                city = city_name
                break
        
        # Extract year if mentioned
        year_match = re.search(r'(20\d{2})', query)
        year = int(year_match.group(1)) if year_match else None
        
        return {'crime_type': crime_type, 'city': city, 'year': year}

    def find_suspects(self, crime_type: str, city: str, year: Optional[int] = None) -> pd.DataFrame:
        """Find suspects matching the crime type and city"""
        if self.persons_df is None or self.criminal_history_df is None:
            return pd.DataFrame()
            
        # Filter persons by city
        city_suspects = self.persons_df[self.persons_df['city'].str.lower() == city.lower()]
        
        if city_suspects.empty:
            print(f"No persons found in city: {city}")
            return pd.DataFrame()
        
        # Filter criminal history by crime type
        crime_suspects = self.criminal_history_df[
            self.criminal_history_df['incident_type'].str.lower() == crime_type.lower()
        ]
        
        # Filter by year if specified
        if year:
            crime_suspects = crime_suspects[crime_suspects['incident_year'] == year]
        
        if crime_suspects.empty:
            print(f"No criminal history found for crime type: {crime_type}")
            if year:
                print(f"Filtered for year: {year}")
            return pd.DataFrame()
        
        # Merge to get complete suspect information
        suspects = city_suspects.merge(
            crime_suspects, 
            on='person_id', 
            how='inner'
        )
        
        return suspects

    def analyze_finances(self, suspects_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze financial transactions of suspects and calculate risk scores"""
        if suspects_df.empty or self.financial_transactions_df is None:
            return pd.DataFrame()
        
        # Get transactions for our suspects (both as sender and receiver)
        suspect_ids = suspects_df['person_id'].tolist()
        
        # Get transactions where suspects are either sender or receiver
        suspect_transactions = self.financial_transactions_df[
            (self.financial_transactions_df['sender_id'].isin(suspect_ids)) |
            (self.financial_transactions_df['receiver_id'].isin(suspect_ids))
        ]
        
        # Calculate financial risk indicators for each suspect
        financial_analysis = []
        
        for person_id in suspect_ids:
            # Get transactions where this person is either sender or receiver
            person_transactions = suspect_transactions[
                (suspect_transactions['sender_id'] == person_id) |
                (suspect_transactions['receiver_id'] == person_id)
            ]
            
            if not person_transactions.empty:
                # Calculate various risk indicators
                total_amount = person_transactions['amount'].sum()
                avg_transaction = person_transactions['amount'].mean()
                max_transaction = person_transactions['amount'].max()
                flagged_count = person_transactions['is_flagged'].sum()
                transaction_count = len(person_transactions)
                
                # Calculate a risk score based on transaction patterns
                risk_score = (
                    (total_amount / 100000) +  # Larger amounts are riskier
                    (max_transaction / 50000) +  # Large single transactions are riskier
                    (flagged_count * 2) +  # Flagged transactions significantly increase risk
                    (transaction_count / 20)  # More transactions slightly increase risk
                )
                
                # Normalize risk score to 0-100 range
                risk_score = min(100, risk_score * 10)
                
                financial_analysis.append({
                    'person_id': person_id,
                    'total_amount': total_amount,
                    'avg_transaction': avg_transaction,
                    'max_transaction': max_transaction,
                    'flagged_count': flagged_count,
                    'transaction_count': transaction_count,
                    'financial_risk_score': risk_score
                })
        
        financial_df = pd.DataFrame(financial_analysis)
        
        # Merge financial analysis with suspect information
        if not financial_df.empty:
            full_suspect_info = suspects_df.merge(financial_df, on='person_id', how='left')
            # Fill NaN values for suspects with no financial data
            full_suspect_info.fillna({
                'financial_risk_score': 0,
                'total_amount': 0,
                'avg_transaction': 0,
                'max_transaction': 0,
                'flagged_count': 0,
                'transaction_count': 0
            }, inplace=True)
            return full_suspect_info
        else:
            return suspects_df.assign(
                financial_risk_score=0,
                total_amount=0,
                avg_transaction=0,
                max_transaction=0,
                flagged_count=0,
                transaction_count=0
            )

    def criminal_finder(self, query: str) -> Dict[str, Any]:
        """Main function to find criminals based on natural language query"""
        if self.persons_df is None or self.criminal_history_df is None:
            return {"error": "Could not load data files"}
        
        # Parse query
        query_info = self.parse_query(query)
        crime_type = query_info['crime_type']
        city = query_info['city']
        year = query_info['year']
        
        if not crime_type:
            return {"error": "Could not determine crime type from query"}
        if not city:
            return {"error": "Could not determine city from query"}
        
        # Find suspects
        suspects = self.find_suspects(crime_type, city, year)
        
        if suspects.empty:
            return {"message": "No suspects found matching your criteria"}
        
        # Analyze financial transactions
        suspects_with_finances = self.analyze_finances(suspects)
        
        # Sort by financial risk score (highest risk first)
        ranked_suspects = suspects_with_finances.sort_values('financial_risk_score', ascending=False)
        
        # Format results
        results = []
        for _, suspect in ranked_suspects.iterrows():
            result = {
                'person_id': suspect['person_id'],
                'name': suspect['name'],
                'age': suspect['age'],
                'city': suspect['city'],
                'crime_type': suspect['incident_type'],
                'ipc_section': suspect['ipc_section'],
                'crime_year': suspect['incident_year'],
                'crime_severity': suspect['severity_level'],
                'conviction_status': suspect['conviction_status'],
                'financial_risk_score': round(suspect['financial_risk_score'], 2),
                'suspicious_transactions': suspect.get('flagged_count', 0),
                'total_transaction_amount': suspect.get('total_amount', 0)
            }
            results.append(result)
        
        return {
            'query': query,
            'crime_type': crime_type,
            'city': city,
            'year_filter': year,
            'suspect_count': len(results),
            'suspects': results
        }

    def print_results(self, result: Dict[str, Any]):
        """Print the results in a readable format"""
        if 'error' in result:
            print(f"Error: {result['error']}")
        elif 'message' in result:
            print(result['message'])
        else:
            print(f"\n{'='*80}")
            print(f"QUERY: {result['query']}")
            print(f"Found {result['suspect_count']} suspects for {result['crime_type']} in {result['city']}")
            if result['year_filter']:
                print(f"Filtered for year: {result['year_filter']}")
            print(f"{'='*80}")
            
            for i, suspect in enumerate(result['suspects'], 1):
                print(f"\n{i}. {suspect['name']} (ID: {suspect['person_id']}, Age: {suspect['age']})")
                print(f"   Crime: {suspect['crime_type']} ({suspect['ipc_section']})")
                print(f"   Year: {suspect['crime_year']}, Severity: {suspect['crime_severity']}/5")
                print(f"   Status: {suspect['conviction_status']}")
                print(f"   Financial Risk Score: {suspect['financial_risk_score']}/100")
                print(f"   Suspicious Transactions: {suspect['suspicious_transactions']}")
                print(f"   Total Transaction Amount: ₹{suspect['total_transaction_amount']:,.2f}")

# Example usage
def main():
    finder = CriminalFinder()
    
    test_queries = [
        "bank robbery in Pune",
        "assault in Delhi in 2022",
        "cyber crime in pune 2023"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Processing query: {query}")
        result = finder.criminal_finder(query)
        finder.print_results(result)

if __name__ == "__main__":
    main()