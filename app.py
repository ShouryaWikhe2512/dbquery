# app.py
from flask import Flask, request, jsonify
from server_processor import PrivateCriminalFinder
from client_processor import ClientDataProcessor

app = Flask(__name__)
finder = PrivateCriminalFinder()
client_processor = ClientDataProcessor()

@app.route('/process-financials', methods=['POST'])
def process_financials():
    """Endpoint for processing financial data with privacy"""
    data = request.json
    encrypted_result = finder.process_encrypted_financials(data)
    return jsonify(encrypted_result)

@app.route('/find-suspects', methods=['POST'])
def find_suspects():
    """Endpoint for finding suspects with privacy protections"""
    query_data = request.json
    results = finder.find_suspects_with_privacy(query_data)
    return jsonify(results)

@app.route('/decrypt-result', methods=['POST'])
def decrypt_result():
    """Endpoint for decrypting results (would be called by client)"""
    encrypted_result = request.json['encrypted_result']
    decrypted = client_processor.privacy_engine.decrypt_result(
        ts.bfv_vector_from(client_processor.privacy_engine.context, encrypted_result)
    )
    return jsonify({"result": decrypted})

if __name__ == '__main__':
    app.run(debug=True)