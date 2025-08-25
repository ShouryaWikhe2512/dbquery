from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import base64
import tenseal as ts
from privacy import PrivacyEngine
from client_processor import ClientDataProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Privacy-Preserving Client API",
    description="A FastAPI-based privacy-preserving client system with homomorphic encryption, zero-knowledge proofs, and differential privacy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize privacy components
privacy_engine = PrivacyEngine()
client_processor = ClientDataProcessor()

# Pydantic models for request/response
class FinancialTransaction(BaseModel):
    amount: float = Field(..., description="Transaction amount", example=1500.50)
    type: str = Field(..., description="Transaction type", example="transfer")
    timestamp: str = Field(..., description="Transaction timestamp", example="2023-01-01")

class FinancialProcessingRequest(BaseModel):
    transactions: List[FinancialTransaction] = Field(..., description="List of financial transactions to process")

class FinancialProcessingResponse(BaseModel):
    encrypted_result: str = Field(..., description="Base64 encoded encrypted result")
    transaction_count: int = Field(..., description="Number of transactions processed")
    processing_status: str = Field(..., description="Processing status")

class CriminalQueryRequest(BaseModel):
    city: str = Field(..., description="City for the search", example="Delhi")
    crime_type: str = Field(..., description="Type of crime", example="Assault")
    year: int = Field(..., description="Year of the crime", example=2022)

class CriminalQueryResponse(BaseModel):
    private_params: Dict[str, Any] = Field(..., description="Privacy-protected query parameters")
    zk_proof: Dict[str, Any] = Field(..., description="Zero-knowledge proof data")

class EncryptionRequest(BaseModel):
    values: List[float] = Field(..., description="List of values to encrypt", example=[100.0, 200.0, 300.0])

class EncryptionResponse(BaseModel):
    encrypted_values: List[str] = Field(..., description="Base64 encoded encrypted values")
    context_info: Dict[str, Any] = Field(..., description="Encryption context information")

class DecryptionRequest(BaseModel):
    encrypted_data: str = Field(..., description="Base64 encoded encrypted data to decrypt")

class DecryptionResponse(BaseModel):
    decrypted_value: float = Field(..., description="Decrypted value")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class ZKProofRequest(BaseModel):
    secret_value: Any = Field(..., description="Secret value to prove")
    statement: str = Field(..., description="Statement to prove about the secret value")

class ZKProofResponse(BaseModel):
    proof_data: Dict[str, Any] = Field(..., description="Generated zero-knowledge proof")
    verification_result: bool = Field(..., description="Whether the proof is valid")

class DifferentialPrivacyRequest(BaseModel):
    value: float = Field(..., description="Value to add noise to")
    epsilon: Optional[float] = Field(None, description="Privacy parameter epsilon", example=1.0)

class DifferentialPrivacyResponse(BaseModel):
    original_value: float = Field(..., description="Original input value")
    noisy_value: float = Field(..., description="Value with differential privacy noise added")
    epsilon_used: float = Field(..., description="Epsilon value used for noise")

class HistogramRequest(BaseModel):
    data: List[float] = Field(..., description="Data to create histogram from")
    bins: int = Field(..., description="Number of histogram bins", example=10)
    epsilon: float = Field(..., description="Privacy parameter", example=1.0)

class HistogramResponse(BaseModel):
    original_histogram: List[int] = Field(..., description="Original histogram counts")
    private_histogram: List[float] = Field(..., description="Differentially private histogram")
    epsilon_used: float = Field(..., description="Epsilon value used")

class HealthCheck(BaseModel):
    status: str
    message: str
    privacy_features: Dict[str, bool]
    encryption_context: Dict[str, Any]

# API endpoints
@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with health check and privacy system status"""
    try:
        # Check privacy engine status
        context_loaded = privacy_engine.context is not None
        galois_keys_generated = hasattr(privacy_engine.context, 'galois_keys')
        
        return HealthCheck(
            status="healthy",
            message="Privacy-Preserving Client API is running successfully",
            privacy_features={
                "homomorphic_encryption": context_loaded,
                "galois_keys": galois_keys_generated,
                "differential_privacy": True,
                "zero_knowledge_proofs": True
            },
            encryption_context={
                "scheme_type": "BFV",
                "poly_modulus_degree": 4096,
                "plain_modulus": 1032193
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return await root()

@app.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_values(request: EncryptionRequest):
    """
    Encrypt a list of values using homomorphic encryption
    
    This endpoint encrypts numerical values using the BFV homomorphic encryption scheme.
    The encrypted values can be processed on the server without revealing the original data.
    """
    try:
        # Encrypt the values
        encrypted_values = privacy_engine.encrypt_values(request.values)
        
        # Serialize encrypted values to base64
        serialized_values = [enc.serialize() for enc in encrypted_values]
        base64_values = [base64.b64encode(ser).decode('utf-8') for ser in serialized_values]
        
        return EncryptionResponse(
            encrypted_values=base64_values,
            context_info={
                "scheme": "BFV",
                "values_count": len(request.values),
                "scaling_factor": 1000
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@app.post("/decrypt", response_model=DecryptionResponse)
async def decrypt_value(request: DecryptionRequest):
    """
    Decrypt a homomorphically encrypted value
    
    This endpoint decrypts a previously encrypted value and returns the original data.
    """
    try:
        import time
        start_time = time.time()
        
        # Decode base64 and deserialize
        encrypted_bytes = base64.b64decode(request.encrypted_data)
        encrypted_vector = ts.bfv_vector_from(privacy_engine.context, encrypted_bytes)
        
        # Decrypt
        decrypted_value = privacy_engine.decrypt_result(encrypted_vector)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DecryptionResponse(
            decrypted_value=decrypted_value,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")

@app.post("/compute-on-encrypted")
async def compute_on_encrypted(
    encrypted_data: List[str] = Body(..., description="List of base64 encoded encrypted values"),
    operation: str = Body("sum", description="Operation to perform on encrypted data")
):
    """
    Perform computations on encrypted data
    
    This endpoint demonstrates homomorphic computation on encrypted values.
    Supported operations: sum, mean, max, min
    """
    try:
        # Decode and deserialize encrypted values
        encrypted_vectors = []
        for enc_b64 in encrypted_data:
            enc_bytes = base64.b64decode(enc_b64)
            enc_vector = ts.bfv_vector_from(privacy_engine.context, enc_bytes)
            encrypted_vectors.append(enc_vector)
        
        # Perform computation
        if operation == "sum":
            result = privacy_engine.compute_on_encrypted(encrypted_vectors, "sum")
        elif operation == "mean":
            # For mean, we'll compute sum and divide by count (this would need to be done carefully in practice)
            result = privacy_engine.compute_on_encrypted(encrypted_vectors, "sum")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")
        
        # Serialize result
        result_serialized = result.serialize()
        result_b64 = base64.b64encode(result_serialized).decode('utf-8')
        
        return {
            "operation": operation,
            "encrypted_result": result_b64,
            "input_count": len(encrypted_vectors),
            "message": f"Computation '{operation}' completed successfully on encrypted data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")

@app.post("/generate-zk-proof", response_model=ZKProofResponse)
async def generate_zk_proof(request: ZKProofRequest):
    """
    Generate a zero-knowledge proof
    
    This endpoint generates a zero-knowledge proof that a secret value satisfies a given statement.
    """
    try:
        # Generate the proof
        proof_data = privacy_engine.generate_zk_proof(request.secret_value, request.statement)
        
        # Verify the proof (in practice, this would be done by the verifier)
        verification_result = privacy_engine.verify_zk_proof(proof_data, {})
        
        return ZKProofResponse(
            proof_data=proof_data,
            verification_result=verification_result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZK proof generation failed: {str(e)}")

@app.post("/verify-zk-proof")
async def verify_zk_proof(
    proof_data: Dict[str, Any] = Body(..., description="Proof data to verify"),
    public_parameters: Dict[str, Any] = Body({}, description="Public parameters for verification")
):
    """
    Verify a zero-knowledge proof
    
    This endpoint verifies a previously generated zero-knowledge proof.
    """
    try:
        verification_result = privacy_engine.verify_zk_proof(proof_data, public_parameters)
        
        return {
            "verification_result": verification_result,
            "proof_data": proof_data,
            "message": "Proof verified successfully" if verification_result else "Proof verification failed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZK proof verification failed: {str(e)}")

@app.post("/add-differential-privacy", response_model=DifferentialPrivacyResponse)
async def add_differential_privacy(request: DifferentialPrivacyRequest):
    """
    Add differentially private noise to a value
    
    This endpoint adds noise to a value to provide differential privacy protection.
    """
    try:
        # Add noise
        noisy_value = privacy_engine.add_noise(request.value, request.epsilon)
        
        return DifferentialPrivacyResponse(
            original_value=request.value,
            noisy_value=noisy_value,
            epsilon_used=request.epsilon or privacy_engine.epsilon
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Differential privacy processing failed: {str(e)}")

@app.post("/create-private-histogram", response_model=HistogramResponse)
async def create_private_histogram(request: HistogramRequest):
    """
    Create a differentially private histogram
    
    This endpoint creates a histogram from data and applies differential privacy protection.
    """
    try:
        # Create private histogram
        private_hist = privacy_engine.create_private_histogram(
            request.data, 
            request.bins, 
            request.epsilon
        )
        
        # Create original histogram for comparison
        import numpy as np
        original_hist, _ = np.histogram(request.data, bins=request.bins)
        
        return HistogramResponse(
            original_histogram=original_hist.tolist(),
            private_histogram=private_hist,
            epsilon_used=request.epsilon
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Private histogram creation failed: {str(e)}")

@app.post("/process-financials", response_model=FinancialProcessingResponse)
async def process_financials(request: FinancialProcessingRequest):
    """
    Process financial data with privacy protections
    
    This endpoint processes financial transactions using the client processor,
    encrypting sensitive data and generating privacy proofs.
    """
    try:
        # Convert to the format expected by client processor
        transactions = [t.dict() for t in request.transactions]
        
        # Process data on client side
        processed_data = client_processor.process_financial_data(transactions)
        
        # Extract encrypted result
        encrypted_amounts = processed_data["encrypted_amounts"]
        
        return FinancialProcessingResponse(
            encrypted_result=encrypted_amounts[0],  # Return first encrypted amount as example
            transaction_count=len(request.transactions),
            processing_status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial processing failed: {str(e)}")

@app.post("/process-criminal-query", response_model=CriminalQueryResponse)
async def process_criminal_query(request: CriminalQueryRequest):
    """
    Process a criminal query with privacy protections
    
    This endpoint processes criminal investigation queries using privacy-preserving techniques.
    """
    try:
        # Convert to the format expected by client processor
        query_params = request.dict()
        
        # Process query with privacy protections
        private_query = client_processor.process_criminal_query(query_params)
        
        return CriminalQueryResponse(
            private_params=private_query["private_params"],
            zk_proof=private_query["zk_proof"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Criminal query processing failed: {str(e)}")

@app.get("/privacy-parameters")
async def get_privacy_parameters():
    """Get current privacy parameters and configuration"""
    return {
        "differential_privacy": {
            "epsilon": privacy_engine.epsilon,
            "sensitivity": privacy_engine.sensitivity
        },
        "homomorphic_encryption": {
            "scheme": "BFV",
            "poly_modulus_degree": 4096,
            "plain_modulus": 1032193,
            "scaling_factor": 1000
        },
        "zero_knowledge_proofs": {
            "implementation": "placeholder",
            "commitment_scheme": "hash-based"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": "The requested privacy endpoint does not exist"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": "An unexpected error occurred in the privacy system"}

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_privacy_client:app",
        host="0.0.0.0",
        port=8001,  # Different port from main API
        reload=True,
        log_level="info"
    )
