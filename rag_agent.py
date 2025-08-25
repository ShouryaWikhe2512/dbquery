import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# RAG Agent Configuration
@dataclass
class RAGConfig:
    model_name: str = "all-MiniLM-L6-v2"  # Lightweight sentence transformer
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    criminal_finder_url: str = "http://localhost:8000"
    privacy_client_url: str = "http://localhost:8001"

class RAGAgent:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.model_name)
        self.index = None
        self.documents = []
        self.document_embeddings = []
        
        # Initialize document store
        self._initialize_document_store()
    
    def _initialize_document_store(self):
        """Initialize the document store with criminal investigation knowledge"""
        # Criminal investigation knowledge base
        knowledge_base = [
            "Criminal investigation involves systematic collection of evidence to solve crimes.",
            "Financial forensics examines financial transactions for suspicious patterns.",
            "Digital forensics recovers and investigates material found in digital devices.",
            "Forensic psychology studies criminal behavior and mental state.",
            "DNA analysis is crucial for identifying suspects and victims.",
            "Surveillance techniques include electronic monitoring and physical observation.",
            "Interrogation methods must follow legal and ethical guidelines.",
            "Evidence preservation is critical for maintaining chain of custody.",
            "Witness testimony can provide crucial information for investigations.",
            "Crime scene analysis helps reconstruct events and identify perpetrators.",
            "Cybercrime investigation requires specialized digital forensic tools.",
            "Money laundering involves disguising the origins of illegally obtained money.",
            "Fraud investigation examines deceptive practices for financial gain.",
            "Assault cases require evidence of physical harm and intent.",
            "Theft investigations focus on stolen property and perpetrator identification.",
            "Homicide investigations are the most complex criminal cases.",
            "Drug trafficking investigations target distribution networks.",
            "Organized crime involves structured criminal enterprises.",
            "White-collar crime includes financial fraud and corporate misconduct.",
            "Juvenile crime requires special handling and rehabilitation focus."
        ]
        
        # Add your specific crime types and procedures
        crime_specific_knowledge = [
            "IPC 66C covers cyber crimes including unauthorized access to computer systems.",
            "IPC 351 defines assault as causing apprehension of harm or offensive contact.",
            "IPC 420 covers cheating and dishonestly inducing delivery of property.",
            "IPC 379 defines theft as dishonestly taking property without consent.",
            "Financial risk scoring considers transaction amounts, frequency, and flags.",
            "Suspicious transaction patterns include unusual amounts and timing.",
            "Risk assessment combines criminal history with financial behavior analysis.",
            "Privacy-preserving techniques protect sensitive investigation data.",
            "Homomorphic encryption allows computation on encrypted data.",
            "Differential privacy adds noise to protect individual identities.",
            "Zero-knowledge proofs verify claims without revealing secrets."
        ]
        
        self.documents = knowledge_base + crime_specific_knowledge
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index for document retrieval"""
        # Generate embeddings for all documents
        embeddings = self.embedding_model.encode(self.documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.document_embeddings = embeddings
    
    def retrieve_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        similarities, indices = self.index.search(
            query_embedding.astype('float32'), 
            self.config.top_k
        )
        
        # Filter by similarity threshold
        relevant_docs = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= self.config.similarity_threshold:
                relevant_docs.append({
                    "document": self.documents[idx],
                    "similarity": float(sim),
                    "index": int(idx)
                })
        
        return relevant_docs
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved context"""
        if not context_docs:
            return "I don't have enough information to answer that question accurately."
        
        # Create context from retrieved documents
        context = "\n".join([doc["document"] for doc in context_docs])
        
        # Generate response based on query type and context
        if "crime" in query.lower() or "investigation" in query.lower():
            response = self._generate_crime_investigation_response(query, context)
        elif "privacy" in query.lower() or "encryption" in query.lower():
            response = self._generate_privacy_response(query, context)
        elif "financial" in query.lower() or "risk" in query.lower():
            response = self._generate_financial_response(query, context)
        else:
            response = self._generate_general_response(query, context)
        
        return response
    
    def _generate_crime_investigation_response(self, query: str, context: str) -> str:
        """Generate response for crime investigation queries"""
        base_response = f"Based on the available information about criminal investigations:\n\n{context}\n\n"
        
        if "assault" in query.lower():
            base_response += "For assault cases (IPC 351), focus on evidence of physical harm, intent, and witness testimony. Financial risk analysis can help identify patterns in suspect behavior."
        elif "cyber" in query.lower():
            base_response += "Cyber crime investigations (IPC 66C) require digital forensics expertise. Analyze financial transactions for suspicious patterns and use privacy-preserving techniques for sensitive data."
        elif "theft" in query.lower():
            base_response += "Theft investigations (IPC 379) should examine financial records, surveillance footage, and physical evidence. Risk scoring can help prioritize suspects."
        
        return base_response
    
    def _generate_privacy_response(self, query: str, context: str) -> str:
        """Generate response for privacy-related queries"""
        base_response = f"Regarding privacy and security in investigations:\n\n{context}\n\n"
        
        if "encryption" in query.lower():
            base_response += "Homomorphic encryption allows you to perform computations on encrypted data without revealing the original values. This is crucial for protecting sensitive financial information during investigations."
        elif "differential" in query.lower():
            base_response += "Differential privacy adds controlled noise to data, protecting individual identities while maintaining statistical accuracy for analysis."
        
        return base_response
    
    def _generate_financial_response(self, query: str, context: str) -> str:
        """Generate response for financial investigation queries"""
        base_response = f"Financial investigation guidance:\n\n{context}\n\n"
        
        if "risk" in query.lower():
            base_response += "Financial risk scoring considers transaction amounts, frequency, suspicious flags, and patterns. Higher scores indicate increased risk and require closer investigation."
        elif "suspicious" in query.lower():
            base_response += "Look for unusual transaction patterns, large amounts, frequent transfers, and flagged transactions. These indicators suggest potential criminal activity."
        
        return base_response
    
    def _generate_general_response(self, query: str, context: str) -> str:
        """Generate general response for other queries"""
        return f"Based on the available information:\n\n{context}\n\nThis context should help guide your investigation approach."
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return comprehensive response"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query)
        
        # Generate response
        response = self.generate_response(query, relevant_docs)
        
        # Check if we should call external APIs
        api_calls = self._determine_api_calls(query)
        
        return {
            "query": query,
            "response": response,
            "relevant_documents": relevant_docs,
            "api_calls": api_calls,
            "confidence_score": self._calculate_confidence(relevant_docs)
        }
    
    def _determine_api_calls(self, query: str) -> List[str]:
        """Determine which external APIs should be called"""
        api_calls = []
        
        if any(word in query.lower() for word in ["suspect", "crime", "delhi", "mumbai", "assault", "theft"]):
            api_calls.append("criminal_finder")
        
        if any(word in query.lower() for word in ["privacy", "encrypt", "financial", "transaction"]):
            api_calls.append("privacy_client")
        
        return api_calls
    
    def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on document relevance"""
        if not relevant_docs:
            return 0.0
        
        avg_similarity = np.mean([doc["similarity"] for doc in relevant_docs])
        return min(1.0, avg_similarity * 1.2)  # Boost confidence slightly
    
    def call_criminal_finder_api(self, query: str) -> Dict[str, Any]:
        """Call the criminal finder API"""
        try:
            response = requests.get(
                f"{self.config.criminal_finder_url}/find-suspects",
                params={"query": query}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API call failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"API call error: {str(e)}"}
    
    def call_privacy_client_api(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the privacy client API"""
        try:
            if operation == "encrypt":
                response = requests.post(
                    f"{self.config.privacy_client_url}/encrypt",
                    json=data
                )
            elif operation == "add-differential-privacy":
                response = requests.post(
                    f"{self.config.privacy_client_url}/add-differential-privacy",
                    json=data
                )
            else:
                return {"error": f"Unsupported operation: {operation}"}
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API call failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"API call error: {str(e)}"}

# FastAPI Application
app = FastAPI(
    title="RAG Agent for Criminal Investigation",
    description="An intelligent RAG agent that combines document retrieval with external APIs for comprehensive criminal investigation support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG agent
rag_agent = RAGAgent(RAGConfig())

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query to process", example="How do I investigate an assault case in Delhi?")
    include_api_calls: bool = Field(True, description="Whether to include external API calls")

class QueryResponse(BaseModel):
    query: str
    response: str
    relevant_documents: List[Dict[str, Any]]
    api_calls: List[str]
    confidence_score: float
    external_results: Optional[Dict[str, Any]] = None

class DocumentUpdateRequest(BaseModel):
    documents: List[str] = Field(..., description="New documents to add to the knowledge base")

class DocumentUpdateResponse(BaseModel):
    message: str
    total_documents: int

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Agent for Criminal Investigation",
        "status": "active",
        "endpoints": {
            "query": "/query",
            "documents": "/documents",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "RAG Agent",
        "documents_count": len(rag_agent.documents),
        "index_built": rag_agent.index is not None
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the RAG agent"""
    try:
        # Process query with RAG agent
        result = rag_agent.process_query(request.query)
        
        # Add external API results if requested
        external_results = {}
        if request.include_api_calls and result["api_calls"]:
            if "criminal_finder" in result["api_calls"]:
                external_results["criminal_finder"] = rag_agent.call_criminal_finder_api(request.query)
            
            if "privacy_client" in result["api_calls"]:
                # Example privacy operation
                external_results["privacy_client"] = rag_agent.call_privacy_client_api(
                    "add-differential-privacy",
                    {"value": 1000.0, "epsilon": 1.0}
                )
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            relevant_documents=result["relevant_documents"],
            api_calls=result["api_calls"],
            confidence_score=result["confidence_score"],
            external_results=external_results if external_results else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get all documents in the knowledge base"""
    return {
        "total_count": len(rag_agent.documents),
        "documents": rag_agent.documents
    }

@app.post("/documents", response_model=DocumentUpdateResponse)
async def update_documents(request: DocumentUpdateRequest):
    """Add new documents to the knowledge base"""
    try:
        # Add new documents
        rag_agent.documents.extend(request.documents)
        
        # Rebuild index
        rag_agent._build_index()
        
        return DocumentUpdateResponse(
            message=f"Added {len(request.documents)} new documents",
            total_documents=len(rag_agent.documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document update failed: {str(e)}")

@app.get("/suggestions")
async def get_suggestions(q: str):
    """Get query suggestions based on partial input"""
    try:
        # Find documents that contain the partial query
        suggestions = []
        for doc in rag_agent.documents:
            if q.lower() in doc.lower():
                suggestions.append(doc)
        
        return {
            "query": q,
            "suggestions": suggestions[:5]  # Limit to 5 suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "rag_agent:app",
        host="0.0.0.0",
        port=8002,  # Different port from other APIs
        reload=True,
        log_level="info"
    )
