from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from criminal_finder import CriminalFinder

# Initialize FastAPI app
app = FastAPI(
    title="Privacy-Preserving Criminal Finder API",
    description="A FastAPI-based criminal investigation system with privacy protections",
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

# Initialize the criminal finder
finder = CriminalFinder()

# Pydantic models for request/response
class CriminalQuery(BaseModel):
    query: str = Field(..., description="Natural language query describing the crime and location", example="assault in Delhi in 2022")
    
class SuspectInfo(BaseModel):
    person_id: str
    name: str
    age: int
    city: str
    crime_type: str
    ipc_section: str
    crime_year: int
    crime_severity: int
    conviction_status: str
    financial_risk_score: float
    suspicious_transactions: int
    total_transaction_amount: float

class CriminalQueryResponse(BaseModel):
    query: str
    crime_type: str
    city: str
    year_filter: Optional[int]
    suspect_count: int
    suspects: List[SuspectInfo]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    message: str
    data_sources: Dict[str, int]

# API endpoints
@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with health check and system status"""
    try:
        # Check if data is loaded
        persons_count = len(finder.persons_df) if finder.persons_df is not None else 0
        criminal_count = len(finder.criminal_history_df) if finder.criminal_history_df is not None else 0
        financial_count = len(finder.financial_transactions_df) if finder.financial_transactions_df is not None else 0
        
        return HealthCheck(
            status="healthy",
            message="Criminal Finder API is running successfully",
            data_sources={
                "persons": persons_count,
                "criminal_history": criminal_count,
                "financial_transactions": financial_count
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return await root()

@app.post("/find-suspects", response_model=CriminalQueryResponse)
async def find_suspects(query: CriminalQuery):
    """
    Find suspects based on natural language query
    
    This endpoint processes natural language queries to find potential suspects
    based on crime type, location, and year. It also provides financial risk analysis.
    """
    try:
        result = finder.criminal_finder(query.query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        if "message" in result:
            # No suspects found
            return CriminalQueryResponse(
                query=result["query"],
                crime_type=result.get("crime_type", ""),
                city=result.get("city", ""),
                year_filter=result.get("year_filter"),
                suspect_count=0,
                suspects=[]
            )
        
        # Convert suspects to Pydantic models
        suspects = []
        for suspect_data in result["suspects"]:
            suspect = SuspectInfo(**suspect_data)
            suspects.append(suspect)
        
        return CriminalQueryResponse(
            query=result["query"],
            crime_type=result["crime_type"],
            city=result["city"],
            year_filter=result["year_filter"],
            suspect_count=result["suspect_count"],
            suspects=suspects
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/find-suspects", response_model=CriminalQueryResponse)
async def find_suspects_get(
    query: str = Query(..., description="Natural language query", example="assault in Delhi in 2022")
):
    """
    Find suspects using GET method (alternative to POST)
    
    This endpoint allows you to search for suspects using a GET request with query parameters.
    """
    query_obj = CriminalQuery(query=query)
    return await find_suspects(query_obj)

@app.get("/cities", response_model=List[str])
async def get_available_cities():
    """Get list of available cities in the database"""
    try:
        if finder.persons_df is None:
            raise HTTPException(status_code=500, detail="Data not loaded")
        
        cities = finder.persons_df['city'].unique().tolist()
        return sorted(cities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cities: {str(e)}")

@app.get("/crime-types", response_model=List[str])
async def get_available_crime_types():
    """Get list of available crime types in the database"""
    try:
        if finder.criminal_history_df is None:
            raise HTTPException(status_code=500, detail="Data not loaded")
        
        crime_types = finder.criminal_history_df['incident_type'].unique().tolist()
        return sorted(crime_types)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving crime types: {str(e)}")

@app.get("/statistics", response_model=Dict[str, Any])
async def get_system_statistics():
    """Get system statistics and data overview"""
    try:
        if finder.persons_df is None or finder.criminal_history_df is None:
            raise HTTPException(status_code=500, detail="Data not loaded")
        
        # Basic statistics
        total_persons = len(finder.persons_df)
        total_crimes = len(finder.criminal_history_df)
        total_financial = len(finder.financial_transactions_df) if finder.financial_transactions_df is not None else 0
        
        # Crime type distribution
        crime_distribution = finder.criminal_history_df['incident_type'].value_counts().to_dict()
        
        # City distribution
        city_distribution = finder.persons_df['city'].value_counts().to_dict()
        
        # Year distribution
        year_distribution = finder.criminal_history_df['incident_year'].value_counts().sort_index().to_dict()
        
        return {
            "total_records": {
                "persons": total_persons,
                "criminal_history": total_crimes,
                "financial_transactions": total_financial
            },
            "crime_type_distribution": crime_distribution,
            "city_distribution": city_distribution,
            "year_distribution": year_distribution
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.get("/search-suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial search term", example="assault")
):
    """Get search suggestions based on partial input"""
    try:
        suggestions = []
        
        # Crime type suggestions
        if finder.criminal_history_df is not None:
            crime_types = finder.criminal_history_df['incident_type'].unique()
            for crime in crime_types:
                if q.lower() in crime.lower():
                    suggestions.append({"type": "crime", "value": crime})
        
        # City suggestions
        if finder.persons_df is not None:
            cities = finder.persons_df['city'].unique()
            for city in cities:
                if q.lower() in city.lower():
                    suggestions.append({"type": "city", "value": city})
        
        return {"suggestions": suggestions[:10]}  # Limit to 10 suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving suggestions: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_criminal_finder:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
