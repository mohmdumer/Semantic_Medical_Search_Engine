# medical_search_backend.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import traceback

# Import your search engine
try:
    from search import SemanticMedicalSearcher
except ImportError as e:
    print(f"Warning: Could not import SemanticMedicalSearcher: {e}")
    print("Make sure your search.py file is in the same directory")
    SemanticMedicalSearcher = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Medical Semantic Search API",
    description="AI-powered semantic search for medical information using BioBERT embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global search engine instance
search_engine = None

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")

class SearchResult(BaseModel):
    score: float = Field(..., description="Relevance score")
    text: str = Field(..., description="Result text")
    rank: int = Field(..., description="Result rank")

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Initialize search engine
@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    global search_engine
    
    if SemanticMedicalSearcher is None:
        logger.error("SemanticMedicalSearcher not available - check your imports")
        return
    
    try:
        logger.info("Initializing Medical Semantic Search Engine...")
        search_engine = SemanticMedicalSearcher()
        logger.info("✅ Medical Semantic Search Engine initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize search engine: {str(e)}")
        logger.error(traceback.format_exc())

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if search_engine is not None else "unhealthy",
        message="Medical Semantic Search Engine is running" if search_engine else "Search engine not initialized",
        model_loaded=search_engine is not None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# Main search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_medical_qa(request: SearchRequest):
    """
    Perform semantic search on medical Q&A database
    
    - **query**: Your medical question or search terms
    - **top_k**: Number of results to return (1-50)
    """
    if search_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Search engine not initialized. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        # Perform the search
        logger.info(f"Searching for: '{request.query}' (top_k={request.top_k})")
        
        # Your search engine returns [(text, score), ...]
        raw_results = search_engine.search(request.query, top_k=request.top_k)
        
        # Format results
        formatted_results = []
        for rank, (text, score) in enumerate(raw_results, 1):
            formatted_results.append(SearchResult(
                score=round(float(score), 4),
                text=str(text),
                rank=rank
            ))
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Search completed in {search_time:.2f}ms - Found {len(formatted_results)} results")
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time, 2),
            model_info={
                "model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                "embedding_dim": 768,
                "search_type": "semantic_similarity"
            }
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

# Get search engine status
@app.get("/status")
async def get_status():
    """Get detailed status of the search engine"""
    if search_engine is None:
        return {
            "status": "error",
            "message": "Search engine not initialized",
            "model_loaded": False
        }
    
    try:
        # Try a test search to verify everything works
        test_results = search_engine.search("test query", top_k=1)
        
        return {
            "status": "ready",
            "message": "Search engine is ready and functional",
            "model_loaded": True,
            "model_info": {
                "model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                "embedding_dimension": 768
            },
            "test_search_successful": len(test_results) >= 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search engine error: {str(e)}",
            "model_loaded": True,
            "error_details": str(e)
        }

# Alternative GET endpoint for simple searches
@app.get("/search_simple")
async def search_simple(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Number of results")
):
    """Simple GET endpoint for quick searches"""
    request = SearchRequest(query=q, top_k=limit)
    return await search_medical_qa(request)

# Get model information
@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    return {
        "model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "model_type": "BioBERT",
        "embedding_dimension": 768,
        "framework": "sentence-transformers",
        "search_backend": "FAISS",
        "normalized_embeddings": True
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print("🏥 Starting Medical Semantic Search API...")
    print("📡 Server will be available at: http://127.0.0.1:8000")
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("🔧 Alternative docs: http://127.0.0.1:8000/redoc")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )