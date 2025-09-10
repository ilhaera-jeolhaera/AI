import os
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import chromadb
from fastapi.middleware.cors import CORSMiddleware
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from openai import OpenAI
import logging

# Disable telemetry and configure ChromaDB settings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG server...")
    initialize_openai()
    await initialize_chromadb()
    logger.info("RAG server startup completed")
    yield
    # Shutdown
    logger.info("RAG server shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Server",
    description="A FastAPI-based RAG server combining ChromaDB vector search with OpenAI",
    version="1.0.0",
    lifespan=lifespan
)
# 배포/개발 도메인을 환경변수로 관리 (쉼표로 구분)
# 예: CORS_ALLOW_ORIGINS="http://localhost:5173,http://localhost:5174,https://your-frontend.com"
origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://localhost:5174, https://uiseong-hwnsngs-projects.vercel.app/ ,https://uiseong.vercel.app/"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,                     # 쿠키/인증 사용 시 true (axios withCredentials 필요)
    allow_methods=["GET", "POST", "OPTIONS"],   # 필요 시 ["*"]
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    max_results: int = Field(default=5, description="Maximum number of search results to use for context")

class QueryResponse(BaseModel):
    question: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]

class ServiceInfo(BaseModel):
    name: str
    description: str
    version: str
    endpoints: List[str]

# Global variables for services
chroma_client = None
collection = None
openai_client = None

# Initialize OpenAI client
def initialize_openai():
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not found")
        raise ValueError("OpenAI API key not configured")
    
    openai_client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")

# Load initial data from CSV
async def load_initial_data():
    """Load data from CSV files into ChromaDB collection"""
    try:
        import glob
        csv_files = glob.glob("./attached_assets/*.csv")
        
        if not csv_files:
            logger.warning("No CSV files found in attached_assets directory")
            return
        
        for csv_file in csv_files:
            logger.info(f"Loading data from {csv_file}")
            df = pd.read_csv(csv_file)
            
            # Process each row
            for idx, row in df.iterrows():
                # Create document text from all columns
                doc_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                
                # Generate embedding
                embedding = await generate_embedding(doc_text)
                
                # Add to collection
                collection.add(
                    documents=[doc_text],
                    embeddings=[embedding],
                    metadatas=[{"source": os.path.basename(csv_file), "row_id": idx}],
                    ids=[f"{os.path.basename(csv_file)}_{idx}"]
                )
        
        logger.info(f"Data loading completed. Collection now has {collection.count()} documents")
        
    except Exception as e:
        logger.error(f"Failed to load initial data: {str(e)}")

# Initialize ChromaDB client and collection
async def initialize_chromadb():
    global chroma_client, collection
    
    try:
        # ChromaDB settings with telemetry disabled
        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            persist_directory="./chroma_db_uiseong_20250831_new"
        )
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db_uiseong_20250831_new",
            settings=chroma_settings
        )
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name="uiseong_policies")
            # Check if collection has data
            count = collection.count()
            if count == 0:
                logger.info("Collection exists but is empty, loading data...")
                await load_initial_data()
            else:
                logger.info(f"ChromaDB collection 'uiseong_policies' loaded successfully with {count} documents")
        except NotFoundError:
            logger.info("Collection 'uiseong_policies' not found, creating new collection...")
            collection = chroma_client.create_collection(name="uiseong_policies")
            logger.info(f"ChromaDB collection 'uiseong_policies' created successfully")
            # Load initial data
            await load_initial_data()
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        raise e

# Generate embeddings using OpenAI
async def generate_embedding(text: str) -> List[float]:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# Generate answer using OpenAI GPT
async def generate_answer(question: str, context: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        system_prompt = """당신은 도움이 되는 AI 어시스턴트입니다. 주어진 문서들의 내용을 바탕으로 질문에 정확하고 상세하게 답변해주세요. 
        답변은 한국어로 제공하며, 제공된 문서에서 찾을 수 없는 정보는 추측하지 말고 모른다고 말해주세요."""
        
        user_prompt = f"""
        질문: {question}
        
        참고 문서:
        {context}
        
        위 문서들을 참고하여 질문에 대한 답변을 제공해주세요.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        if answer is None:
            return "죄송합니다. 답변을 생성할 수 없었습니다."
        return answer
        
    except Exception as e:
        logger.error(f"Failed to generate answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")


# Root endpoint - service information
@app.get("/", response_model=ServiceInfo)
async def root():
    return ServiceInfo(
        name="RAG Server",
        description="A FastAPI-based RAG server combining ChromaDB vector search with OpenAI for intelligent document retrieval and question answering",
        version="1.0.0",
        endpoints=["/", "/health", "/healthz", "/search/{query}", "/query"]
    )

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    services = {}
    
    # Check OpenAI connection
    try:
        if openai_client:
            services["openai"] = "healthy"
        else:
            services["openai"] = "not_initialized"
    except Exception:
        services["openai"] = "unhealthy"
    
    # Check ChromaDB connection
    try:
        if chroma_client and collection:
            collection_count = collection.count()
            services["chromadb"] = f"healthy ({collection_count} documents)"
        else:
            services["chromadb"] = "not_initialized"
    except Exception:
        services["chromadb"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(status.startswith("healthy") for status in services.values()) else "degraded",
        version="1.0.0",
        services=services
    )

# Search endpoint
@app.get("/search/{query}", response_model=SearchResponse)
async def search_documents(
    query: str,
    n_results: int = Query(default=10, description="Number of results to return", ge=1, le=100)
):
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")
    
    try:
        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                })
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total_results=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")

# RAG Query endpoint
@app.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")
    
    try:
        # Generate embedding for the question
        question_embedding = await generate_embedding(request.question)
        
        # Search for relevant documents
        search_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=request.max_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Prepare context from search results
        context_parts = []
        
        if search_results['documents'] and len(search_results['documents']) > 0:
            for i in range(len(search_results['documents'][0])):
                document = search_results['documents'][0][i]
                context_parts.append(f"문서 {i+1}: {document}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using OpenAI
        answer = await generate_answer(request.question, context)
        
        return QueryResponse(
            question=request.question,
            answer=answer
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Run the server
if __name__ == "__main__":
    # Configure uvicorn to run on 0.0.0.0:8000 for cloud deployment
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )