import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# 텔레메트리/프록시 비활성화 (CloudType 환경 대응)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY", "NO_PROXY", "no_proxy"):
    os.environ.pop(k, None)
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"
# CloudType 환경에서 프록시 관련 추가 제거
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 고정 경로/컬렉션
CHROMA_DB_PATH = "./chroma_db_uiseong_20250831_new"
COLLECTION_NAME = "uiseong_policies"

# FastAPI 앱 초기화
app = FastAPI(title="RAG API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
openai_client: Optional[OpenAI] = None
chroma_client = None
collection = None

class QueryRequest(BaseModel):
    question: str
    k: int = 3

class SearchDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    distance: float

class SearchResponse(BaseModel):
    documents: List[SearchDocument]
    total_count: int

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    global openai_client, chroma_client, collection
    
    # OpenAI 클라이언트 초기화 (CloudType 환경 대응)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    try:
        # 프록시 관련 인자를 명시적으로 제거하여 초기화
        openai_client = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=3
        )
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        # 더 단순한 방식으로 재시도
        try:
            import openai as openai_module
            openai_client = openai_module.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized with fallback method")
        except Exception as e2:
            logger.error(f"Fallback OpenAI initialization also failed: {e2}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    # ChromaDB 클라이언트 초기화
    try:
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=False)
        )
        collection = chroma_client.get_collection(COLLECTION_NAME)
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded from '{CHROMA_DB_PATH}'")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

def get_embedding(text: str) -> List[float]:
    """OpenAI 임베딩 생성"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding: {e}")

@app.get("/")
async def root():
    return {
        "service": "RAG API",
        "version": "1.0.0",
        "chroma_path": CHROMA_DB_PATH,
        "collection_name": COLLECTION_NAME
    }

@app.get("/health")
@app.get("/healthz")
async def health_check():
    try:
        db_exists = os.path.exists(CHROMA_DB_PATH)
        document_count = None
        
        if collection is not None:
            try:
                document_count = collection.count()
            except Exception as e:
                logger.warning(f"Could not get document count: {e}")
        
        return {
            "status": "healthy",
            "db_exists": db_exists,
            "collection_name": COLLECTION_NAME,
            "document_count": document_count,
            "initialized": collection is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@app.get("/search/{query}", response_model=SearchResponse)
async def search_documents(query: str, k: int = 5):
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")
    
    try:
        # 쿼리 임베딩 생성
        query_embedding = get_embedding(query)
        
        # ChromaDB에서 유사 문서 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                documents.append(SearchDocument(
                    content=doc,
                    metadata=results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    distance=results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                ))
        
        return SearchResponse(
            documents=documents,
            total_count=len(documents)
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")
    
    try:
        # 1. 쿼리 임베딩 생성
        query_embedding = get_embedding(request.question)
        
        # 2. ChromaDB에서 상위 k개 문서 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.k,
            include=['metadatas', 'documents', 'distances']
        )
        
        # 3. 컨텍스트 구성
        context_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                context_docs.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                })
        
        if not context_docs:
            return QueryResponse(
                answer="죄송합니다. 관련 문서를 찾을 수 없습니다.",
                source_documents=[]
            )
        
        # 4. 컨텍스트 텍스트 생성
        context_text = "\n\n".join([doc["content"] for doc in context_docs])
        
        # 5. OpenAI Chat Completions로 답변 생성
        system_message = """당신은 의성군 정책에 대한 질문에 답변하는 AI 어시스턴트입니다. 
주어진 문서들을 바탕으로 정확하고 도움이 되는 답변을 제공하세요.
문서에 없는 내용은 추측하지 말고, 모르겠다고 답변하세요."""
        
        user_message = f"""다음 문서들을 참고하여 질문에 답변해주세요:

문서들:
{context_text}

질문: {request.question}"""
        
        chat_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = chat_response.choices[0].message.content
        
        return QueryResponse(
            answer=answer,
            source_documents=context_docs
        )
    
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)