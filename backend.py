import os
import re
import textwrap
import asyncio
import pandas as pd
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
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

# ---------- Readability Utilities ----------
def sanitize_markdown(text: str) -> str:
    """마크다운 기호 제거 및 기본 정리"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    # 굵게/기울임/코드블록/헤더/인용문 등 기호 제거
    text = re.sub(r'[*_`#>]{1,}', '', text)
    # 불릿 기호 통일
    text = re.sub(r'^\s*[-•]\s*', '- ', text, flags=re.MULTILINE)
    # 번호 리스트 "1. " → "- "
    text = re.sub(r'^\s*\d+\.\s*', '- ', text, flags=re.MULTILINE)
    # 여분 공백 정리
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def normalize_linebreaks(text: str, wrap_width: int = 90) -> str:
    """문장/항목 단위 줄바꿈 및 적당한 폭으로 래핑"""
    if not text:
        return ""
    s = text

    # 라벨 뒤에는 줄바꿈 보장
    s = re.sub(r'(정책명|대상|지원내용|신청방법|필요서류|참고사항)\s*:\s*', r'\n\1: ', s)

    # 종결 어미 뒤에는 줄바꿈 힌트
    s = re.sub(r'(다\.|요\.|함\.|임\.)\s+', r'\1\n', s)

    # 불필요한 불릿 제거
    s = re.sub(r'^\s*[-•]\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'^\s*\d+\.\s*', '', s, flags=re.MULTILINE)

    # 연속 개행 정리
    s = re.sub(r'\n{3,}', '\n\n', s).strip()

    # 단락 래핑
    wrapped_lines = []
    for para in s.split('\n'):
        wrapped_lines.append(textwrap.fill(para, width=wrap_width))
    return '\n'.join(wrapped_lines)


def format_answer_for_readability(raw: str) -> str:
    """마크다운 제거 → 줄바꿈 정리 → 래핑"""
    return normalize_linebreaks(sanitize_markdown(raw))

# ---------- Lifespan ----------
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

# ---------- FastAPI ----------
app = FastAPI(
    title="RAG Server",
    description="A FastAPI-based RAG server combining ChromaDB vector search with OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - 환경변수에서 슬래시/공백 정리하여 완전일치 비교
origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://localhost:5174,https://uiseong-hwnsngs-projects.vercel.app,https://uiseong.vercel.app"
).split(",")

allowed_origins = [o.strip().rstrip("/") for o in origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,                     # 쿠키/인증 사용 시 true
    allow_methods=["GET", "POST", "OPTIONS"],   # 필요 시 ["*"]
    allow_headers=["*"],
)

# ---------- Models ----------
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

# ---------- Globals ----------
chroma_client = None
collection = None
openai_client = None

# ---------- OpenAI ----------
def initialize_openai():
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not found")
        raise ValueError("OpenAI API key not configured")
    openai_client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")

# ---------- ChromaDB ----------
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

            for idx, row in df.iterrows():
                # 문서 텍스트 생성
                doc_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                # 임베딩 생성
                embedding = await generate_embedding(doc_text)
                # 컬렉션 추가
                collection.add(
                    documents=[doc_text],
                    embeddings=[embedding],
                    metadatas=[{"source": os.path.basename(csv_file), "row_id": idx}],
                    ids=[f"{os.path.basename(csv_file)}_{idx}"]
                )

        logger.info(f"Data loading completed. Collection now has {collection.count()} documents")

    except Exception as e:
        logger.error(f"Failed to load initial data: {str(e)}")

async def initialize_chromadb():
    global chroma_client, collection
    try:
        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            persist_directory="./chroma_db_uiseong_20250831_new"
        )

        chroma_client = chromadb.PersistentClient(
            path="./chroma_db_uiseong_20250831_new",
            settings=chroma_settings
        )

        try:
            collection = chroma_client.get_collection(name="uiseong_policies")
            count = collection.count()
            if count == 0:
                logger.info("Collection exists but is empty, loading data...")
                await load_initial_data()
            else:
                logger.info(f"ChromaDB collection 'uiseong_policies' loaded successfully with {count} documents")
        except NotFoundError:
            logger.info("Collection 'uiseong_policies' not found, creating new collection...")
            collection = chroma_client.create_collection(name="uiseong_policies")
            logger.info("ChromaDB collection 'uiseong_policies' created successfully")
            await load_initial_data()

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        raise e

# ---------- Embedding & Answer ----------
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

async def generate_answer(question: str, context: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        system_prompt = (
            "당신은 한국어로 간결하고 읽기 쉬운 답변을 작성하는 어시스턴트입니다. "
            "반드시 순수 텍스트만 사용하세요(마크다운 금지). "
            "가능하면 다음 라벨을 활용해 주세요: 정책명:, 대상:, 지원내용:, 신청방법:, 필요서류:, 참고사항:. "
            "각 항목은 줄바꿈으로 분리하고, 목록은 '- ' 불릿으로 표시하세요. "
            "주어진 문서에서 확인되지 않는 내용은 추측하지 말고 모른다고 답변하세요."
        )
        user_prompt = (
            f"질문:\n{question}\n\n"
            f"참고 문서:\n{context}\n\n"
            "위 내용을 바탕으로 사실만 답변하세요."
        )

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        raw = response.choices[0].message.content or "죄송합니다. 답변을 생성할 수 없었습니다."
        return format_answer_for_readability(raw)

    except Exception as e:
        logger.error(f"Failed to generate answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

# ---------- Endpoints ----------
@app.get("/", response_model=ServiceInfo)
async def root():
    return ServiceInfo(
        name="RAG Server",
        description="A FastAPI-based RAG server combining ChromaDB vector search with OpenAI for intelligent document retrieval and question answering",
        version="1.0.0",
        endpoints=["/", "/health", "/healthz", "/search/{query}", "/query"]
    )

@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    services = {}

    try:
        services["openai"] = "healthy" if openai_client else "not_initialized"
    except Exception:
        services["openai"] = "unhealthy"

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

@app.get("/search/{query}", response_model=SearchResponse)
async def search_documents(
    query: str,
    n_results: int = Query(default=10, description="Number of results to return", ge=1, le=100)
):
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")

    try:
        query_embedding = await generate_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                })

        # 검색 결과 문서 본문도 최소한의 가독 정리(옵션)
        for item in formatted_results:
            item["document"] = format_answer_for_readability(item["document"])

        return SearchResponse(
            query=query,
            results=formatted_results,
            total_results=len(formatted_results)
        )

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized")

    try:
        question_embedding = await generate_embedding(request.question)
        search_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=request.max_results,
            include=["documents", "metadatas", "distances"]
        )

        context_parts = []
        if search_results['documents'] and len(search_results['documents']) > 0:
            for i in range(len(search_results['documents'][0])):
                document = search_results['documents'][0][i]
                # 컨텍스트도 최소한의 정리
                context_parts.append(f"문서 {i+1}: {sanitize_markdown(document)}")

        context = "\n\n".join(context_parts)

        answer = await generate_answer(request.question, context)
        # 안전하게 한 번 더 포맷(중복 호출 무해)
        answer = format_answer_for_readability(answer)

        return QueryResponse(
            question=request.question,
            answer=answer
        )

    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
