# backend.py
import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── 로깅 ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uiseong_chatbot")

# ── 프록시/텔레메트리 차단(로컬 깔끔 실행용) ─────────────────────
for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","OPENAI_PROXY"):
    os.environ.pop(k, None)
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"

# ── 라이브러리 임포트 (LangChain + OpenAI + Chroma 0.5.x) ────────
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── OpenAI 키 (로컬에서만: .env 없이 환경변수나 OS keyring 쓰지 않고 코드에서 직접 읽게 하려면 아래를 채우세요) ──
# 1) 일반적으로는 환경변수로 설정:  export OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 환경변수가 비어 있습니다. 로컬에서 실행 전 설정해주세요.")

# ── ChromaDB 0.5.x 로컬 경로(하드코딩) ───────────────────────────
# 새로 생성한 0.5.x DB 폴더 경로로 바꿔 쓰세요.
CHROMA_DB_PATH = "./chroma_db_uiseong_20250831_new"
COLLECTION_NAME = "uiseong_policies"

# ── FastAPI 앱 ───────────────────────────────────────────────────
app = FastAPI(
    title="의성군 정책 검색 API",
    description="ChromaDB 0.5.x + LangChain(OpenAI 1.x) 기반 / 로컬(127.0.0.1) 전용",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── 전역 상태 ────────────────────────────────────────────────────
vectorstore: Optional[Chroma] = None
qa_chain: Optional[RetrievalQA] = None

def initialize_components():
    """OpenAI(1.x) client 주입 + Chroma 0.5.x 로드 + QA 체인 구성"""
    global vectorstore, qa_chain
    try:
        # OpenAI 1.x 클라이언트 생성 (proxies 문제 회피)
        oa_client = OpenAI(api_key=OPENAI_API_KEY)

        # 임베딩/LLM에 client 직접 주입 (중요!)
        embeddings = OpenAIEmbeddings(
            client=oa_client,
            model="text-embedding-3-small",
        )
        llm = ChatOpenAI(
            client=oa_client,
            model="gpt-3.5-turbo",
            temperature=0.3,
        )

        # Chroma 0.5.x 로드 (폴더가 반드시 0.5.x로 생성된 DB여야 함)
        if not os.path.exists(CHROMA_DB_PATH):
            logger.warning(f"⚠️ ChromaDB 경로 없음: {CHROMA_DB_PATH}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        # 프롬프트 & QA 체인
        prompt_template = """당신은 의성군 정책 전문가입니다.
아래 문맥을 바탕으로 질문에 대해 간결하고 정확한 답변을 작성하세요.

문맥:
{context}

질문: {question}

답변:"""
        prompt = PromptTemplate(
            input_variables=["context","question"],
            template=prompt_template,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        logger.info("✅ 초기화 성공: Chroma(0.5.x) + QA 체인 준비 완료")
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        vectorstore = None
        qa_chain = None

@app.on_event("startup")
async def on_startup():
    initialize_components()

# ── 스키마 ───────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int

# ── 엔드포인트 ───────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "의성군 정책 검색 API (로컬용)",
        "status": "running",
        "db_path": CHROMA_DB_PATH,
        "collection": COLLECTION_NAME,
        "docs": "/docs",
    }

@app.get("/health")
@app.get("/healthz")
async def health():
    info = {
        "status": "healthy" if vectorstore else "error",
        "db_exists": os.path.exists(CHROMA_DB_PATH),
        "collection": COLLECTION_NAME,
        "vectorstore_initialized": vectorstore is not None,
        "qa_chain_initialized": qa_chain is not None,
    }
    if vectorstore is not None:
        try:
            info["document_count"] = vectorstore._collection.count()
        except Exception as e:
            info["vectorstore_error"] = str(e)
    return info

@app.get("/search/{query}", response_model=SearchResponse)
async def search(query: str, k: int = 5):
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vectorstore가 초기화되지 않았습니다")
    docs = vectorstore.similarity_search(query, k=k)
    out = [{"content": d.page_content, "metadata": getattr(d,"metadata",{})} for d in docs]
    return SearchResponse(documents=out, total_count=len(out))

@app.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="QA 체인이 초기화되지 않았습니다")
    result = qa_chain({"query": req.question})
    answer = result.get("result", "답변을 생성할 수 없습니다.")
    sources = [{"content": d.page_content, "metadata": getattr(d,"metadata",{})} for d in result.get("source_documents", [])]
    return QueryResponse(answer=answer, source_documents=sources)

# ── 로컬 전용 실행(127.0.0.1 고정) ──────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 의성군 정책 검색 API 서버(로컬) 시작")
    print(f"📍 http://127.0.0.1:8000   📚 /docs")
    print(f"📁 ChromaDB: {CHROMA_DB_PATH}  (exists={os.path.exists(CHROMA_DB_PATH)})")
    print("=" * 60)
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)