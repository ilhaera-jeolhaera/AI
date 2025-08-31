# backend.py
import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 1) .env 로드
load_dotenv()

# 2) 프록시 환경변수 제거 (proxies 인자 충돌 예방)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY"):
    os.environ.pop(k, None)

# 3) 텔레메트리 끄기 (Chroma 경고 최소화)
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"

# 4) OpenAI 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 .env에 없습니다!")

# 5) Chroma 경로/컬렉션 (env > 기본값)
DEFAULT_DB_PATH = "./chroma_db_uiseong_20250831_new"  # 새로 만든 0.5.x DB 폴더
CHROMA_DB_PATH = os.getenv("CHROMADB_PATH", DEFAULT_DB_PATH).strip()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "uiseong_policies").strip()

# 6) 로깅/앱
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uiseong_chatbot")

app = FastAPI(
    title="의성군 정책 검색 API",
    description="ChromaDB 0.5.x + LangChain 기반 정책 질의응답",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 오리진으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7) LangChain + OpenAI + Chroma (0.5.x)
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 전역 상태
vectorstore: Optional[Chroma] = None
qa_chain: Optional[RetrievalQA] = None

def initialize_components():
    """OpenAI 클라이언트 직접 주입 + Chroma 0.5.x 로드 + QA 체인 구성"""
    global vectorstore, qa_chain

    try:
        # OpenAI 1.x 클라이언트 직접 생성 (내부 proxies 인자 문제 예방)
        oa_client = OpenAI(api_key=OPENAI_API_KEY)

        # 임베딩/LLM에 client 주입
        embeddings = OpenAIEmbeddings(
            client=oa_client,
            model="text-embedding-3-small",
        )
        llm = ChatOpenAI(
            client=oa_client,
            model="gpt-3.5-turbo",
            temperature=0.3,
        )

        # Chroma 0.5.x 로드
        if not os.path.exists(CHROMA_DB_PATH):
            logger.warning(f"⚠️ ChromaDB 경로가 없습니다: {CHROMA_DB_PATH}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        # 프롬프트 & RetrievalQA
        prompt_template = """당신은 의성군 정책 전문가입니다.
아래 문맥을 바탕으로 질문에 대해 간결하고 정확한 답변을 작성하세요.

문맥:
{context}

질문: {question}

답변:"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        logger.info("✅ 초기화 성공: Chroma + QA 체인 준비 완료")

    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        vectorstore, qa_chain = None, None

@app.on_event("startup")
async def on_startup():
    initialize_components()

# 8) 스키마
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int

# 9) 엔드포인트
@app.get("/")
async def root():
    return {
        "service": "의성군 정책 검색 API",
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
    out = [{"content": d.page_content, "metadata": getattr(d, "metadata", {})} for d in docs]
    return SearchResponse(documents=out, total_count=len(out))

@app.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="QA 체인이 초기화되지 않았습니다")
    result = qa_chain({"query": req.question})
    answer = result.get("result", "답변을 생성할 수 없습니다.")
    sources = [{"content": d.page_content, "metadata": getattr(d, "metadata", {})} for d in result.get("source_documents", [])]
    return QueryResponse(answer=answer, source_documents=sources)

# 10) 로컬 실행 (reload 경고 없이 import string 사용)
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 의성군 정책 검색 API (Chroma 0.5.x)")
    print("=" * 60)
    print(f"📍 로컬 접속: http://127.0.0.1:8000")
    print(f"📍 문서: http://127.0.0.1:8000/docs")
    print(f"📍 ChromaDB: {CHROMA_DB_PATH} (exists={os.path.exists(CHROMA_DB_PATH)})")
    print("=" * 60)

    # reload=True를 쓰면서 경고 없애려면 import string 방식 사용
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
