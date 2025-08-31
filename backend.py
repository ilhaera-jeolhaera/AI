import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------- LangChain + Chroma (0.5.x) --------------------
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -------------------- 환경 변수 로드 --------------------
load_dotenv()

for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","OPENAI_PROXY"):
    os.environ.pop(k, None)
    
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 .env 파일에 없습니다!")

# -------------------- 로깅 설정 --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uiseong_chatbot")

# -------------------- FastAPI 앱 --------------------
app = FastAPI(
    title="의성군 정책 검색 API",
    description="ChromaDB 0.5.x + LangChain 기반 정책 질의응답",
    version="2.0.0"
)

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- ChromaDB 설정 --------------------
CHROMA_DB_PATH = "./chroma_db_uiseong_100_20250831_new"
COLLECTION_NAME = "uiseong_policies"

# -------------------- Pydantic 모델 --------------------
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int

# -------------------- 글로벌 변수 --------------------
vectorstore = None
qa_chain = None

def initialize_chroma_and_qa():
    """Chroma 0.5.x + QA 체인 초기화"""
    global vectorstore, qa_chain
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # 0.5.x 버전 호환 방식
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # ChatOpenAI 초기화
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            temperature=0.3
        )

        # 프롬프트 템플릿
        prompt_template = """당신은 의성군 정책 전문가입니다.
아래 문맥을 바탕으로 사용자의 질문에 대해 간결하고 정확한 답변을 작성하세요.

문맥:
{context}

질문: {question}

답변:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        logger.info("✅ ChromaDB + QA 체인 초기화 성공")

    except Exception as e:
        logger.error(f"❌ 초기화 실패: {str(e)}")
        vectorstore, qa_chain = None, None

@app.on_event("startup")
async def startup_event():
    initialize_chroma_and_qa()

# -------------------- 엔드포인트 --------------------
@app.get("/")
async def root():
    return {
        "service": "의성군 정책 검색 API",
        "status": "running",
        "db_path": CHROMA_DB_PATH,
        "collection": COLLECTION_NAME,
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if vectorstore else "error",
        "db_exists": os.path.exists(CHROMA_DB_PATH),
        "collection": COLLECTION_NAME,
        "vectorstore_initialized": vectorstore is not None,
        "qa_chain_initialized": qa_chain is not None
    }

@app.get("/search/{query}", response_model=SearchResponse)
async def search_documents(query: str, k: int = 5):
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vectorstore가 초기화되지 않았습니다")

    docs = vectorstore.similarity_search(query, k=k)
    formatted_docs = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    return SearchResponse(documents=formatted_docs, total_count=len(formatted_docs))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="QA 체인이 초기화되지 않았습니다")

    result = qa_chain({"query": request.question})
    answer = result.get("result", "답변을 생성할 수 없습니다.")
    source_docs = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in result.get("source_documents", [])
    ]

    return QueryResponse(answer=answer, source_documents=source_docs)

# -------------------- 실행 --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
