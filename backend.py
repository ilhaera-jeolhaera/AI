import os
import json
import re
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 레거시 스택용 임포트 (Chroma 0.4.x + langchain 0.0.x 계열)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------- 초기 설정 --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uiseong_chatbot")

def sanitize_markdown(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    return text

app = FastAPI(title="의성군 정책 챗봇", version="1.0.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------- 환경 변수 --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 비어 있습니다. .env 또는 환경변수로 설정하세요.")

# 로컬용: 폴더 경로는 프로젝트 내 상대경로 그대로 사용
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db_uiseong_100_20250820_090753").strip()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "uiseong_policies").strip()

# -------------------- 모델 --------------------
class QueryRequest(BaseModel):
    query: str

class PolicyResponse(BaseModel):
    name: str
    target: str
    benefits: list
    application: dict
    notes: list

class ContactInfo(BaseModel):
    phone: str
    website: str

class ChatbotResponse(BaseModel):
    policy: PolicyResponse
    contact: ContactInfo
    source_summary: str

# -------------------- Chroma + LLM 초기화 --------------------
qa_chain = None
vectorstore = None
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION
    )
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3
    )

    prompt_template = """당신은 친근하고 전문적인 의성군 정책 상담사입니다.

정책 정보:
{context}

주민 질문: {question}

다음 형식(마크다운 금지, 순수 텍스트)으로 답변:
정책명: ...
대상: ...
지원내용1: ...
지원내용2: ...
지원내용3: ...
신청방법: ...
필요서류1: ...
필요서류2: ...
필요서류3: ...
참고사항1: ...
참고사항2: 외국인 자격은 의성군청에 직접 문의해서 확인받으세요
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
    )

    logger.info("✅ 의성군 정책 챗봇 초기화 완료 (로컬 / 레거시 스택, Chroma 0.4.x)")
except Exception as e:
    logger.error("❌ 초기화 오류: %s", e)
    qa_chain = None

# -------------------- 응답 파서 --------------------
def parse_chatbot_response(answer: str, source_content: str = "") -> dict:
    answer = sanitize_markdown(answer)
    source_content = sanitize_markdown(source_content)

    policy_data = {
        "name": "", "target": "", "benefits": [],
        "application": {"method": "", "documents": []}, "notes": []
    }
    contact_data = {"phone": "054-830-6114", "website": "https://www.uisung.go.kr"}

    for line in answer.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "정책명:" in line: policy_data["name"] = line.split(":", 1)[1].strip()
        elif "대상:" in line: policy_data["target"] = line.split(":", 1)[1].strip()
        elif "지원내용" in line and ":" in line:
            b = line.split(":", 1)[1].strip()
            if b and b not in policy_data["benefits"]:
                policy_data["benefits"].append(b)
        elif "신청방법:" in line: policy_data["application"]["method"] = line.split(":", 1)[1].strip()
        elif "필요서류" in line and ":" in line:
            d = line.split(":", 1)[1].strip()
            if d and d not in policy_data["application"]["documents"]:
                policy_data["application"]["documents"].append(d)
        elif "참고사항" in line and ":" in line:
            n = line.split(":", 1)[1].strip()
            if n and n not in policy_data["notes"]:
                policy_data["notes"].append(n)

    if not policy_data["benefits"]:
        policy_data["benefits"] = ["정책 혜택 정보는 의성군청에 문의하세요"]
    if not policy_data["application"]["documents"]:
        policy_data["application"]["documents"] = ["신청서", "주민등록등본"]
    if not policy_data["notes"]:
        policy_data["notes"] = ["외국인 자격은 의성군청 확인 필요"]
    elif not any("외국인" in n for n in policy_data["notes"]):
        policy_data["notes"].append("외국인 자격은 의성군청 확인 필요")

    return {
        "policy": policy_data,
        "contact": contact_data,
        "source_summary": (source_content[:200] + "...") if len(source_content) > 200 else source_content,
    }

# -------------------- 엔드포인트 --------------------
@app.post("/query", response_model=ChatbotResponse)
async def process_query(request: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="시스템이 초기화되지 않았습니다")
    try:
        result = qa_chain.invoke({"query": request.query})
        raw_source = ""
        if result.get("source_documents"):
            raw_source = result["source_documents"][0].page_content
        parsed = parse_chatbot_response(result["result"], raw_source)
        return ChatbotResponse(**parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{query}")
async def simple_search(query: str, limit: int = 5):
    try:
        results = vectorstore.similarity_search(query, k=limit)
        policies = [{
            "name": r.metadata.get("정책명", ""),
            "category": r.metadata.get("카테고리", ""),
            "target": r.metadata.get("대상", ""),
            "benefits": r.metadata.get("지원내용", ""),
        } for r in results]
        return {"query": query, "total_results": len(policies), "policies": policies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    try:
        count = len(vectorstore.get()["ids"]) if vectorstore else 0
        return {
            "status": "healthy",
            "model": "gpt-3.5-turbo",
            "chromadb_path": CHROMADB_PATH,
            "collection": CHROMA_COLLECTION,
            "total_policies": count,
            "api_available": qa_chain is not None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/")
def root():
    return {
        "message": "의성군 정책 챗봇 API",
        "version": "1.0.0",
        "endpoints": {"query": "POST /query", "search": "GET /search/{query}", "health": "GET /health"}
    }

# -------------------- 실행부 (로컬 고정) --------------------
if __name__ == "__main__":
    import uvicorn
    HOST = "127.0.0.1"   # 로컬에서만 접근
    PORT = 8000          # 고정 포트
    logger.info(f"🚀 Starting Uvicorn on {HOST}:{PORT} (LOCAL)")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
