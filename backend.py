import os
import json
import re
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ChromaDB telemetry 억제
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"

def sanitize_api_key(key: str) -> str:
    """API 키 공백/따옴표 제거"""
    if not key:
        return ""
    return key.strip().strip('"').strip("'")

def sanitize_markdown(text: str) -> str:
    """마크다운 기호 제거"""
    if not text:
        return text
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    return text

# .env 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("uiseong_chatbot")

# FastAPI 앱
app = FastAPI(
    title="의성군 정책 챗봇 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경변수
OPENAI_API_KEY = sanitize_api_key(os.getenv("OPENAI_API_KEY", ""))
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db_uiseong_100_20250820_090753").strip()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "uiseong_policies").strip()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# 요청/응답 모델
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

# 전역 변수
qa_chain = None
vectorstore = None
api_available = False

def initialize_system():
    """시스템 초기화 (에러 내성)"""
    global qa_chain, vectorstore, api_available
    
    try:
        if not OPENAI_API_KEY:
            logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음. API 기능 비활성화.")
            return False
            
        # OpenAI 임베딩 - openai_api_key 파라미터 사용 (호환성)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # ChromaDB 로드
        vectorstore = Chroma(
            persist_directory=CHROMADB_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION
        )
        
        # 데이터 확인
        try:
            doc_count = len(vectorstore.get()["ids"])
            logger.info(f"📊 ChromaDB 로드 완료: {doc_count}개 정책")
        except:
            doc_count = 0
            logger.warning("📊 ChromaDB 문서 수 확인 실패")
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.3)
        
        # 프롬프트 템플릿
        prompt_template = """당신은 친근하고 전문적인 의성군 정책 상담사입니다.

정책 정보:
{context}

주민 질문: {question}

다음 형식으로, 마크다운 없이 순수 텍스트로 답변:
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

규칙: 마크다운(**, *, ###) 절대 사용 금지. 순수 텍스트만 출력.
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        # QA 체인
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
        )
        
        api_available = True
        logger.info("✅ 의성군 정책 챗봇 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 초기화 오류: {e}")
        api_available = False
        return False

# 서버 시작 시 초기화
@app.on_event("startup")
async def startup_event():
    initialize_system()

def parse_chatbot_response(answer: str, source_content: str = "") -> dict:
    """GPT 응답을 구조화된 JSON으로 파싱"""
    answer = sanitize_markdown(answer)
    source_content = sanitize_markdown(source_content)
    
    policy_data = {
        "name": "",
        "target": "", 
        "benefits": [],
        "application": {"method": "", "documents": []},
        "notes": []
    }
    
    contact_data = {
        "phone": "054-830-6114",
        "website": "https://www.uisung.go.kr"
    }
    
    for line in answer.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        if "정책명:" in line:
            policy_data["name"] = line.split(":", 1)[1].strip()
        elif "대상:" in line:
            policy_data["target"] = line.split(":", 1)[1].strip()
        elif "지원내용" in line and ":" in line:
            benefit = line.split(":", 1)[1].strip()
            if benefit and benefit not in policy_data["benefits"]:
                policy_data["benefits"].append(benefit)
        elif "신청방법:" in line:
            policy_data["application"]["method"] = line.split(":", 1)[1].strip()
        elif "필요서류" in line and ":" in line:
            doc = line.split(":", 1)[1].strip()
            if doc and doc not in policy_data["application"]["documents"]:
                policy_data["application"]["documents"].append(doc)
        elif "참고사항" in line and ":" in line:
            note = line.split(":", 1)[1].strip()
            if note and note not in policy_data["notes"]:
                policy_data["notes"].append(note)
    
    # 기본값 보완
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
        "source_summary": source_content[:200] + "..." if len(source_content) > 200 else source_content
    }

@app.post("/query", response_model=ChatbotResponse)
async def process_query(request: QueryRequest):
    """정책 질문 처리"""
    if not api_available or qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="챗봇 서비스가 현재 이용할 수 없습니다. OPENAI_API_KEY를 확인해주세요."
        )
    
    try:
        logger.info(f"🔍 질의: {request.query}")
        
        result = qa_chain.invoke({"query": request.query})
        
        raw_source = ""
        if result.get("source_documents") and len(result["source_documents"]) > 0:
            raw_source = result["source_documents"][0].page_content
        
        clean_answer = sanitize_markdown(result["result"])
        clean_source = sanitize_markdown(raw_source)
        
        parsed = parse_chatbot_response(clean_answer, clean_source)
        
        logger.info(f"✅ 응답 완료: {parsed['policy']['name']}")
        return ChatbotResponse(**parsed)
        
    except Exception as e:
        logger.error(f"❌ 처리 오류: {e}")
        raise HTTPException(status_code=500, detail="정책 검색 중 오류가 발생했습니다.")

@app.get("/search/{query}")
async def simple_search(query: str, limit: int = 5):
    """간단한 유사도 검색"""
    try:
        if not vectorstore:
            raise HTTPException(status_code=503, detail="검색 서비스가 초기화되지 않았습니다.")
            
        results = vectorstore.similarity_search(query, k=limit)
        
        policies = []
        for result in results:
            policies.append({
                "name": result.metadata.get("정책명", ""),
                "category": result.metadata.get("카테고리", ""),
                "target": result.metadata.get("대상", ""),
                "benefits": result.metadata.get("지원내용", ""),
            })
        
        return {
            "query": query,
            "total_results": len(policies),
            "policies": policies
        }
        
    except Exception as e:
        logger.error(f"❌ 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """건강 상태 확인 (CloudType Health Check)"""
    try:
        policy_count = 0
        if vectorstore:
            try:
                policy_count = len(vectorstore.get()["ids"])
            except:
                policy_count = 0
        
        return {
            "status": "healthy",
            "model": "gpt-4o-mini",
            "chromadb_path": CHROMADB_PATH,
            "collection": CHROMA_COLLECTION,
            "total_policies": policy_count,
            "api_available": api_available,
            "openai_key_set": bool(OPENAI_API_KEY)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "api_available": False
        }

@app.get("/")
def root():
    """루트 엔드포인트"""
    return {
        "message": "의성군 정책 챗봇 API",
        "version": "1.0.0",
        "model": "gpt-4o-mini",
        "status": "healthy" if api_available else "limited",
        "endpoints": {
            "query": "POST /query - 정책 질문",
            "search": "GET /search/{query} - 유사도 검색",
            "health": "GET /health - 상태 확인",
            "docs": "GET /docs - API 문서"
        }
    }

def test_query(query: str):
    """로컬 테스트용"""
    if not api_available or qa_chain is None:
        return {"error": "시스템이 초기화되지 않았습니다"}
    
    try:
        result = qa_chain.invoke({"query": query})
        source_content = result["source_documents"][0].page_content if result.get("source_documents") else ""
        parsed = parse_chatbot_response(result["result"], source_content)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    print("🏛️ 의성군 정책 챗봇 서버 시작")
    print(f"🤖 모델: GPT-4o mini")
    print(f"📊 ChromaDB: {CHROMADB_PATH}")
    print(f"🌐 서버: http://{HOST}:{PORT}")
    print(f"📚 문서: http://{HOST}:{PORT}/docs")
    print(f"💊 헬스체크: http://{HOST}:{PORT}/health")
    
    # 초기화
    if initialize_system():
        print("✅ 시스템 초기화 완료")
        print("\n🧪 테스트:")
        print(test_query("청년 주거 지원"))
    else:
        print("⚠️ 제한된 모드로 시작 (API 키 확인 필요)")
    
    uvicorn.run(app, host=HOST, port=PORT, reload=False)