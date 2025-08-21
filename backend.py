import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

def sanitize_markdown(text: str) -> str:
    """마크다운 기호 제거 함수"""
    if not text:
        return text
    # 굵게/기울임 별표 제거
    text = re.sub(r'\*{1,3}', '', text)
    # 헤더/리스트 등 기타 흔한 마크다운 기호 최소 제거
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)  # ### 헤더 제거
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE) # 불릿 기호 제거
    return text

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="의성군 정책 챗봇 로컬 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY를 .env 파일에 설정하거나 환경변수로 설정해주세요")

# ChromaDB 경로 (.env 파일에서 설정 가능)
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db_uiseong_100_20250820_090753")

# 서버 설정 (.env 파일에서 설정 가능)
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# 요청 모델
class QueryRequest(BaseModel):
    query: str

# 응답 모델 - 사용자가 요청한 JSON 구조
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

# OpenAI 및 ChromaDB 초기화
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=CHROMADB_PATH, embedding_function=embeddings)
    
    # GPT-4o mini 모델 사용
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)
    
    # 프롬프트 템플릿 - 대화형으로 수정
    prompt_template = """당신은 친근하고 전문적인 의성군 정책 상담사입니다. 주민들이 쉽게 이해할 수 있도록 정책을 친절하게 설명해주세요.

정책 정보:
{context}

주민 질문: {question}

다음과 같은 친근한 대화 형식으로 답변해주세요:

정책명: [정책명을 자연스럽게 소개]
대상: [누가 신청할 수 있는지 친근하게 설명하고, 외국인 가능성도 언급]
지원내용1: [첫 번째 혜택을 구체적이고 이해하기 쉽게 설명]
지원내용2: [두 번째 혜택이 있다면 추가 설명]
지원내용3: [세 번째 혜택이 있다면 추가 설명]
신청방법: [어디서 어떻게 신청하면 되는지 단계별로 친절하게 안내]
필요서류1: [준비해야 할 첫 번째 서류]
필요서류2: [준비해야 할 두 번째 서류]
필요서류3: [준비해야 할 세 번째 서류]
참고사항1: [꼭 알아야 할 주의사항이나 팁]
참고사항2: 외국인 자격은 의성군청에 직접 문의해서 확인받으세요

마치 옆에서 친절하게 설명해주는 것처럼 따뜻하고 이해하기 쉽게 답변해주세요.

규칙:
- 마크다운을 사용하지 마세요. 특히 **, *, ###, #### 등의 기호를 절대 출력하지 마세요.
- 답변은 순수 텍스트만 사용하세요. 굵게/기울임/헤더/코드블록 등 금지.
- 형식 라벨(정책명:, 대상:, 지원내용1:, 신청방법:, 필요서류1:, 참고사항1:)은 그대로 출력하되, 어떠한 마크다운 기호도 포함하지 마세요.
"""
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    
    logger.info("✅ 의성군 정책 챗봇이 성공적으로 초기화되었습니다.")
    
except Exception as e:
    logger.error(f"❌ 초기화 오류: {e}")
    qa_chain = None

def parse_chatbot_response(answer: str, source_content: str = "") -> dict:
    """사용자 요청 JSON 구조로 응답 파싱"""
    
    # ✅ 마크다운 기호 제거
    answer = sanitize_markdown(answer)
    source_content = sanitize_markdown(source_content)
    
    # 기본값 설정
    policy_data = {
        "name": "",
        "target": "",
        "benefits": [],
        "application": {
            "method": "",
            "documents": []
        },
        "notes": []
    }
    
    contact_data = {
        "phone": "054-830-6114",
        "website": "https://www.uisung.go.kr"
    }
    
    lines = sanitize_markdown(answer).split("\n")
    
    for line in lines:
        line = sanitize_markdown(line).strip()
        if not line:
            continue
            
        # 정책명 추출
        if "정책명:" in line:
            policy_data["name"] = line.split(":", 1)[1].strip()
        
        # 대상 추출
        elif "대상:" in line:
            policy_data["target"] = line.split(":", 1)[1].strip()
        
        # 지원내용 추출 (배열로)
        elif "지원내용" in line and ":" in line:
            benefit = line.split(":", 1)[1].strip()
            if benefit and benefit not in policy_data["benefits"]:
                policy_data["benefits"].append(benefit)
        
        # 신청방법 추출
        elif "신청방법:" in line:
            policy_data["application"]["method"] = line.split(":", 1)[1].strip()
        
        # 필요서류 추출 (배열로)
        elif "필요서류" in line and ":" in line:
            document = line.split(":", 1)[1].strip()
            if document and document not in policy_data["application"]["documents"]:
                policy_data["application"]["documents"].append(document)
        
        # 참고사항 추출 (배열로)
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
    elif not any("외국인" in note for note in policy_data["notes"]):
        policy_data["notes"].append("외국인 자격은 의성군청 확인 필요")
    
    return {
        "policy": policy_data,
        "contact": contact_data,
        "source_summary": source_content[:200] + "..." if len(source_content) > 200 else source_content
    }

@app.post("/query", response_model=ChatbotResponse)
async def process_query(request: QueryRequest):
    """정책 질문 처리 - 요청된 JSON 구조로 답변"""
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="시스템이 초기화되지 않았습니다")
        
        logger.info(f"🔍 검색 질의: {request.query}")
        
        # 1. Chroma 벡터DB에서 유사 정책 검색
        response = qa_chain.invoke({"query": request.query})
        
        # 2. 검색된 소스 문서 추출
        raw_source = ""
        if response.get('source_documents') and len(response['source_documents']) > 0:
            raw_source = response['source_documents'][0].page_content
        
        # ✅ 후처리 적용 (마크다운 기호 제거)
        clean_answer = sanitize_markdown(response['result'])
        clean_source = sanitize_markdown(raw_source)
        
        # 3. GPT-4o mini 답변을 요청된 JSON 구조로 파싱
        parsed_response = parse_chatbot_response(
            answer=clean_answer,
            source_content=clean_source
        )
        
        logger.info(f"✅ 응답 생성 완료: {parsed_response['policy']['name']}")
        return ChatbotResponse(**parsed_response)
        
    except Exception as e:
        logger.error(f"❌ 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{query}")
async def simple_search(query: str, limit: int = 5):
    """간단한 유사도 검색"""
    try:
        results = vectorstore.similarity_search(query, k=limit)
        
        policies = []
        for result in results:
            policies.append({
                "name": result.metadata.get('정책명', ''),
                "category": result.metadata.get('카테고리', ''),
                "target": result.metadata.get('대상', ''),
                "benefits": result.metadata.get('지원내용', ''),
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
    """시스템 상태 확인"""
    try:
        policy_count = len(vectorstore.get()["ids"]) if vectorstore else 0
        return {
            "status": "healthy",
            "model": "gpt-4o-mini",
            "chromadb_path": CHROMADB_PATH,
            "total_policies": policy_count,
            "api_available": qa_chain is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
def root():
    """루트 엔드포인트"""
    return {
        "message": "의성군 정책 챗봇 로컬 API",
        "version": "1.0.0",
        "model": "gpt-4o-mini",
        "description": "ChromaDB + GPT-4o mini 기반 의성군 정책 질의응답",
        "endpoints": {
            "query": "POST /query - 정책 질문하기",
            "search": "GET /search/{query} - 유사도 검색",
            "health": "GET /health - 시스템 상태",
            "docs": "GET /docs - API 문서"
        }
    }

# 콘솔에서 직접 테스트할 수 있는 함수
def test_query(query: str):
    """콘솔에서 직접 테스트"""
    if qa_chain is None:
        return {"error": "시스템이 초기화되지 않았습니다"}
    
    try:
        response = qa_chain.invoke({"query": query})
        source_content = response['source_documents'][0].page_content if response.get('source_documents') else ""
        parsed = parse_chatbot_response(response['result'], source_content)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    print("🏛️ 의성군 정책 챗봇 로컬 서버를 시작합니다...")
    print(f"🤖 모델: GPT-4o mini")
    print(f"📊 ChromaDB: {CHROMADB_PATH}")
    print(f"🌐 서버: http://{HOST}:{PORT}")
    print(f"📚 문서: http://{HOST}:{PORT}/docs")
    print(f"⚙️ 환경 파일: .env")
    print("\n📌 사용법:")
    print("1. POST /query - 정책 질문 (JSON 응답)")
    print("2. GET /search/{질문} - 간단 검색")
    print("3. 콘솔 테스트: test_query('청년 지원 정책 알려줘')")
    
    # 콘솔 테스트 예제
    print("\n🧪 콘솔 테스트 예제:")
    print(test_query("청년 주거 지원 정책이 있나요?"))
    
    uvicorn.run(app, host=HOST, port=PORT, reload=False)