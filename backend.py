import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 텔레메트리 비활성화 (ChromaDB 경고 최소화) - 더 강력한 설정
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 토크나이저 병렬 처리 비활성화

# ChromaDB 추가 설정
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 레거시 LangChain 임포트 (langchain_openai, langchain_community 사용 금지)
# pymongo는 사용하지 않습니다 - ChromaDB만 사용
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="의성군 정책 검색 API",
    description="LangChain + ChromaDB를 활용한 의성군 정책 문서 검색 및 질의응답 서비스",
    version="1.0.0"
)

# CORS 설정 (모든 오리진 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 상수 설정
CHROMA_DB_PATH = "./chroma_db_uiseong_100_20250820_090753"  # 로컬 상대경로 직접 지정
COLLECTION_NAME = "uiseong_policies"

# Pydantic 모델
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3  # 검색할 문서 수

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int

# 전역 변수
vectorstore = None
qa_chain = None

def initialize_components():
    """LangChain 컴포넌트 초기화"""
    global vectorstore, qa_chain
    
    try:
        # OpenAI API 키 확인
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
            logger.info("OpenAI 관련 기능이 제한될 수 있습니다.")
        
        logger.info("LangChain 컴포넌트 초기화 시작...")
        
        # ChromaDB 경로 확인
        if not os.path.exists(CHROMA_DB_PATH):
            logger.warning(f"ChromaDB 경로가 존재하지 않습니다: {CHROMA_DB_PATH}")
            logger.info("검색 기능이 제한될 수 있습니다.")
            return
        
        # 임베딩 모델 초기화 (레거시 방식)
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )
        
        # ChromaDB vectorstore 초기화 (0.4.x 스키마 호환성 강화)
        try:
            # 먼저 기본 방식으로 시도
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            
            # 연결 테스트
            test_count = vectorstore._collection.count()
            logger.info(f"ChromaDB 로드 성공: {test_count}개 문서")
            
        except Exception as chroma_error:
            logger.error(f"ChromaDB 로드 실패: {str(chroma_error)}")
            
            # 0.4.x 호환성을 위한 대안 시도
            try:
                logger.info("0.4.x 호환 모드로 재시도...")
                
                # ChromaDB 클라이언트 직접 생성
                import chromadb
                from chromadb.config import Settings
                
                # 0.4.x 호환 설정
                chroma_client = chromadb.PersistentClient(
                    path=CHROMA_DB_PATH,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # 컬렉션 가져오기
                collection = chroma_client.get_collection(COLLECTION_NAME)
                
                # Langchain Chroma wrapper로 감싸기
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings
                )
                
                test_count = collection.count()
                logger.info(f"ChromaDB 호환 모드 로드 성공: {test_count}개 문서")
                
            except Exception as fallback_error:
                logger.error(f"ChromaDB 호환 모드도 실패: {str(fallback_error)}")
                logger.warning("ChromaDB를 사용할 수 없습니다. 검색 기능이 비활성화됩니다.")
                return
        
        # ChatOpenAI 모델 초기화 (레거시 방식)
        if openai_api_key:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
                temperature=0.3
            )
            
            # 프롬프트 템플릿 설정
            prompt_template = """당신은 의성군 정책 전문가입니다. 제공된 문맥을 바탕으로 질문에 대해 정확하고 상세한 답변을 제공해주세요.

문맥: {context}

질문: {question}

답변: 제공된 문맥을 바탕으로 답변드리겠습니다."""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # RetrievalQA 체인 초기화
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info("QA 체인 초기화 완료")
        else:
            logger.warning("OpenAI API 키가 없어 QA 체인을 초기화할 수 없습니다.")
        
        logger.info("LangChain 컴포넌트 초기화 완료")
        
    except Exception as e:
        logger.error(f"컴포넌트 초기화 실패: {str(e)}")
        logger.info("일부 기능이 제한될 수 있지만 서버는 계속 실행됩니다.")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
    try:
        initialize_components()
        logger.info("서비스가 성공적으로 시작되었습니다.")
    except Exception as e:
        logger.error(f"서비스 시작 실패: {str(e)}")
        # 서비스는 계속 실행하되, 에러 상태를 유지

@app.get("/")
async def root():
    """기본 정보 반환"""
    return {
        "service": "의성군 정책 검색 API",
        "version": "1.0.0",
        "status": "running",
        "description": "FastAPI + LangChain + ChromaDB를 활용한 의성군 정책 문서 검색 및 질의응답 서비스",
        "endpoints": {
            "health": "/health 또는 /healthz",
            "search": "/search/{query}",
            "query": "/query (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # 기본 상태 확인
        status_info = {
            "status": "healthy",
            "message": "서비스가 정상 동작 중입니다",
            "chroma_db_path": CHROMA_DB_PATH,
            "chroma_db_exists": os.path.exists(CHROMA_DB_PATH),
            "vectorstore_initialized": vectorstore is not None,
            "qa_chain_initialized": qa_chain is not None
        }
        
        # vectorstore 연결 확인
        if vectorstore is not None:
            try:
                collection = vectorstore._collection
                document_count = collection.count() if collection else 0
                status_info["document_count"] = document_count
            except Exception as e:
                status_info["vectorstore_error"] = str(e)
        
        return status_info
            
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Health check failed: {str(e)}",
            "chroma_db_path": CHROMA_DB_PATH,
            "chroma_db_exists": os.path.exists(CHROMA_DB_PATH)
        }

@app.get("/healthz")
async def healthz():
    """Cloudtype 헬스체크용 엔드포인트"""
    return await health_check()

@app.get("/search/{query}", response_model=SearchResponse)
async def search_documents(query: str, k: int = 5):
    """문서 유사도 검색"""
    try:
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Vectorstore가 초기화되지 않았습니다")
        
        # 유사도 검색 수행
        docs = vectorstore.similarity_search(query, k=k)
        
        # 결과 포맷팅
        formatted_docs = []
        for doc in docs:
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            formatted_docs.append(doc_dict)
        
        return SearchResponse(
            documents=formatted_docs,
            total_count=len(formatted_docs)
        )
        
    except Exception as e:
        logger.error(f"검색 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """질의응답 수행"""
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="QA 체인이 초기화되지 않았습니다")
        
        # RetrievalQA 체인으로 답변 생성
        result = qa_chain({
            "query": request.question
        })
        
        # 결과 파싱
        answer = result.get("result", "답변을 생성할 수 없습니다.")
        source_docs = result.get("source_documents", [])
        
        # 소스 문서 포맷팅
        formatted_sources = []
        for doc in source_docs:
            source_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            formatted_sources.append(source_dict)
        
        return QueryResponse(
            answer=answer,
            source_documents=formatted_sources
        )
        
    except Exception as e:
        logger.error(f"질의응답 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"질의응답 중 오류가 발생했습니다: {str(e)}")

@app.get("/collections")
async def get_collections():
    """ChromaDB 컬렉션 정보 조회"""
    try:
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Vectorstore가 초기화되지 않았습니다")
        
        # 컬렉션 정보 가져오기 (0.4.x 방식)
        collection = vectorstore._collection
        count = collection.count()
        
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": count,
            "persist_directory": CHROMA_DB_PATH
        }
        
    except Exception as e:
        logger.error(f"컬렉션 정보 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"컬렉션 정보 조회 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 의성군 정책 검색 API 서버 시작")
    print("=" * 60)
    print(f"📍 로컬 접속 URL: http://127.0.0.1:8000")
    print(f"📍 API 문서: http://127.0.0.1:8000/docs")
    print(f"📍 ChromaDB 경로: {CHROMA_DB_PATH}")
    print(f"📍 ChromaDB 폴더 존재: {os.path.exists(CHROMA_DB_PATH)}")
    print("=" * 60)
    
    # 로컬 개발용 설정
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",  # 로컬호스트로 명시적 설정
        port=8000,
        reload=True,
        log_level="info"
    )