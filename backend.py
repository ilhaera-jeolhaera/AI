from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import re
from datetime import datetime
import uuid
from typing import Optional
import json
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 및 MongoDB 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise ValueError("MONGODB_URI가 .env 파일에 설정되지 않았습니다.")
logger.info(f"사용된 MONGODB_URI: {mongodb_uri}")

# MongoDB 클라이언트 초기화
try:
    client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    logger.info("MongoDB에 성공적으로 연결되었습니다.")
except Exception as e:
    logger.error(f"MongoDB 연결 오류: {e}")
    sys.exit(1)

db = client["us"]
conversations_collection = db["us_policy"]

# 요청 및 응답 모델
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    policy: dict
    contact: dict
    source_summary: str

class HistoryResponse(BaseModel):
    conversations: list[dict]

# OpenAI API 상태 점검
try:
    from openai import OpenAI
    client_openai = OpenAI(api_key=openai_api_key)
    client_openai.embeddings.create(model="text-embedding-ada-002", input="테스트")
except Exception as e:
    logger.error(f"OpenAI API 초기화 오류: {e}")
    sys.exit(1)

# Chroma 벡터 저장소 로드
persist_directory = "chroma_db_20250523_092057"
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
except Exception as e:
    logger.error(f"Chroma 벡터 저장소 로드 오류: {e}")
    sys.exit(1)

# 챗봇 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key, temperature=0.7)
prompt_template = """당신은 의성군 지원 정책 전문가입니다. 주어진 문맥을 바탕으로 자연스럽고 정확한 답변을 제공하세요.

문맥:
{context}

질문: {question}
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)

# 대화 기록 저장
def save_conversation(user_input: str, bot_response: str):
    try:
        logger.info(f"저장하려는 데이터: user_input={user_input}, bot_response={bot_response}")
        timestamp = datetime.now().isoformat()
        conversation = {
            "timestamp": timestamp,
            "user_input": user_input,
            "bot_response": bot_response
        }
        result = conversations_collection.insert_one(conversation)
        logger.info(f"대화 기록 저장됨: input={user_input}, _id={result.inserted_id}")
    except Exception as e:
        logger.error(f"MongoDB 저장 오류: {e}")
        raise

# 대화 기록 조회
def get_conversation_history():
    try:
        history = list(conversations_collection.find().sort("timestamp", 1))
        return [
            {
                "timestamp": doc["timestamp"],
                "user_input": doc["user_input"],
                "bot_response": json.loads(doc["bot_response"]) if doc["bot_response"] and doc["bot_response"].strip() else {}
            }
            for doc in history
        ]
    except Exception as e:
        logger.error(f"MongoDB 조회 오류: {e}")
        return []

# 응답 가공 함수
def parse_policy_response(answer: str, source_content: str = ""):
    if source_content:
        source_content = re.sub(r"질문\s*:.*?(?=\n답변:|\Z)", "", source_content, flags=re.DOTALL)
        source_content = re.sub(r"답변\s*:", "", source_content, flags=re.MULTILINE).strip()
    
    policy = {
        "name": "",
        "target": "",
        "benefits": [],
        "application": {"method": "", "documents": []},
        "notes": []
    }

    lines = answer.split("\n")
    for line in lines:
        line = line.strip()
        if not line: continue
        if "정책명:" in line:
            policy["name"] = line.split(":", 1)[1].strip()
        elif "대상:" in line:
            policy["target"] = line.split(":", 1)[1].strip() + " (외국인 가능성 있음)"
        elif "지원 내용:" in line or line.startswith("-"):
            if "지원 내용:" not in line:
                policy["benefits"].append(line.strip("- ").strip())
        elif "신청 방법:" in line:
            policy["application"]["method"] = line.split(":", 1)[1].strip()
        elif "필요 서류:" in line:
            policy["application"]["documents"] = [doc.strip() for doc in line.split(":", 1)[1].split(",")]
        elif "참고 사항:" in line or line.startswith("-") and "지원 내용" not in line:
            if "참고 사항:" not in line:
                policy["notes"].append(line.strip("- ").strip())

    policy["notes"].append("외국인 자격은 의성군청 확인 필요")

    contact = {
        "phone": "054-830-6114",
        "website": "https://www.uisung.go.kr"
    }

    source_summary = source_content if source_content else "데이터 없음"
    logger.info(f"가공 후 source_summary: {source_summary!r}")

    return {
        "policy": policy,
        "contact": contact,
        "source_summary": source_summary
    }

# /query 엔드포인트
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        logger.info(f"처리 시작 - query: {request.query}")
        logger.info("OpenAI API 호출 시작")
        response = qa_chain.invoke({"query": request.query})
        logger.info(f"OpenAI API 호출 완료 - response['result']: {response['result']}, source_documents 길이: {len(response.get('source_documents', []))}")
        
        source_content = response['source_documents'][0].page_content if response.get('source_documents') and len(response['source_documents']) > 0 else "정보 없음"
        logger.info(f"원본 source_content: {source_content}")

        policy_keywords = ["정책", "지원", "주거", "청년", "신혼부부", "의성", "주택", "금융", "신청", "자격"]
        query_lower = request.query.lower()
        is_policy_related = any(keyword in query_lower for keyword in policy_keywords)

        if not is_policy_related:
            default_response = {
                "policy": {
                    "name": "",
                    "target": "",
                    "benefits": [],
                    "application": {"method": "", "documents": []},
                    "notes": ["이 질문은 의성군 정책과 관련이 없는 것 같습니다. 예를 들어, '의성군 청년층 지원 정책'이나 '신혼부부 주거 지원 조건'에 대해 물어보세요!"]
                },
                "contact": {
                    "phone": "054-830-6114",
                    "website": "https://www.uisung.go.kr"
                },
                "source_summary": "정책 관련 질문을 기다립니다."
            }
            logger.info(f"정책 관련성 없음 - 반환: {json.dumps(default_response, ensure_ascii=False)}")
            return QueryResponse(**default_response)

        logger.info("응답 가공 시작")
        parsed_response = parse_policy_response(
            answer=response['result'],
            source_content=source_content
        )
        logger.info(f"가공 완료 - parsed_response: {json.dumps(parsed_response, ensure_ascii=False)}")
        
        logger.info("메모리 저장 시작")
        memory.save_context({"input": request.query}, {"output": response["result"]})
        logger.info("메모리 저장 완료")
        
        logger.info("DB 저장 시작")
        save_conversation(request.query, json.dumps(parsed_response))
        logger.info("DB 저장 완료")
        
        logger.info(f"응답 반환 - response: {json.dumps(parsed_response, ensure_ascii=False)}")
        return QueryResponse(**parsed_response)
    except Exception as e:
        logger.error(f"/query 오류: {e}")
        raise HTTPException(status_code=500, detail=f"오류: {str(e)}")

# /history 엔드포인트
# 기존 문제 코드 삭제 후 다음으로 수정
@app.get("/history", response_model=HistoryResponse)
async def get_history():  # async def로 통일
    try:
        history = get_conversation_history()
        return {"conversations": history}  # 모델과 일치하도록 반환 
    except Exception as e:
        logger.error(f"/history 오류: {e}")
        raise HTTPException(status_code=500, detail=f"오류: {str(e)}")

# health_check는 별도 엔드포인트로 분리 (필요시)
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
    

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)