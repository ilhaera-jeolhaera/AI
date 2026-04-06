# 의성군 정책 RAG 서버 (Uiseong Policy RAG Server)

FastAPI 기반 RAG(Retrieval-Augmented Generation) 서버입니다. ChromaDB 벡터 검색과 OpenAI를 결합하여 의성군 정책 문서에 대한 지능형 검색 및 질의응답을 제공합니다.

## 주요 기능

- **벡터 검색**: ChromaDB를 이용한 의성군 정책 문서 유사도 검색
- **RAG 질의응답**: OpenAI GPT-3.5-turbo 기반 한국어 답변 생성
- **가독성 포맷팅**: 마크다운 제거 및 라벨 기반(정책명/대상/지원내용/신청방법/필요서류/참고사항) 정리
- **CORS 지원**: 프론트엔드 통합용 CORS 미들웨어 설정
- **헬스 체크**: `/health`, `/healthz` 엔드포인트 제공

## 기술 스택

- **Backend**: FastAPI, Uvicorn
- **Vector DB**: ChromaDB (`text-embedding-3-small` 임베딩)
- **LLM**: OpenAI GPT-3.5-turbo
- **Data**: Pandas (CSV 로더)
- **Deploy**: Docker

## 프로젝트 구조

```
.
├── backend.py                          # FastAPI RAG 서버 메인
├── tet.py                              # ChromaDB 초기 빌드 스크립트 (LangChain)
├── requirements.txt                    # Python 의존성
├── dockerfile                          # 컨테이너 빌드 정의
├── attached_assets/                    # 원본 CSV 정책 문서
└── chroma_db_uiseong_20250831_new/     # 영속화된 ChromaDB 저장소
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
export OPENAI_API_KEY=sk-...
export PORT=8000                   # (선택)
export CORS_ALLOW_ORIGINS=...      # (선택, 쉼표 구분)
```

### 3. 서버 실행

```bash
python backend.py
```

서버는 기본적으로 `http://0.0.0.0:8000`에서 실행됩니다.

### 4. Docker 실행

```bash
docker build -t uiseong-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... uiseong-rag
```

## API 엔드포인트

| Method | Path              | 설명                        |
| ------ | ----------------- | --------------------------- |
| GET    | `/`               | 서비스 정보                 |
| GET    | `/health`         | 헬스 체크                   |
| GET    | `/healthz`        | 헬스 체크 (k8s 호환)        |
| GET    | `/search/{query}` | 벡터 유사도 검색            |
| POST   | `/query`          | RAG 기반 질의응답           |

### 예시: `/query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "청년 창업 지원 정책 알려줘", "max_results": 5}'
```

응답:

```json
{
  "question": "청년 창업 지원 정책 알려줘",
  "answer": "정책명: ...\n대상: ...\n지원내용: ..."
}
```

## 데이터 로딩

`attached_assets/` 내 CSV 파일들이 자동으로 읽혀 ChromaDB 컬렉션 `uiseong_policies`에 적재됩니다. 컬렉션이 비어 있거나 존재하지 않을 때만 초기 로딩이 수행됩니다.
