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
- **Deploy**: Docker, [Cloudtype](https://cloudtype.io/) (배포 완료)

> 본 서버는 **Cloudtype**을 통해 클라우드에 배포되어 운영 중입니다. `dockerfile` 기반으로 이미지가 빌드되며, Cloudtype 환경에서 `OPENAI_API_KEY` 등 환경 변수를 설정해 실행됩니다.

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

---

## 🌾 동기

지자체에는 훌륭한 귀농·귀촌 지원 정책이 많지만, 정작 수요자인 초보 귀농인들은 파편화된 PDF와 복잡한 공문서 속에서 '그래서 내가 대상인지, 얼마를 받을 수 있는지'를 파악하는 데 큰 어려움을 겪고 있었습니다.

저는 기존의 단순한 키워드 검색이나 기계적인 llm 챗봇으로는 이 '정보의 비대칭성'을 결코 해결할 수 없다고 판단했습니다. 따라서 단순히 문서를 찾아주는 수동적인 시스템을 넘어, **AI가 먼저 복잡한 정책을 읽고 해석하여 개개인의 상황에 맞는 맞춤형 답변을 제공하는 '진짜 상담사'를 구축**함으로써, 공공 데이터의 실질적인 활용 가치를 극대화하고자 본 프로젝트를 기획했습니다.

---

## 💻 담당 역할 및 핵심 구현 내용

### 1. AI Data Pipeline & Prompt Engineering

- 분산된 원본 정책 문서(PDF, 웹)를 분석하고, GPT-4를 활용해 5가지 규격화된 **Q&A 데이터셋으로 일괄 변환하는 자동화 파이프라인 직접 구축**
- '전문 상담사 페르소나' 부여 및 불확실한 정보에 대한 '환각(Hallucination) 방지' 조건을 완벽하게 통제하는 프롬프트 엔지니어링 수행
  - `OpenAI API`

---

### 2. Vector DB 구축 & RAG Optimization

- 텍스트 청킹(Chunking)으로 인한 문맥 손실을 방지하기 위해, 정보 결손이 없는 **'1정책=1문서' 매핑 전략을 적용하여 ChromaDB 벡터 스토어 초기 설계 및 임베딩 전 과정 수행**
- 단순 벡터 유사도 검색의 한계를 극복하기 위해, 메타데이터(카테고리) 기반의 이중 필터링 로직을 구현하여 검색 정확도를 실무 수준으로 극대화
  - `LangChain`, `ChromaDB`, `OpenAI Embeddings`

---

### 3. Backend Framework & Integration

- 직접 구축한 ChromaDB와 LangChain의 `RetrievalQA` 체인을 연동하여, 사용자 질문의 맥락을 파악하고 최적의 답변을 반환하는 핵심 추론 로직 설계
- 프론트엔드와 실시간으로 통신하며 AI 예측 결과를 안정적으로 제공하는 **빠르고 가벼운 FastAPI 기반 AI 챗봇 API 서버 개발 및 배포**
  - `FastAPI`, `Python`

---

## TroubleShooting

### 1. [데이터 파편화 및 환각] AI 전처리 파이프라인 도입

- **원인:** 규격이 제각각인 원본 문서를 단순 검색(RAG)에 직결하여 AI의 맥락 파악 실패 및 환각(Hallucination) 발생.
- **해결:** GPT-4를 활용해 흩어진 정보를 5가지 규격의 'Q&A 데이터'로 사전 정제(Parsing)하는 전처리 과정을 거쳐 환각을 원천 차단함.

---

### 2. [문맥 손실] '1정책 = 1문서' 임베딩 전략

- **원인:** 기계적인 텍스트 쪼개기(Chunking)로 인해 정책의 필수 조건과 대상이 분리되는 현상 발생.
- **해결:** 무분별한 청킹을 지양하고, AI가 정제한 Q&A 1건을 온전한 1개의 문서(Document)로 묶어 ChromaDB에 벡터화하여 완벽한 문맥 보존.

---

### 3. [검색 정확도 저하] 메타데이터(Metadata) 다중 필터링

- **원인:** 단순 벡터 유사도(Similarity) 검색만으로는 엉뚱한 카테고리의 정책이 매칭되는 한계 노출.
- **해결:** 문서 임베딩 시 카테고리 메타데이터를 부여하고, LangChain 검색 시 필터링 로직을 이중으로 적용.
