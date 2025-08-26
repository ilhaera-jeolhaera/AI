# CloudType 최적화 Dockerfile
FROM python:3.10-slim

# 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=false \
    CHROMA_TELEMETRY_IMPLEMENTATION=none

WORKDIR /app

# 시스템 패키지 (빌드 도구)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements 복사 및 설치
COPY requirements_cloudtype.txt /app/requirements.txt

# 의존성 설치 (CloudType 최적화)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# 버전 확인 (디버깅용)
RUN python -c "import langchain, chromadb, openai; print(f'langchain: {langchain.__version__}, chromadb: {chromadb.__version__}, openai: {openai.__version__}')"

# 앱 파일 복사
COPY backend_cloudtype.py /app/backend.py
COPY chroma_db_uiseong_100_20250820_090753/ /app/chroma_db_uiseong_100_20250820_090753/

# 포트 노출
EXPOSE 8000

# 건강 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 시작 명령
CMD ["python", "backend.py"]