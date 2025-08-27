FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=false \
    CHROMA_TELEMETRY_IMPLEMENTATION=none

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (캐시 최적화)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

# 앱 복사
COPY backend.py /app/backend.py

# (옵션) DB를 이미지에 포함하고 싶을 때만 사용
# COPY chroma_db_uiseong_100_20250820_090753/ /app/chroma_db_uiseong_100_20250820_090753/

CMD ["python", "backend.py"]
