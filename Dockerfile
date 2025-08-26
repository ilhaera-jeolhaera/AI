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

# requirements.txt (레포에 실제로 존재하는 파일명!)
COPY requirements.txt /app/requirements.txt
RUN python -V && pip -V \
 && pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

# 앱 코드 복사 (파일명 일치!)
COPY backend.py /app/backend.py

# (선택) ChromaDB 폴더를 이미지 안에 포함하고 싶을 때만:
# 폴더명이 실제와 다르면 아래 경로 수정
# COPY chroma_db_uiseong_100_20250820_090753/ /app/chroma_db_uiseong_100_20250820_090753/

CMD ["python", "backend.py"]
