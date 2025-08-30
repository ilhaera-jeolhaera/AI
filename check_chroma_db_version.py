# check_chroma_db_version.py
import os, sys, sqlite3, glob
from dotenv import load_dotenv

# -----------------------------
# 1. .env 파일 로드
# -----------------------------
load_dotenv()

# .env에 정의된 CHROMADB_PATH 읽기
env_path = os.getenv("CHROMADB_PATH")

# 실행 인자가 있으면 우선 사용, 없으면 .env → 기본값 순서로 결정
DEFAULT_DB_DIR = "chroma_db_uiseong_100_20250820_090753"
path = sys.argv[1] if len(sys.argv) > 1 else (env_path or DEFAULT_DB_DIR)

print(f"🔍 검사할 ChromaDB 경로: {path}")

# -----------------------------
# 2. DB 파일 검사
# -----------------------------
db = os.path.join(path, "chroma.sqlite3")

if not os.path.exists(db):
    # 0.3~0.4 계열은 parquet/duckdb 구조일 수도 있어서 검사
    has_parquet = bool(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))
    if has_parquet:
        print("✅ Detected: very old Chroma (duckdb/parquet) → treat as 0.4.x (legacy).")
    else:
        print("❌ chroma.sqlite3 not found → 지정한 경로가 ChromaDB가 아닙니다.")
    sys.exit(0)

# -----------------------------
# 3. 스키마 버전 확인
# -----------------------------
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("PRAGMA table_info(collections)")
cols = [r[1] for r in cur.fetchall()]
con.close()

if "topic" in cols:
    print("✅ Detected: ChromaDB 0.5.x schema (new).")
else:
    print("✅ Detected: ChromaDB 0.4.x schema (legacy).")
