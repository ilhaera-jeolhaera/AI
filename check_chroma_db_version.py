# check_chroma_db_version.py
import os, sys, sqlite3, glob

DEFAULT_DB_DIR = "chroma_db_uiseong_100_20250820_090753"  # ← 기본 폴더 지정

path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_DIR
db = os.path.join(path, "chroma.sqlite3")

if not os.path.exists(db):
    # 0.3~0.4 계열은 parquet/duckdb 구조일 수도 있어서 검사
    has_parquet = bool(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))
    if has_parquet:
        print("✅ Detected: very old Chroma (duckdb/parquet) → treat as 0.4.x (legacy).")
    else:
        print("❌ chroma.sqlite3 not found → 지정한 경로가 ChromaDB가 아닙니다.")
    sys.exit(0)

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("PRAGMA table_info(collections)")
cols = [r[1] for r in cur.fetchall()]
con.close()

if "topic" in cols:
    print("✅ Detected: ChromaDB 0.5.x schema (new).")
else:
    print("✅ Detected: ChromaDB 0.4.x schema (legacy).")
