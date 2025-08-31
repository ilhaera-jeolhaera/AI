# build_chroma_db.py
import os
import glob
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# ------------------------------
# 1. 환경 변수 로드
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 .env에 설정되지 않았습니다.")

# ------------------------------
# 2. 경로 설정
# ------------------------------
SOURCE_DIR = "./data"  # 문서들이 들어있는 폴더
PERSIST_DIR = "./chroma_db_uiseong_100_20250831_new"  # 새 DB 저장 경로
COLLECTION_NAME = "uiseong_policies"

# ------------------------------
# 3. 문서 로드
# ------------------------------
def load_documents() -> list:
    documents = []

    # txt, md 파일 로드
    for filepath in glob.glob(os.path.join(SOURCE_DIR, "*.txt")) + glob.glob(os.path.join(SOURCE_DIR, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(Document(page_content=text, metadata={"source": filepath}))

    # csv 파일 로드
    for filepath in glob.glob(os.path.join(SOURCE_DIR, "*.csv")):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            row_text = " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            documents.append(Document(page_content=row_text, metadata={"source": filepath}))

    return documents

# ------------------------------
# 4. ChromaDB 0.5.x 빌드
# ------------------------------
def build_chroma():
    docs = load_documents()
    if not docs:
        print("❌ 로드된 문서가 없습니다. ./data 폴더에 txt/md/csv 파일을 넣어주세요.")
        return

    print(f"📂 로드된 문서 개수: {len(docs)}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # 새 DB 생성
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    vectordb.persist()
    print(f"✅ 새로운 ChromaDB 생성 완료: {PERSIST_DIR}")

# ------------------------------
# 5. 실행
# ------------------------------
if __name__ == "__main__":
    build_chroma()
