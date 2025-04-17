from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json
import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# JSON 데이터 로딩
with open("data/credit_loans_final_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data:
    loan_name = item.get("대출 상품명")
    bank_name = item.get("은행 이름")
    if not loan_name or not bank_name:
        continue
    content = json.dumps(item, ensure_ascii=False, indent=2)
    doc = Document(
        page_content=content,
        metadata={"loan_name": loan_name, "bank_name": bank_name}
    )
    documents.append(doc)

# 임베딩 및 저장
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
).persist()

print(f"✅ {len(documents)}개의 대출 상품이 임베딩되어 저장되었습니다.")