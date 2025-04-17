import os
import re
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"
SEARCH_K = 8
CHROMA_PATH = "./chroma_db"      # 기존 Chroma용 DB (초기 문서 로딩용)
FAISS_PATH = "./faiss_db"        # 저장할 FAISS 벡터 DB 경로

# 베딩 모델 및 LLM
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model="gpt-4o-mini")

# 공통 프롬프트 정의
recommend_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 은행 대출 상품 추천 전문가입니다.

[대출 상품 정보]
{context}

[고객 정보 및 요청]
{question}

아래 지침을 반드시 따르세요:

1. 동일한 상품이 context에 여러 번 등장하거나 다양한 조건에 맞더라도 **한 번만 추천**하세요.
2. **서로 다른 상품만 최대 3개까지** 추천하세요.
3. 상품명은 반드시 context 내에 존재하는 실제 상품만 사용하세요.
4. 고객의 신용 점수에 해당하는 구간(예: 800점대, 900점 이상 등)의 이자율만 포함하세요.
5. 존재하지 않는 신용 점수 구간(예: context에 없는 700점대 이자율 등)은 절대 생성하지 마세요.
"""
)

general_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 은행 대출 상품에 대해 정보를 제공하는 AI 상담사입니다.

[대출 상품 정보]
{context}

[고객 질문]
{question}

고객이 질문한 특정 대출 상품에 대해 정확하고 간결하게 답변해 주세요.

- 필요서류, 상환방식, 신청 조건 등 질문에 맞는 정보만 제공하세요.
- 지어내지 마세요. context에 있는 상품만 기준으로 답변하세요.
"""
)

# 신용 점수 구간 변환
def get_score_band(score: int) -> str:
    if score >= 900: return "900점 이상"
    elif score >= 800: return "800점대"
    elif score >= 700: return "700점대"
    elif score >= 600: return "600점대"
    elif score >= 500: return "500점대"
    elif score >= 400: return "400점대"
    else: return "400점 미만"

# 중복 상품 제거
def remove_duplicate_products(response: str) -> str:
    seen = set()
    lines = response.split("\n")
    filtered = []
    for line in lines:
        match = re.match(r"^\d+\.\s+\*\*(.*?)\*\*", line)
        if match:
            product_name = match.group(1).strip()
            if product_name in seen:
                continue
            seen.add(product_name)
        filtered.append(line)
    return "\n".join(filtered)

# 안전하게 저장 & 로드되는 FAISS 벡터스토어
def load_or_create_faiss_index() -> FAISS:
    # 이미 저장된 FAISS 인덱스가 있는 경우 → 보안 예외 설정 포함
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            folder_path=FAISS_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # ✅ 안전한 환경에서만 허용
        )

    # 최초 실행: Chroma에서 문서를 불러와 FAISS 인덱스 생성
    chroma_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    data = chroma_store.get()

    docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]

    faiss_index = FAISS.from_documents(docs, embedding=embeddings)
    faiss_index.save_local(FAISS_PATH)

    return faiss_index

# ✅ 선택 은행 문서 필터링
def filter_documents_by_bank(bank_name: str) -> list[Document]:
    faiss_index = load_or_create_faiss_index()
    all_docs = faiss_index.docstore._dict.values()
    
    # 전체은행 선택 시 모든 문서 반환
    if bank_name == "전체은행":
        return list(all_docs)
        
    # 특정 은행 선택 시 필터링
    return [doc for doc in all_docs if doc.metadata.get("bank_name") == bank_name]

# ✅ RAG 체인 생성
def create_rag_chain(docs: list[Document], prompt: PromptTemplate) -> RetrievalQA:
    temp_vs = FAISS.from_documents(docs, embedding=embeddings)  # 임시 retriever
    retriever = temp_vs.as_retriever(search_type="mmr", search_kwargs={"k": SEARCH_K})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

# 대출 상품 추천 
def recommend_loans(user_profile: str, selected_bank: str, credit_score: int, risk_percent: float = None) -> str:
    """
    사용자의 정보와 선택한 은행을 바탕으로 대출 상품을 추천합니다.

    Args:
        user_profile (str): 직업 유무, 신용점수 등 사용자 프로필 정보 (자연어 형태)
        selected_bank (str): 선택된 은행 (예: "신한은행", "전체은행")
        credit_score (int): 고객의 신용 점수 (0~1000)
        risk_percent (float, optional): 채무불이행 확률 (%). 기본값 None.

    Returns:
        str: 추천 대출 상품 리스트 (최대 3개, 중복 제거)
    """

    # 선택한 은행의 문서 필터링
    docs = filter_documents_by_bank(selected_bank)
    
    # 문서 존재 여부 검증
    if not docs:
        return f"'{selected_bank}' 은행에 적합한 대출 상품을 찾지 못했습니다."

    # RAG 체인 초기화
    qa_chain = create_rag_chain(docs, recommend_prompt)
    
    # 신용점수 구간 계산
    score_band = get_score_band(credit_score)

    # 프롬프트 문구 설정
    if selected_bank == "전체은행":
        bank_instruction = "※ 은행에 제한 없이 고객에게 적합한 상품을 추천해 주세요."
    else:
        bank_instruction = f"※ 반드시 '{selected_bank}' 은행의 상품만 추천하세요."
    
    # 동적 쿼리 생성
    query = (
        f"{user_profile}\n"
        f"{bank_instruction}\n"
        f"※ 고객의 신용 점수는 {credit_score}점이며, '{score_band}' 이자율만 고려해 주세요.\n"
    )
    
    # 리스크 퍼센티지 조건부 추가
    if risk_percent is not None:
        query += f"※ 고객의 예상 채무불이행 확률은 {risk_percent:.1f}%입니다. 이를 참고하여 상품을 추천해 주세요.\n"

    # LLM 실행 및 응답 생성
    response = qa_chain.run({"query": query})
    
    # 중복 상품 제거 후 반환
    return remove_duplicate_products(response)

# ✅ 챗봇 질문 응답
def chat_response(question: str, selected_bank: str) -> str:
    """
    고객 질문에 대해 선택한 은행의 문서를 기반으로 정보를 제공하는 챗봇 응답 함수

    Args:
        question (str): 고객 질문
        selected_bank (str): 선택된 은행 ("전체은행" 포함)

    Returns:
        str: AI의 응답 메시지
    """
     # 선택한 은행의 문서 필터링
    docs = filter_documents_by_bank(selected_bank)

    # 문서 없음 처리
    if not docs:
        return f"'{selected_bank}' 은행의 정보가 없어 답변드릴 수 없습니다."

    # RAG 체인 구성
    qa_chain = create_rag_chain(docs, general_prompt)

    # 은행 조건 문구 설정
    if selected_bank == "전체은행":
        bank_instruction = "※ 은행에 제한 없이 답변해 주세요. ※ 상품 추천은 최대 3개까지만 해주세요."
    else:
        bank_instruction = f"※ 반드시 '{selected_bank}' 은행의 상품 정보만 참고하세요."

    # 쿼리 구성
    refined_question = (
        f"아래 질문에 답변해 주세요.\n"
        f"{bank_instruction}\n\n"
        f"질문: {question}"
    )

    # 실행 및 응답
    return qa_chain.run({"query": refined_question})