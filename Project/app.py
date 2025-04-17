import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rag_model import recommend_loans, chat_response
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ✅ 모델 및 인코더 로드
try:
    with open('pkl/xgb_model_grid.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('pkl/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except Exception as e:
    st.error(f"모델 또는 인코더 로드 실패: {e}")
    raise

# 🎯 정규화 범위 설정
ranges = {
    'int_rate': (5, 31),
    'dti': (0.0, 999.0),
    'annual_inc': (0, 110_000_000),
    'funded_amnt': (500, 40_000),
    'mort_acc': (0, 61),
    'fico_range_high': (0, 100),
    'emp_length': (0, 10),
    'num_rev_tl_bal_gt_0': (0, 65)
}

def normalize(value: float, feature: str) -> float:
    min_val, max_val = ranges[feature]
    return (value - min_val) / (max_val - min_val)

# 🎯 채무불이행 예측 함수
def predict_paid(user_data):
    try:
        # 데이터 변환
        input_data = {
            'int_rate': normalize(user_data['int_rate'], 'int_rate'),
            'dti': normalize(user_data['dti'], 'dti'),
            'annual_inc': normalize(user_data['annual_inc']*10000, 'annual_inc'),
            'funded_amnt': normalize(user_data['funded_amnt'], 'funded_amnt'),
            'mort_acc': normalize(user_data['mort_acc'], 'mort_acc'),
            'fico_range_high': normalize(user_data['fico_range_high'], 'fico_range_high'),
            'term': 0 if user_data['term'] == 36 else 1,
            'emp_length': normalize(user_data['emp_length'], 'emp_length'),
            'num_rev_tl_bal_gt_0': normalize(user_data['num_rev_tl_bal_gt_0'], 'num_rev_tl_bal_gt_0'),
            'home_ownership': user_data['home_ownership'],
            'purpose': user_data['purpose'],
            'verification_status': user_data['verification_status']
        }

        # 데이터프레임 변환
        user_df = pd.DataFrame([input_data])[model.feature_names_in_]

        # 범주형 변수 인코딩
        categorical_columns = ['home_ownership', 'purpose', 'verification_status']
        for col in categorical_columns:
            le = label_encoders[col]
            if user_df[col].iloc[0] not in le.classes_:
                user_df[col] = 'Unknown'
                le.classes_ = np.append(le.classes_, 'Unknown')
            user_df[col] = le.transform(user_df[col])

        # 예측 수행
        probability = model.predict_proba(user_df)[0, 1]
        return f"부채 상환 확률: {probability:.2%}"
    
    except Exception as e:
        st.error(f"예측 오류: {str(e)}")
        return None

# ✅ Streamlit 앱 시작
# st.title("🏦 대출 상품 추천 챗봇")
st.markdown("""
<div style='
    background: #d0eaff;
    padding: 2rem;
    margin-bottom:5rem;
    border-radius: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
'>
    <h1 style='color:#004085;'>🏦 대출 상품 추천 챗봇</h1>
    <p style='font-size: 1.2rem; color: #333;'>나에게 딱 맞는 대출 상품을 AI가 추천해드립니다.<br>대출 상환 확률 예측까지 한번에!</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("images/loni.png", width=100)
    st.markdown("## 📘 사용 안내")
    st.info("1. 은행과 신용점수를 입력하세요.\n2. 대출 상품을 추천받으세요.\n3. 자세한 정보를 입력해 예측해보세요.\n4. 챗봇과 추가 상담하세요!")


# 세션 상태 초기화
if "stage" not in st.session_state:
    st.session_state.stage = "select_bank"
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# ================================================================================
# 1페이지: 은행 선택
# ================================================================================
if st.session_state.stage == "select_bank":
    st.subheader("기본 정보 입력")
    
    # 단일 컬럼 레이아웃
    bank = st.selectbox(
        "은행 선택", 
        ["전체은행", "경남은행", "광주은행", "국민은행", "기업은행", "농협은행", 
         "부산은행", "수협은행", "신한은행", "우리은행", "전북은행", 
         "제주은행", "카카오뱅크", "케이뱅크", "토스뱅크", 
         "하나은행", "SC제일은행", "iM뱅크"]
    )
    
    job_status = st.radio(
        "직업 유무", 
        ["예", "아니오"], 
        horizontal=True  # 가로 배치 옵션 추가
    )
    
    credit_score = st.number_input(
        "신용점수 (0~1000)", 
        min_value=0, 
        max_value=1000, 
        value=800,
        help="0부터 1000까지의 점수를 입력하세요"
    )

    # 하단 버튼 정렬
    st.markdown("---")
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("다음 ▶️"):
            st.session_state.update({
                "selected_bank": bank,
                "job_status": job_status,
                "credit_score": credit_score,
                "stage": "recommend_loans"
            })
            st.rerun()



# ================================================================================
# 2페이지: 대출상품 추천
# ================================================================================
elif st.session_state.stage == "recommend_loans":
    st.subheader("✅ 추천 대출 상품")
    
    # 추천 로직
    user_profile = f"직업 유무: {st.session_state.job_status}\n신용점수: {st.session_state.credit_score}"
    recommendations = recommend_loans(user_profile, 
                                    st.session_state.selected_bank, 
                                    st.session_state.credit_score)
    st.markdown(recommendations)
    
    # ✅ 하단 버튼 정렬
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("◀️ 이전"):
            st.session_state.stage = "select_bank"
            st.rerun()
    with col3:
        if st.button("다음 ▶️"):
            st.session_state.stage = "user_input"
            st.rerun()

# ================================================================================
# 3페이지: 사용자 정보 입력 (변경 부분)
# ================================================================================
elif st.session_state.stage == "user_input":
    st.subheader("상세 정보 입력")

    # 입력 필드
    col1, col2 = st.columns(2)
    with col1:
        funded_amnt = st.number_input("대출 금액(만원)", min_value=0, value=0)
        term = st.selectbox("대출 기간(개월)", options=[36, 60])
        int_rate = st.number_input("이자율(%)", min_value=0.0, value=10.0, step=0.1)
        emp_length = st.number_input("근무 기간(년)", min_value=0, value=5)
    with col2:
        annual_inc = st.number_input("연간 소득(만원)", min_value=0, value=3000)
        dti = st.number_input("소득 대비 부채의 비율", min_value=0.0, value=20.0, step=0.1)
        fico_range_high = st.number_input("KCB 점수", min_value=0, max_value=1000, value=st.session_state.get("credit_score", 700))
        mort_acc = st.number_input("담보주택 계좌 수", min_value=0, value=1)
    
    # 한글-영어 매핑 딕셔너리
    home_ownership_map = {
        "담보대출": "MORTGAGE",
        "임대": "RENT",
        "소유": "OWN",
        "기타": "OTHER",
        "없음": "NONE"
    }
    
    verification_status_map = {
        "소득 검증 완료": "Verified",
        "소득 검증 중": "Source Verified",
        "소득 검증 안 됨": "Not Verified"
    }
    
    purpose_map = {
        "부채 통합": "debt_consolidation",
        "신용카드": "credit_card",
        "주택 개선": "home_improvement",
        "기타 용도": "other",
        "대규모 구매": "major_purchase",
        "의료비": "medical",
        "소규모 사업": "small_business",
        "자동차 구매": "car",
        "휴가": "vacation",
        "이사 비용": "moving",
        "주택 구입": "house",
        "결혼 비용": "wedding",
        "재생 에너지": "renewable_energy",
        "교육비": "educational"
    }

    num_rev_tl_bal_gt_0 = st.number_input("리볼빙 계좌 수", min_value=0, value=2)
    home_ownership = st.selectbox("주택 소유 여부", options=list(home_ownership_map.keys()))
    verification_status = st.selectbox("소득 검증", options=list(verification_status_map.keys()))
    purpose = st.selectbox("대출 목적", options=list(purpose_map.keys()))

    # ✅ 하단 버튼 정렬
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 6, 2])
    with col1:
        if st.button("◀️ 이전"):
            st.session_state.stage = "recommend_loans"
            st.rerun()
    with col3:
        if st.button("예측 실행 🔍"):
            user_data = {
                'int_rate': int_rate,
                'dti': dti,
                'annual_inc': annual_inc,
                'funded_amnt': funded_amnt,
                'mort_acc': mort_acc,
                'fico_range_high': fico_range_high,
                'term': term,
                'emp_length': emp_length,
                'num_rev_tl_bal_gt_0': num_rev_tl_bal_gt_0,
                # 한글값을 영어값으로 변환
                'home_ownership': home_ownership_map[home_ownership],
                'verification_status': verification_status_map[verification_status],
                'purpose': purpose_map[purpose]
            }

            result = predict_paid(user_data)
            if result:
                st.session_state.prediction_result = result
                st.session_state.stage = "chatbot"
                st.rerun()

# ================================================================================
# 4페이지: 결과 및 챗봇
# ================================================================================
elif st.session_state.stage == "chatbot":
    st.subheader("분석 결과")
    
    # 결과 표시
    if hasattr(st.session_state, 'prediction_result'):
        st.markdown(f"""
        <div style="
            border: 1px solid #004085; 
            padding: 0.5rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1rem;
        ">
            <h2 style="color: #000; margin-bottom: 0.3rem;font-size: 1.6rem;">
                📊 예상 상환 확률
            </h2>
            <div style="
                font-size: 2.5rem;
                font-weight: bold;
                color: rgb(0, 64, 133);
            ">
                {st.session_state.prediction_result}
            </div>
            <p style="color: #555; font-size: 1rem; margin-top: 1rem;">
                본 확률은 고객님의 소득, 신용, 부채 비율 등의 정보를 기반으로 계산된 AI 예측 결과입니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("예측 결과를 찾을 수 없습니다")

     # 하단 버튼 정렬
    col1, col2, col3 = st.columns([2, 6, 1.5])
    with col1:
        if st.button("◀️ 이전"):
            st.session_state.stage = "user_input"
            st.rerun()
    with col3:
        if st.button("🏠 홈으로"):
            st.session_state.stage = "select_bank"
            st.rerun()

    # 챗봇
    st.markdown("---")
    st.subheader("💬 대출 상담 챗봇")
    
    user_question = st.chat_input("질문을 입력하세요")

    if user_question:
        st.session_state.chat_log.append(("user", user_question))
        bot_answer = chat_response(user_question, st.session_state.selected_bank)
        st.session_state.chat_log.append(("bot", bot_answer))

    for sender, msg in st.session_state.chat_log:
        with st.chat_message("user" if sender == "user" else "assistant"):
            st.markdown(msg)


