import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ======================
# 1. 모델, 인코더, 스케일러 로드
# ======================
@st.cache_resource
def load_ml_assets():
    model = joblib.load("./models/modeljunghoKim.pkl")
    scaler = joblib.load("./models/scalerjunghoKim.pkl")
    return model, scaler

model, scaler = load_ml_assets()

st.title("퇴사 여부 예측 시스템")
st.write("직원의 정보를 입력하여 퇴사 가능성을 예측합니다.")

# ======================
# 2. 사용자 입력
# ======================
# satisfaction_level은 0.0 ~ 1.0 사이의 실수
satisfaction_level = st.slider("만족도 (0.0 ~ 1.0)", 0.0, 1.0, 0.5)
# number_project는 프로젝트 수 (정수)
number_project = st.number_input("프로젝트 수", min_value=1, max_value=10, value=5)
# time_spend_company는 보통 연수 (정수)
time_spend_company = st.number_input("근무 연수", min_value=1, max_value=20, value=10)

# ======================
# 3. 데이터 구성 및 예측
# ======================
if st.button("예측하기"):
    # 1. 입력 데이터 프레임 생성
    input_data = pd.DataFrame({
        'satisfaction_level': [satisfaction_level],
        'number_project': [number_project],
        'time_spend_company': [time_spend_company]
    })

    # 2. 인코딩
    try:
        
        input_data_scaled = scaler.transform(input_data)
        # 모델이 피처 이름을 기대한다면 DataFrame으로 다시 변환
        input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)

        # 4. 예측 수행
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0][1] # 퇴사(1) 확률

        if prediction == 1:
            st.error(f" :red[퇴사확률: {probability:.2%}]")
        else:
            st.success(f" :green[잔류 확률: {probability:.2%}]")

        st.write("### 입력 데이터 상세")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")
        st.info("팁: 학습 데이터에 없던 숫자가 입력되었을 수 있습니다 (LabelEncoder 문제).")