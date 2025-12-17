1. 프로그램 개요

본 프로그램은 학생의 학습 관련 요인들을 입력값으로 받아, 딥러닝(신경망) 모델을 이용해 시험 성적을 예측하는 회귀(Regression) 문제를 해결하는 것을 목표로 한다.
머신러닝 중에서도 다층 퍼셉트론(MLP) 기반의 딥러닝 모델을 사용하여, 단순 선형 관계뿐 아니라 복잡한 비선형 관계까지 학습하도록 설계하였다.

2. 사용 데이터 및 입력 변수

본 프로젝트에서는 실제 데이터를 가정한 **모의 데이터(fake data)**를 생성하여 학습을 진행하였다.
입력 변수는 학생의 성적에 영향을 줄 수 있는 대표적인 요인들로 구성하였다.
변수명	설명
study_hours	주당 공부 시간
attendance	출석률 (%)
sleep_hours	하루 평균 수면 시간
prev_score	이전 시험 점수
tutoring	과외 여부 (0: 없음, 1: 있음)
stress	스트레스 정도 (0~10)

출력값은 **예측된 시험 성적(score)**이며, 0~100 사이의 연속적인 실수값이다.
3. 문제 유형: 회귀(Regression)

이 문제는 결과가 “합격/불합격”처럼 범주로 나뉘는 분류 문제가 아니라, 연속적인 점수 값을 예측하는 회귀 문제이다.
따라서 신경망의 마지막 출력층에는 활성화 함수를 사용하지 않고, 손실 함수로는 **평균제곱오차(MSE)**를 사용하였다.

4. 데이터 전처리

딥러닝 모델은 입력 데이터의 스케일에 민감하기 때문에, 모든 입력 변수에 대해 **표준화(Standardization)**를 수행하였다.
이를 통해 각 변수는 평균 0, 표준편차 1의 분포를 갖게 되며, 학습 속도와 안정성이 향상된다.

또한 전체 데이터를 학습용과 테스트용으로 분리하여, 모델이 보지 않은 데이터에 대해서도 일반화 성능을 평가할 수 있도록 하였다.

5. 딥러닝 모델 구조

본 모델은 다층 퍼셉트론(MLP) 구조를 기반으로 하며, 다음과 같은 특징을 가진다.

입력층: 학생의 학습 관련 변수들

은닉층: ReLU 활성화 함수를 사용하는 여러 개의 Dense Layer

Dropout: 과적합(overfitting)을 방지하기 위한 기법

출력층: 시험 성적을 예측하는 단일 노드

이러한 구조를 통해 모델은 입력 변수 간의 복잡한 상호작용을 학습할 수 있다.

6. 모델 학습 방법

모델 학습에는 Adam 옵티마이저를 사용하였으며, 검증 데이터의 손실 값이 더 이상 감소하지 않을 경우 학습을 자동으로 중단하는 조기 종료(Early Stopping) 기법을 적용하였다.
이를 통해 불필요한 반복 학습을 방지하고, 최적의 성능을 가진 모델을 선택할 수 있도록 하였다.

7. 성능 평가 지표

모델의 성능은 다음과 같은 회귀 평가 지표를 통해 측정하였다.

MAE (Mean Absolute Error): 평균 절대 오차

RMSE (Root Mean Squared Error): 큰 오차에 더 민감한 지표

R² Score: 모델이 데이터 분산을 얼마나 잘 설명하는지 나타내는 지표

이러한 지표를 통해 예측 성능을 정량적으로 분석할 수 있다.

8. 예측 기능

학습이 완료된 모델은 새로운 학생의 정보를 입력받아, 예상 시험 성적을 출력할 수 있다.
이를 통해 학생의 학습 패턴에 따른 성적 변화를 사전에 예측하고, 학습 전략 수립에 활용할 수 있다.

9. 기대 효과 및 활용 가능성

본 프로그램은 딥러닝 기반 회귀 모델의 기본적인 구조와 학습 과정을 이해하는 데 도움을 주며, 다음과 같은 확장이 가능하다.

실제 학생 데이터를 활용한 성적 예측

웹 기반 인터페이스(Streamlit)를 통한 사용자 입력

중요 변수 분석을 통한 학습 전략 제안

10. 결론

본 프로젝트는 비교적 단순한 데이터 구조를 활용하면서도, 딥러닝 모델을 적용하여 실제 문제 해결에 머신러닝을 활용하는 전체 과정을 경험할 수 있는 예제이다.
이를 통해 데이터 전처리, 모델 설계, 학습, 평가까지의 흐름을 종합적으로 이해할 수 있다.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# 1) (예시) 데이터 생성
# =========================
def make_fake_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(0, 25, size=n)          # 공부 시간 (0~25시간/주)
    attendance  = rng.uniform(50, 100, size=n)        # 출석률 (50~100%)
    sleep_hours = rng.uniform(3, 9, size=n)           # 평균 수면 (3~9시간/일)
    prev_score  = rng.uniform(0, 100, size=n)         # 이전 시험 점수
    tutoring    = rng.integers(0, 2, size=n)          # 과외 여부(0/1)
    stress      = rng.uniform(0, 10, size=n)          # 스트레스(0~10)

    # 점수 생성 (현실적으로 보이도록 가중치 + 노이즈)
    noise = rng.normal(0, 6, size=n)
    score = (
        2.2 * study_hours
        + 0.35 * attendance
        + 1.5 * sleep_hours
        + 0.45 * prev_score
        + 4.0 * tutoring
        - 1.2 * stress
        + noise
    )

    # 0~100으로 클립
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "prev_score": prev_score,
        "tutoring": tutoring,
        "stress": stress,
        "score": score
    })
    return df

df = make_fake_data(n=2500)
print(df.head())

# =========================
# 2) 학습/테스트 분리
# =========================
X = df.drop(columns=["score"]).values
y = df["score"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3) 스케일링 (딥러닝에 중요)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# 4) 딥러닝 모델 구성 (회귀)
# =========================
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(1)  # 회귀: 마지막은 활성화 없음(선형)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

# =========================
# 5) 학습 (조기 종료)
# =========================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 6) 평가
# =========================
pred = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n=== Test Evaluation ===")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R^2  : {r2:.3f}")

# =========================
# 7) 새 입력 예측 함수
# =========================
def predict_score(study_hours, attendance, sleep_hours, prev_score, tutoring, stress):
    x = np.array([[study_hours, attendance, sleep_hours, prev_score, tutoring, stress]], dtype=float)
    x_scaled = scaler.transform(x)
    y_pred = model.predict(x_scaled).item()
    return float(np.clip(y_pred, 0, 100))

# 예시 예측
example = predict_score(
    study_hours=12,
    attendance=95,
    sleep_hours=7,
    prev_score=78,
    tutoring=1,
    stress=4
)
print(f"\n예측 점수: {example:.2f}")
