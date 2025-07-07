
# 🤖 머신러닝(Machine Learning) 개요 및 기초 정리

> 머신러닝(Machine Learning)은 데이터를 기반으로 컴퓨터가 스스로 학습하고 예측하거나 분류하는 기술입니다.  
> 이 문서는 머신러닝의 개요, 종류, 알고리즘, 흐름까지 기초 개념을 전체적으로 정리한 자료입니다.

---

## 📌 머신러닝 개요

- **머신러닝(Machine Learning)**은 명시적으로 프로그래밍하지 않고 **데이터로부터 자동으로 학습**하는 알고리즘 기술입니다.
- 전통적인 프로그래밍은 규칙과 데이터를 입력해 결과를 얻는 방식이라면, 머신러닝은 **데이터와 결과를 입력해 규칙을 학습**합니다.
- 인공지능(AI)의 한 분류로, 특히 **패턴 인식, 예측, 분류, 최적화** 등에 사용됩니다.

---

## 📂 머신러닝의 주요 종류

| 구분 | 설명 | 대표 알고리즘 | 예시 |
|------|------|----------------|------|
| **지도 학습 (Supervised Learning)** | 정답(레이블)이 있는 데이터를 학습 | 선형 회귀, 로지스틱 회귀, 결정 트리 등 | 이메일 스팸 분류, 집값 예측 |
| **비지도 학습 (Unsupervised Learning)** | 정답 없이 데이터의 구조를 파악 | K-평균, PCA 등 | 고객 군집화, 차원 축소 |
| **강화 학습 (Reinforcement Learning)** | 보상을 통해 최적의 행동을 학습 | Q-Learning, DQN 등 | 게임, 로봇 제어, 자율주행 |
| **준지도 학습 (Semi-supervised Learning)** | 일부 라벨 데이터와 많은 비라벨 데이터 혼합 학습 | 혼합 알고리즘 | 텍스트 분류, 음성 인식 |
| **자가 지도 학습 (Self-supervised Learning)** | 데이터에서 라벨을 스스로 생성하여 학습 | 트랜스포머 기반 모델 | 언어 모델, 이미지 생성 |

---

## 🧠 지도 학습 유형

### 1. 분류 (Classification)
- 입력 데이터를 카테고리로 분류
- 예: 고양이/개 분류, 암/정상 진단, 이메일 스팸 여부 등

### 2. 회귀 (Regression)
- 연속적인 수치 예측
- 예: 주택 가격 예측, 연봉 예측, 기온 예측 등

---

## 📊 머신러닝의 기본 프로세스

1. **문제 정의**  
2. **데이터 수집 및 전처리**  
3. **훈련/검증/테스트 데이터 분할**  
4. **모델 선택 및 학습**  
5. **성능 평가 및 튜닝**  
6. **예측/서비스에 적용**

---

## 📁 데이터 전처리

| 항목 | 설명 |
|------|------|
| 결측값 처리 | 평균값, 중앙값으로 대체 또는 제거 |
| 이상치 처리 | IQR 또는 Z-score 등으로 탐지 |
| 인코딩 | 라벨 인코딩, 원-핫 인코딩 등 |
| 정규화/표준화 | `MinMaxScaler`, `StandardScaler` |
| 데이터 분할 | `train_test_split()` (scikit-learn) |

---

## 🔧 대표적인 머신러닝 알고리즘

| 알고리즘 | 유형 | 설명 |
|----------|------|------|
| 선형 회귀 (Linear Regression) | 회귀 | 직선 형태로 예측 |
| 로지스틱 회귀 (Logistic Regression) | 분류 | 이진 또는 다중 클래스 분류 |
| 결정 트리 (Decision Tree) | 분류/회귀 | 트리 기반 규칙 분할 |
| 랜덤 포레스트 (Random Forest) | 분류/회귀 | 여러 결정 트리의 앙상블 |
| SVM (Support Vector Machine) | 분류 | 최적의 결정 경계 계산 |
| KNN (K-Nearest Neighbors) | 분류/회귀 | 주변 이웃 기준 |
| 나이브 베이즈 (Naive Bayes) | 분류 | 조건부 확률 기반 분류기 |
| K-평균 (K-Means) | 군집화 | 중심점 기반의 비지도 군집화 |

---

## 📐 성능 평가 지표

| 지표 | 유형 | 설명 |
|------|------|------|
| Accuracy | 분류 | 전체 중 맞춘 비율 |
| Precision | 분류 | 예측 Positive 중 실제 Positive 비율 |
| Recall | 분류 | 실제 Positive 중 예측 Positive 비율 |
| F1 Score | 분류 | 정밀도와 재현율의 조화 평균 |
| ROC-AUC | 분류 | 분류기의 구분 능력 |
| MAE / MSE / RMSE | 회귀 | 평균 오차, 제곱 오차, 제곱근 오차 |

---

## 🧰 머신러닝 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| NumPy | 수치 연산 |
| pandas | 데이터 처리 |
| matplotlib / seaborn | 시각화 |
| scikit-learn | ML 모델, 전처리, 평가 |
| XGBoost / LightGBM | 고성능 부스팅 모델 |
| TensorFlow / PyTorch | 딥러닝 프레임워크 (심화) |

---

## 📚 참고 자료

- [Scikit-learn 공식 문서](https://scikit-learn.org/)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Awesome Machine Learning GitHub](https://github.com/josephmisiti/awesome-machine-learning)

---

> 작성자: [사용자 이름]  
> 날짜: 2025년 7월
