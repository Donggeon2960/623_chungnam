
# 🤖 머신러닝(Machine Learning) 기초 정리

> 머신러닝(Machine Learning)은 데이터를 기반으로 컴퓨터가 스스로 학습하고 예측하거나 분류하는 기술입니다. 이 문서는 머신러닝의 기초 개념부터 전형적인 흐름까지 정리합니다.

---

## 📌 머신러닝이란?

- 데이터를 분석하여 **패턴을 학습**하고, 학습된 모델로 새로운 데이터를 예측
- 명시적인 프로그래밍 없이도 **스스로 향상되는 알고리즘**
- 인공지능(AI)의 한 분야이며, 딥러닝은 머신러닝의 하위 분야

---

## 📂 머신러닝의 분류

| 종류 | 설명 | 예시 |
|------|------|------|
| 지도 학습 (Supervised Learning) | 입력과 정답(레이블)을 기반으로 학습 | 분류(Classification), 회귀(Regression) |
| 비지도 학습 (Unsupervised Learning) | 정답 없이 데이터 구조를 학습 | 군집화(Clustering), 차원 축소 |
| 강화 학습 (Reinforcement Learning) | 보상을 최대화하는 행동을 학습 | 게임, 로봇 제어 |

---

## 🧠 지도 학습 유형

### 1. 분류 (Classification)
- 목표: 입력 데이터를 **카테고리로 분류**
- 예: 스팸 메일 분류, 질병 진단, 이미지 분류

### 2. 회귀 (Regression)
- 목표: 연속적인 값을 **예측**
- 예: 집값 예측, 온도 예측, 주식 가격 예측

---

## 📊 머신러닝의 기본 흐름

1. 문제 정의  
2. 데이터 수집 및 전처리  
3. 훈련/테스트 데이터 분할  
4. 모델 선택 및 학습  
5. 성능 평가  
6. 하이퍼파라미터 튜닝  
7. 예측 및 실제 서비스 적용

---

## 📁 데이터 전처리

| 항목 | 설명 |
|------|------|
| 결측값 처리 | 평균 대체, 제거 등 |
| 범주형 인코딩 | 원-핫 인코딩, 라벨 인코딩 |
| 정규화/표준화 | Min-Max Scaling, Z-score |
| 훈련/테스트 분리 | `train_test_split()` (sklearn) |

---

## 🔧 대표적인 알고리즘

| 알고리즘 | 설명 |
|----------|------|
| 선형 회귀 (Linear Regression) | 회귀 문제를 위한 기본 모델 |
| 로지스틱 회귀 (Logistic Regression) | 이진 분류 |
| 결정 트리 (Decision Tree) | 분기 구조 기반의 모델 |
| 랜덤 포레스트 (Random Forest) | 여러 트리의 앙상블 |
| KNN (K-Nearest Neighbors) | 주변 이웃 기반 분류 |
| SVM (Support Vector Machine) | 초평면을 이용한 분류 |
| 나이브 베이즈 (Naive Bayes) | 조건부 확률 기반 분류기 |
| K-평균 (K-Means) | 비지도 학습 - 군집화 알고리즘 |

---

## 📐 성능 평가 지표

| 지표 | 설명 |
|------|------|
| 정확도 (Accuracy) | 전체 중 맞춘 비율 |
| 정밀도 (Precision) | 긍정 예측 중 정답 비율 |
| 재현율 (Recall) | 실제 정답 중 맞춘 비율 |
| F1 점수 | 정밀도와 재현율의 조화 평균 |
| RMSE / MAE | 회귀 문제의 오차 지표 |

---

## 🧰 머신러닝 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| NumPy | 수치 계산 |
| pandas | 데이터프레임 처리 |
| matplotlib / seaborn | 데이터 시각화 |
| scikit-learn | 머신러닝 알고리즘 및 평가 도구 |
| XGBoost / LightGBM | 고성능 부스팅 모델 |

---

## 📚 참고

- [Scikit-learn 공식 문서](https://scikit-learn.org/)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---

> 작성자: [사용자 이름]  
> 날짜: 2025년 7월
