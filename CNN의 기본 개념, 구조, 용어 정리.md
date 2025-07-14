
# 🧠 Convolutional Neural Network (CNN) 기초 가이드

> **이미지 처리의 핵심 알고리즘인 CNN을 이해하기 위한 개념과 용어를 쉽게 정리한 문서입니다.**  
> 이미지 분류, 객체 탐지, 자율주행 등에서 CNN은 필수 기술입니다.

---

## 🚀 1. CNN이란?

- **CNN (Convolutional Neural Network)** 는 이미지, 영상 등 **공간적 구조를 갖는 데이터 처리에 특화된 딥러닝 모델**입니다.
- 이미지의 **패턴, 윤곽, 형태** 등을 자동으로 학습합니다.
- 사용 분야: 이미지 분류, 얼굴 인식, 교통 표지판 인식, 의료 영상 진단 등

---

## 🧱 2. CNN 기본 구조

```text
입력 이미지
   ↓
[Convolution → Activation → Pooling] × N
   ↓
Flatten
   ↓
Fully Connected Layer (Dense)
   ↓
Softmax (분류 결과)
```

- ✅ **Convolution Layer**: 이미지 특징 추출 (필터 적용)
- ✅ **Activation Function**: 비선형성 추가 (주로 ReLU)
- ✅ **Pooling Layer**: 이미지 크기 축소 (주로 MaxPooling)
- ✅ **Flatten**: 2D 이미지 → 1D 벡터로 변환
- ✅ **Fully Connected**: 최종 예측 결정
- ✅ **Dropout / BatchNorm**: 과적합 방지 및 학습 안정화

---


## 📘 3. 핵심 용어 정리

| 용어 | 설명 |
|------|------|
| **Kernel / Filter** | 이미지에서 특징을 추출하는 작은 행렬 |
| **Stride** | 필터가 이동하는 간격 |
| **Padding** | 테두리 보존을 위한 여백 추가 |
| **Feature Map** | 합성곱 연산 결과물 |
| **ReLU** | 음수를 0으로 만드는 활성화 함수 |
| **Pooling** | 특징 압축 (Max, Avg) |
| **Flatten** | 다차원 데이터를 1D 벡터로 변환 |
| **Epoch** | 전체 데이터셋을 1회 학습 |
| **Batch Size** | 한 번에 학습하는 샘플 수 |
| **Loss Function** | 예측값과 실제값의 차이 계산 |
| **Optimizer** | 파라미터 업데이트 방식 (SGD, Adam) |
| **Backpropagation** | 오차 역전파로 학습 진행 |
| **Regularization** | 과적합 방지 기법 (Dropout 등) |


| 용어 | 설명 |
|------|------|
| **Kernel / Filter** | 이미지에서 특징을 추출하는 작은 행렬 |
| **Stride** | 필터가 이동하는 간격 |
| **Padding** | 테두리 보존을 위한 여백 추가 |
| **Feature Map** | 합성곱 연산 결과물 |
| **ReLU** | 음수를 0으로 만드는 활성화 함수 |
| **Pooling** | 특징 압축 (Max, Avg) |
| **Flatten** | 다차원 데이터를 1D 벡터로 변환 |
| **Epoch** | 전체 데이터셋을 1회 학습 |
| **Batch Size** | 한 번에 학습하는 샘플 수 |
| **Loss Function** | 예측값과 실제값의 차이 계산 |
| **Optimizer** | 파라미터 업데이트 방식 (SGD, Adam) |
| **Backpropagation** | 오차 역전파로 학습 진행 |
| **Regularization** | 과적합 방지 기법 (Dropout 등) |

---

## 🛠️ 4. 간단한 CNN 모델 예시 (PyTorch)

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
```

---

## 🧪 5. CNN 주요 활용 분야

✅ 이미지 분류 (MNIST, CIFAR, ImageNet)  
✅ 객체 탐지 (YOLO, SSD, R-CNN)  
✅ 자율주행 (차선 인식, 교통 표지판 인식)  
✅ 얼굴 인식 및 감정 분석  
✅ 의료 영상 진단 (X-ray, MRI, CT)

---

## 🧰 6. 자주 사용하는 라이브러리

| 라이브러리 | 기능 |
|------------|------|
| `torch`, `torchvision` | PyTorch 기반 모델 정의 및 학습 |
| `tensorflow`, `keras` | 고수준 딥러닝 API |
| `numpy`, `matplotlib` | 수치 연산 및 시각화 |
| `scikit-learn` | 모델 평가, 지표 계산 등 |

---

## 📚 7. 참고 자료

- [📘 Stanford CS231n 강의](https://cs231n.github.io/convolutional-networks/)
- [🧪 PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [📖 Keras CNN 예제](https://keras.io/examples/vision/)

---

> ⏱️ 작성일: 2025년 7월  
> ✍️ 정리자: [사용자 이름]

