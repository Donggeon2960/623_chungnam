
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


# 🧠 7. CNN 처리 결과 예시

> CNN(합성곱 신경망)은 이미지를 여러 필터로 처리하여 **엣지, 블러, 샤프닝 등 다양한 특징(feature)** 을 추출합니다.  
> 아래 이미지는 5x5 입력에 대해 CNN이 처리한 다양한 결과를 시각적으로 보여줍니다.

![CNN 처리 결과](/mnt/data/cnn_processing_result.png)

---

### 🔍 8. 처리 결과 설명

| 처리 단계 | 설명 |
|-----------|------|
| **원본 5×5 입력** | 5×5 크기의 입력 이미지. 각 숫자는 픽셀의 밝기(명암도)를 의미합니다. |
| **수직 엣지 감지** | 수직 방향의 경계를 강조합니다. 필터가 세로 방향의 픽셀 차이를 감지합니다. |
| **수평 엣지 감지** | 수평 방향의 경계를 강조합니다. 가로 방향 특징을 강조할 때 유용합니다. |
| **블러 처리** | 주변 픽셀의 평균값을 적용하여 부드러운 이미지를 생성합니다. 노이즈 제거에 효과적입니다. |
| **샤프닝 처리** | 경계를 더 뚜렷하게 만들어 이미지의 윤곽을 강조합니다. 흐릿한 이미지에 명확도를 부여합니다. |

---

> CNN은 여러 층에서 다양한 필터를 적용하며, 이러한 단계별 특징 추출을 통해 이미지의 본질적인 구조를 이해하고 분류하는 데 사용됩니다.

https://claude.ai/public/artifacts/2c09bc56-7cc3-4ea0-b3ca-7678aa107756
<img width="1518" height="485" alt="cnn_processing_result (1)" src="https://github.com/user-attachments/assets/912c56c0-4230-422c-9d7b-a4c47f4d1fb4" />

## 📚 9. 참고 자료

- [📘 Stanford CS231n 강의](https://cs231n.github.io/convolutional-networks/)
- [🧪 PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [📖 Keras CNN 예제](https://keras.io/examples/vision/)

---

> ⏱️ 작성일: 2025년 7월  
> ✍️ 정리자: [사용자 이름]

