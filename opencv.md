
# 📷 OpenCV 기본 개념 정리 (쉬운 설명)

> OpenCV(Open Source Computer Vision)는 컴퓨터 비전(영상 처리 및 인식)을 위한 오픈소스 라이브러리입니다. 이미지/영상 처리, 객체 탐지, 얼굴 인식 등 다양한 기능을 제공합니다.

---

## 📌 OpenCV란?

- Open Source Computer Vision의 약자
- 영상 및 이미지 처리에 특화된 C++ 기반 라이브러리 (Python에서도 사용 가능)
- 실시간 영상 분석, 필터링, 객체 탐지, 얼굴 인식, 딥러닝 모델 연동 등 가능
- 크로스 플랫폼: Windows, macOS, Linux, Android, iOS 등 지원

---

## 📥 설치 방법

```bash
pip install opencv-python
```

> 영상 파일 처리, 웹캠 사용 등을 위해 `opencv-python-headless`가 아닌 `opencv-python` 설치 권장

---

## 📂 주요 기능 요약

| 기능 | 설명 |
|------|------|
| 이미지/비디오 불러오기 | `cv2.imread()`, `cv2.VideoCapture()` 등 |
| 이미지 저장 | `cv2.imwrite()` |
| 색상 변환 | RGB ↔ BGR, 그레이스케일, HSV 등 (`cv2.cvtColor()`) |
| 그리기 함수 | 선, 사각형, 원, 텍스트 등 (`cv2.line()`, `cv2.putText()`) |
| 이미지 필터링 | 블러, 경계 검출, 히스토그램 등 |
| 객체 탐지 | 얼굴, 눈, 손 등 탐지 (Haar Cascades, 딥러닝) |
| 윤곽선 찾기 | `cv2.findContours()` |
| 영상 스트리밍 | 웹캠, 실시간 프레임 처리 |
| 딥러닝 모델 연동 | TensorFlow, ONNX 등과 함께 사용 가능 |

---

## 🧱 이미지 구조

- OpenCV에서 이미지는 **NumPy 배열 형태**로 표현됩니다.
  - 컬러 이미지: `(높이, 너비, 채널)` → 채널 순서는 **BGR**
  - 그레이스케일: `(높이, 너비)`
- 이미지 값은 기본적으로 0~255 (uint8 타입)

---

## 🖼 이미지 관련 함수

| 함수 | 설명 |
|------|------|
| `cv2.imread()` | 이미지 파일 불러오기 |
| `cv2.imshow()` | 이미지 창에 띄우기 |
| `cv2.imwrite()` | 이미지 저장 |
| `cv2.resize()` | 이미지 크기 변경 |
| `cv2.cvtColor()` | 색상 공간 변환 |
| `cv2.flip()` | 이미지 반전 (좌우, 상하) |

---

## 🎥 영상 관련 함수

| 함수 | 설명 |
|------|------|
| `cv2.VideoCapture()` | 비디오 파일 또는 웹캠 불러오기 |
| `cv2.VideoWriter()` | 비디오 저장용 객체 생성 |
| `cap.read()` | 프레임 읽기 |
| `cv2.waitKey()` | 키 입력 대기 |
| `cv2.destroyAllWindows()` | 모든 창 닫기 |

---

## 🔍 기본 이미지 처리 기능

- **블러링 (Blur)**: 노이즈 제거  
  `cv2.GaussianBlur(), cv2.medianBlur()`

- **엣지 검출**: 윤곽선 추출  
  `cv2.Canny()`

- **이진화 (Thresholding)**: 이미지 이진 처리  
  `cv2.threshold(), cv2.adaptiveThreshold()`

- **모폴로지(Morphology)**: 침식, 팽창 등  
  `cv2.erode(), cv2.dilate()`

---

## 🧠 얼굴 인식 (Haar Cascade)

- 사전 학습된 모델로 얼굴, 눈, 몸 등 인식 가능
- XML 파일로 제공 (`haarcascade_frontalface_default.xml`)
- `cv2.CascadeClassifier()`를 사용하여 탐지

---

## 🤖 딥러닝과 OpenCV

- ONNX, TensorFlow, Caffe 모델을 로딩하여 추론 가능
- `cv2.dnn.readNetFromONNX()` 등 사용
- YOLO, MobileNet 등의 객체 탐지 모델과 연동 가능

---

## 📚 참고 자료

- [OpenCV 공식 사이트](https://opencv.org/)
- [OpenCV-Python 튜토리얼](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [OpenCV GitHub](https://github.com/opencv/opencv)

---

> 작성자: [사용자 이름]  
> 날짜: 2025년 7월

