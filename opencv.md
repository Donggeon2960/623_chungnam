
# 📷 OpenCV 개념 + 필수 라이브러리 + 용어 정리 (2025년 기준)

> OpenCV(Open Source Computer Vision)는 이미지 및 영상 처리, 분석, 컴퓨터 비전 기술을 구현하는 데 널리 사용되는 오픈소스 라이브러리입니다.  
> 이 문서는 OpenCV의 개념, 함께 쓰는 라이브러리, 자주 나오는 용어를 초보자도 쉽게 이해할 수 있도록 정리했습니다.

---

## 1️⃣ OpenCV란?

- **Open Source Computer Vision Library**의 약자
- 이미지와 동영상을 **읽고, 처리하고, 분석**하는 기능 제공
- 원래 C++로 개발되었으나, **Python API**도 매우 강력함
- 실시간 영상 처리, 얼굴 인식, 물체 추적, 엣지 검출, 딥러닝 모델 연동 등 다양한 기능 가능
- Windows, macOS, Linux, Android 등 대부분의 플랫폼 지원

### 주요 특징
- 무료 오픈소스
- 실시간 이미지 처리
- 딥러닝 모델도 연동 가능 (`cv2.dnn`)
- 다양한 알고리즘 내장: 필터링, 특징 추출, 윤곽선 탐지 등

---

## 2️⃣ OpenCV에 필요한 주요 라이브러리

| 라이브러리 | 설명 | 설치 명령어 |
|-----------|------|--------------|
| `opencv-python` | OpenCV의 Python 버전 | `pip install opencv-python` |
| `numpy` | OpenCV는 이미지를 NumPy 배열로 처리함 | `pip install numpy` |
| `matplotlib` | 이미지 출력 및 시각화에 사용 | `pip install matplotlib` |
| `opencv-contrib-python` | 추가적인 모듈들 (e.g. SIFT, SURF 등) 포함 | `pip install opencv-contrib-python` |
| `Pillow` | 이미지 저장, 포맷 변환 시 보조 용도로 사용 | `pip install pillow` |

> `opencv-python-headless`는 GUI 없는 서버 환경용

---

## 3️⃣ OpenCV 자주 나오는 용어 정리

| 용어 | 의미 |
|------|------|
| `cv2` | OpenCV의 Python 인터페이스 모듈 이름 (`import cv2`) |
| BGR | OpenCV의 기본 색상 순서 (Blue-Green-Red) |
| RGB | 일반적인 색상 순서 (Red-Green-Blue, matplotlib에서 사용) |
| Grayscale | 흑백 이미지 (1채널, 0~255 값) |
| Threshold | 임계값 이진화 처리 (흑/백 이미지) |
| Blurring | 이미지의 노이즈 제거 또는 흐리게 만드는 작업 |
| Canny Edge | 엣지(경계선)를 검출하는 알고리즘 |
| Haar Cascade | 얼굴, 눈 등을 인식하기 위한 기계학습 기반 탐지기 |
| Contour | 윤곽선 추출 결과, 객체 경계를 정의하는 선 |
| ROI (Region of Interest) | 이미지에서 특정 부분만 잘라서 사용하는 영역 |
| Kernel | 필터링에 사용하는 작은 행렬 (예: 블러 필터 등) |
| DNN | OpenCV에서 딥러닝 추론을 지원하는 모듈 (`cv2.dnn`) |
| VideoCapture | 비디오/웹캠 스트리밍을 처리하는 객체 |
| waitKey | 키보드 입력 대기 (영상 출력 루프에서 자주 사용됨) |
| imread / imshow / imwrite | 이미지 불러오기, 보기, 저장 함수 |

---

## 📝 정리 요약

- OpenCV는 영상 처리, 분석, 인식 기술에 매우 유용한 필수 라이브러리입니다.
- NumPy와 함께 사용하며, 다양한 보조 라이브러리와 함께 시각화, 모델 연동 등도 가능합니다.
- 용어를 정확히 알고 있는 것이 이미지 처리의 핵심 이해에 큰 도움이 됩니다.

---

## 📚 참고 링크

- [OpenCV 공식 홈페이지](https://opencv.org/)
- [OpenCV-Python 튜토리얼](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [OpenCV GitHub](https://github.com/opencv/opencv)

---

> 작성자: [사용자 이름]  
> 날짜: 2025년 7월
