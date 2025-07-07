
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

OpenCV에서 자주 등장하는 개념, 함수, 연산, 데이터 구조들을 정리한 표입니다. 초보자부터 중급자까지 참고하기에 좋은 용어 모음입니다.

---

| 용어 | 설명 |
|------|------|
| **Mat** | 이미지 데이터를 담는 OpenCV 기본 객체. 다차원 배열(matrix) 형태로 픽셀 값을 저장함. |
| **BGR / RGB** | OpenCV 기본 컬러 순서는 BGR(Blue, Green, Red). Matplotlib 등은 RGB 순서를 사용함. |
| **ROI** | Region Of Interest. 이미지에서 관심 있는 부분(영역)을 잘라내어 처리할 때 사용함. |
| **픽셀(Pixel)** | 영상의 최소 단위. Mat 객체의 각 요소(element)에 해당하며, 그레이스케일은 1채널, 컬러는 3채널(BGR)로 표현됨. |
| **채널(Channel)** | 각 픽셀이 갖는 값의 축. 그레이스케일=1채널, 컬러=3채널(B, G, R). |
| **스레숄드(Threshold)** | 임계값 기준으로 이진화(흑/백) 이미지로 변환함. 예: `cv2.threshold()` |
| **가우시안 블러(Gaussian Blur)** | 노이즈 제거를 위해 사용. 커널 크기와 표준편차로 설정하며 부드러운 흐림 효과를 줌. |
| **모폴로지 연산(Morphology)** | 침식(erode), 팽창(dilate), 열기(open), 닫기(close) 등을 통해 이미지의 형태를 유지하며 잡음 제거. |
| **엣지 검출(Edge Detection)** | `cv2.Canny()`, Sobel, Laplacian 등을 이용해 경계를 검출하는 알고리즘. |
| **컨투어(Contour)** | `cv2.findContours()`로 객체의 외곽선을 찾음. 윤곽선 추적 및 분석에 사용됨. |
| **히스토그램(Histogram)** | 픽셀 값의 분포를 나타내는 그래프. 밝기/명암 대비 등을 분석하는 데 사용. |
| **계조 향상(Equalization)** | `cv2.equalizeHist()`를 통해 명암 대비를 개선하는 기법. |
| **컬러 공간 변환** | `cv2.cvtColor()`로 RGB↔GRAY, BGR↔HSV 등 색상 기반 변환에 사용. |
| **기하 변환(Geometric Transform)** | `cv2.warpAffine()`, `cv2.warpPerspective()` 등으로 이미지 회전, 이동, 크기 변환 수행. |
| **특징 검출/매칭** | SIFT, SURF, ORB, AKAZE 등 알고리즘으로 키포인트와 디스크립터를 추출하여 유사한 이미지를 비교. |
| **카메라 보정** | 왜곡된 카메라 이미지를 보정하고, 내부 파라미터 추정. chessboard 패턴을 통해 캘리브레이션 수행. |
| **비디오 캡처(VideoCapture)** | `cv2.VideoCapture()`로 웹캠/비디오 파일을 읽음. `read()`로 프레임을 가져올 수 있음. |
| **윈도우(Window)** | `cv2.namedWindow()`, `cv2.imshow()`, `cv2.waitKey()`, `cv2.destroyAllWindows()` 등으로 이미지 창 관리. |

---

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
