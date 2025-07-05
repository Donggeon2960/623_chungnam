좋아, GitHub에 올릴 수 있게 **Markdown (.md)** 형식으로 정리해줄게. 아래 내용을 그대로 복사해서 `README.md`나 다른 `.md` 파일에 붙여넣으면 깔끔하게 정리돼.

---

````markdown
# Google Colab 사용법 정리 (2025년 최신)

> Google Colab은 웹 브라우저에서 실행되는 Jupyter Notebook 환경으로, 파이썬 실습부터 머신러닝까지 모두 가능한 무료 플랫폼입니다.

---

## 1. 시작하기

1. [https://colab.research.google.com](https://colab.research.google.com) 접속
2. Google 계정 로그인
3. **[파일 → 새 노트북]** 클릭 → 파일명 변경
4. 노트북은 자동으로 Google Drive에 저장됨

---

## 2. 기본 조작 단축키

| 기능 | 단축키 |
|------|--------|
| 셀 실행 | `Shift + Enter` |
| 셀 추가 (아래) | `Ctrl + M B` |
| 셀 추가 (위) | `Ctrl + M A` |
| 셀 삭제 | `Ctrl + M D` |

---

## 3. 하드웨어 설정 (GPU/TPU)

1. 상단 메뉴: **[런타임 → 런타임 유형 변경]**
2. 하드웨어 가속기 선택:
   - None = CPU
   - GPU = NVIDIA T4 / A100 (Pro)
   - TPU = TPU v5e (대형 모델용)

> 💡 Pro+ 사용 시 더 긴 세션 시간 및 H100 GPU 사용 가능

---

## 4. 패키지 설치 & 시스템 명령

```python
!pip install numpy pandas matplotlib
!apt-get update && apt-get install -y graphviz
````

* `!`로 시작하는 명령어는 터미널 명령어
* `.py` 코드 외에도 셸 명령어를 직접 실행 가능

---

## 5. Google Drive 연동

```python
from google.colab import drive
drive.mount('/content/drive')
```

* 처음 실행 시 권한 승인 필요
* 이후 `drive/MyDrive/` 경로로 내 파일 접근 가능

---

## 6. 파일 다루기

| 작업             | 방법                                |
| -------------- | --------------------------------- |
| 로컬 파일 업로드      | 왼쪽 사이드바 → 파일 → 업로드                |
| 외부 URL 파일 다운로드 | `!wget [URL]`                     |
| `.csv` 불러오기 예시 | `pd.read_csv('/content/파일명.csv')` |

---

## 7. 결과 저장 & 공유

* **파일 → 다운로드 → .ipynb / .py / .html / .pdf** 등으로 저장 가능
* **GitHub 연동**: Colab에서 바로 `.ipynb`를 GitHub에 저장 가능
* **공유**: 상단의 \[공유] 버튼을 눌러 보기/수정 권한 부여

---

## 8. 유용한 기능 (2025년 기준)

* **Gemini 2.5 Flash 통합**: 코드 설명, 오류 수정, 테스트 생성 등 AI 도우미
* **Agentic Workflow Recorder**: 노트북 실행 과정을 파이프라인 형태로 변환
* **배터리 아이콘**: 세션 남은 시간 시각화

---

## 9. 자주 발생하는 오류 해결

| 오류 메시지               | 해결 방법                                      |
| -------------------- | ------------------------------------------ |
| `CUDA out of memory` | `torch.cuda.empty_cache()` 호출 후 배치 크기 줄이기  |
| 런타임 끊김               | 런타임 → 다시 시작                                |
| 한글 깨짐                | `!apt-get install -y fonts-nanum` 설치 후 재시작 |

---

## 10. 치트시트 요약

```python
# ▶ 셀 실행
Shift + Enter

# ▶ GPU 활성화
런타임 → 런타임 유형 변경 → GPU 선택

# ▶ Drive 연동
from google.colab import drive
drive.mount('/content/drive')

# ▶ 패키지 설치
!pip install 패키지명
```

---

## 📌 참고 링크

* [공식 Colab 도움말](https://research.google.com/colaboratory/faq.html)
* [Colab Pro 요금제 안내](https://colab.research.google.com/signup)
* [Colab GitHub 연동 가이드](https://colab.research.google.com/github)

---

```

---

필요하다면 `.ipynb` 파일도 같이 업로드할 수 있도록 구조까지 잡아줄 수 있어. 혹시 너가 쓰고자 하는 주제나 목차가 따로 있으면 알려줘서 더 맞춤형으로 만들어줄게.
```

