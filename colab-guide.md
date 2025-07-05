
# 🧪 Google Colab 사용법 정리 (2025년 최신)

> Google Colab은 브라우저에서 실행되는 Jupyter Notebook 환경으로, 파이썬 실습부터 데이터 분석, 머신러닝까지 손쉽게 실습할 수 있습니다.

---

## 📌 목차
1. [Colab 시작하기](#colab-시작하기)
2. [기본 조작 단축키](#기본-조작-단축키)
3. [런타임 환경 설정](#런타임-환경-설정)
4. [패키지 설치 명령어](#패키지-설치-명령어)
5. [Google Drive 연동](#google-drive-연동)
6. [파일 업로드 / 다운로드](#파일-업로드--다운로드)
7. [자주 발생하는 오류 해결](#자주-발생하는-오류-해결)
8. [기타 꿀팁](#기타-꿀팁)

---

## 📁 Colab 시작하기

1. [https://colab.research.google.com](https://colab.research.google.com) 접속
2. Google 계정 로그인
3. **[파일 → 새 노트북]** 클릭 후 파일명 변경
4. 노트북은 자동으로 Google Drive에 저장됨

---

## ⌨️ 기본 조작 단축키

| 기능            | 단축키          |
|-----------------|-----------------|
| 셀 실행         | `Shift + Enter` |
| 셀 추가 (아래)  | `Ctrl + M B`    |
| 셀 추가 (위)    | `Ctrl + M A`    |
| 셀 삭제         | `Ctrl + M D`    |

---

## ⚙️ 런타임 환경 설정

| 설정 항목           | 설명                                                                 |
|--------------------|----------------------------------------------------------------------|
| 런타임 유형 변경     | `런타임` 메뉴 → `런타임 유형 변경` 선택                                  |
| 하드웨어 가속기     | `None` (CPU), `GPU` (T4, A100), `TPU` (TPU v5e) 선택 가능               |
| Pro 플랜 활용 시 장점 | 더 빠른 GPU (A100/H100), 더 긴 세션 시간, 더 많은 RAM 사용 가능              |

---

## 📦 패키지 설치 명령어

| 목적                | 명령어 예시                                        |
|---------------------|----------------------------------------------------|
| 파이썬 패키지 설치   | `!pip install pandas numpy matplotlib`             |
| 리눅스 패키지 설치   | `!apt-get install -y graphviz`                     |
| requirements.txt 설치 | `!pip install -r requirements.txt`                 |

---

## 🗃 Google Drive 연동

| 항목        | 설명                                                                 |
|-------------|----------------------------------------------------------------------|
| 연동 코드    | `from google.colab import drive`<br>`drive.mount('/content/drive')` |
| 마운트 경로  | `/content/drive/MyDrive/` 를 통해 내 Google Drive에 접근 가능           |

---

## 📤 파일 업로드 / 다운로드

| 작업              | 방법 또는 명령어                                                        |
|-------------------|-------------------------------------------------------------------------|
| 로컬 파일 업로드   | Colab 왼쪽 → `파일` 탭 → `업로드` 버튼 클릭                                 |
| URL 다운로드       | `!wget [URL]`                                                           |
| Colab 파일 다운로드 | `files.download('파일명')`<br>→ `from google.colab import files` 필요 |

---

## ❗ 자주 발생하는 오류 해결

| 오류 메시지              | 해결 방법                                          |
|--------------------------|---------------------------------------------------|
| `CUDA out of memory`     | `torch.cuda.empty_cache()` 호출 후 배치 크기 줄이기 |
| 런타임 세션 끊김         | `런타임` → `다시 시작`                             |
| 한글 폰트 깨짐           | `!apt-get install -y fonts-nanum` 설치 후 재시작    |
| 설치 후 인식 안 됨       | `런타임` → `런타임 다시 시작`                      |

---

## 💡 기타 꿀팁

- 셀 내부에서 마크다운 작성: `텍스트 셀`을 선택해 `# 제목`, `**굵게**`, `- 리스트` 등 작성 가능
- 노트북 저장: `.ipynb` → `.py` / `.html` / `.pdf`로 변환 가능
- 공유: 오른쪽 상단의 `공유` 버튼 클릭 → 보기 / 편집 권한 설정
- GitHub 연동: Colab에서 GitHub 저장소로 바로 저장 가능

---

## 📎 참고 링크

- [Colab 공식 홈페이지](https://colab.research.google.com/)
- [Colab 도움말 (공식 FAQ)](https://research.google.com/colaboratory/faq.html)
- [Colab Pro 요금제 안내](https://colab.research.google.com/signup)

---

> 작성자: [사용자 이름]  
> 날짜: 2025년 7월
