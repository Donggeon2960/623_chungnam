
# 📊 Matplotlib 기본 개념 정리

> Matplotlib은 파이썬에서 데이터를 시각화할 수 있게 해주는 가장 대표적인 그래프 시각화 라이브러리입니다.

---

## 📌 Matplotlib이란?

- 파이썬에서 데이터 시각화를 위한 라이브러리
- 라인 차트, 바 차트, 산점도, 파이 차트 등 다양한 그래프 지원
- 하위 모듈로 `pyplot`을 자주 사용 (`import matplotlib.pyplot as plt`)
- pandas, numpy와 함께 사용하기 좋음

---

## 🖼 기본 구성

| 개념 | 설명 |
|------|------|
| Figure | 전체 그래프 영역 (캔버스) |
| Axes | 실제 그래프가 그려지는 영역 (하위 플롯) |
| Axis | 축 (x축, y축) |
| Artist | 그래프의 모든 요소 (라인, 텍스트 등) |

---

## 📈 주요 그래프 종류

| 함수 | 설명 |
|------|------|
| `plot()` | 선 그래프 |
| `bar()` | 막대 그래프 |
| `scatter()` | 산점도 |
| `pie()` | 원형 차트 |
| `hist()` | 히스토그램 |
| `boxplot()` | 박스플롯 |
| `imshow()` | 이미지 데이터 시각화 |
| `stackplot()` | 누적 영역 그래프 |

---

## 🎨 그래프 꾸미기

| 항목 | 설명 |
|------|------|
| `title()` | 제목 추가 |
| `xlabel()`, `ylabel()` | x, y축 레이블 추가 |
| `legend()` | 범례 추가 |
| `grid()` | 그리드 표시 |
| `xlim()`, `ylim()` | 축 범위 설정 |
| `style` | `'ro--'`, `'g^'` 등 색상/마커/선 스타일 지정 |

---

## 🔁 서브플롯

- `plt.subplot(rows, cols, index)`를 사용해 하나의 Figure에 여러 그래프 배치
- 또는 `fig, ax = plt.subplots()` 방식으로 객체 기반 사용

---

## 🗂 저장 및 출력

| 함수 | 설명 |
|------|------|
| `show()` | 그래프 화면에 출력 |
| `savefig('파일명.png')` | 이미지로 저장 (PDF, SVG 등 지원) |

---

## ⚙ 설정 관련 기능

| 항목 | 설명 |
|------|------|
| 한글 폰트 설정 | `plt.rcParams['font.family'] = 'Malgun Gothic'` 등 |
| DPI 설정 | `plt.figure(dpi=100)` 등 고해상도 설정 |
| 색상 지정 | `'red'`, `'#FF0000'`, `c='blue'` 등 다양하게 지정 가능 |

---

## 📚 참고

- [Matplotlib 공식 사이트](https://matplotlib.org/)
- [Gallery (예제 모음)](https://matplotlib.org/stable/gallery/index.html)

---

> 작성자: [사용자 이름]  
> 날짜: 2025년 7월
