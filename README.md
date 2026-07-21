# Adult Income Analysis Pipeline

Stack Overflow Developer Survey 2024 데이터를 사용해 **AI 도구에 대한 태도(`AISent`)와 개발자 연봉(`ConvertedCompYearly`)의 관계**를 분석하는 End-to-End 데이터 분석 파이프라인입니다.

## 분석 주제

"AI 도구에 대한 태도가 개발자 연봉과 어떤 관계를 갖는가?"

- 귀무가설 H₀: AI 도구에 우호적인 그룹과 그 외 그룹의 평균 연봉은 같다.
- 대립가설 H₁: 두 그룹의 평균 연봉은 다르다. (양측 검정, α = 0.05)
- 국가·경력·직군 등 교란변수를 통제한 뒤에도 관계가 유지되는지 심층 분석
- 학력·조직규모·근무형태·연령까지 포함한 다변량 회귀로 AI 태도의 순효과 확인

## 프로젝트 구조

```text
adult-income-analysis-pipeline/
├── data/
│   └── results.csv          # 원본 설문 데이터 (직접 내려받아 배치, git에는 포함 안 됨)
├── outputs/
│   ├── charts/               # 실행 시 자동 생성되는 차트
│   └── models/                # 실행 시 자동 생성되는 학습된 모델(.joblib)
├── src/
│   ├── main.py                # 진입점 — 0~8단계를 순서대로 실행
│   ├── config.py              # 경로·컬럼·상수 정의
│   ├── benchmark.py           # 0단계: Pandas vs Polars 성능 벤치마크
│   ├── loading.py             # 1단계: 로딩 결과 정합성 비교
│   ├── preprocess.py          # 2단계: 결측치·중복·이상치 처리
│   ├── eda.py                 # 3단계: 탐색적 데이터 분석
│   ├── viz.py                 # 4단계: 차트 생성 (Seaborn·Plotly)
│   ├── stats.py               # 5단계: 상관분석·t-검정
│   ├── model.py                # 6단계: ML Pipeline·다변량 회귀
│   ├── deep_dive.py           # 7단계: 교란변수 통제 심층 분석
│   ├── report.py              # 8단계: report.md 자동 생성
│   ├── reporting.py           # 콘솔 진행상황 출력 (Reporter)
│   ├── formatting.py          # 통계 해석 문구·마크다운 표 헬퍼
│   └── results.py             # 단계별 결과를 담는 dataclass 모음
├── tests/
├── pyproject.toml             # ruff·pytest 설정
├── .pre-commit-config.yaml    # 커밋 전 자동 검사(ruff·pytest) 설정
├── requirements.txt
├── README.md
└── report.md                  # 실행 후 자동 생성되는 분석 보고서
```

## 데이터 컬럼

원본 설문은 114개 컬럼이지만, 분석에는 아래 컬럼만 사용합니다.

```python
[
    "ConvertedCompYearly",  # 종속 변수: 연봉(USD)
    "YearsCodePro", "YearsCode",  # 전문/전체 코딩 경력
    "AISent",     # 핵심 독립 변수: AI 도구에 대한 태도 (6단계)
    "Country", "DevType",  # 국가, 직군
    "EdLevel", "OrgSize", "RemoteWork", "Age",  # 다변량 분석용 통제 변수
]
```

## 1. 설치

Python 3.12 이상을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 데이터 준비

이 프로젝트는 원본 CSV를 자동으로 내려받지 않습니다. 아래 순서로 직접 준비해야 합니다.

1. [Stack Overflow Developer Survey 2024](https://github.com/StackExchange/Survey/tree/main/packages/archive/2024) 저장소에서 설문 결과 CSV(`results.csv`)를 내려받습니다.
2. 프로젝트 루트에 `data/` 폴더를 만들고 그 안에 `results.csv`로 저장합니다.

```text
adult-income-analysis-pipeline/
└── data/
    └── results.csv
```

다른 경로/파일명을 쓰고 싶다면 실행 시 `--data-path`로 지정하면 됩니다(아래 3번 참고).

> 참고: 이 CSV는 결측치를 빈 칸이 아니라 문자열 `"NA"`로 표기합니다. Pandas는 자동으로 인식하지만 Polars는 `null_values`를 명시해야 하며, 파이프라인이 이를 자동으로 처리합니다.

## 3. 실행

```bash
python3 -m src.main
```

기본 데이터 경로(`data/results.csv`)가 아닌 다른 파일을 쓰고 싶다면:

```bash
python3 -m src.main --data-path 원하는/경로.csv
```

실행하면 콘솔에 0~8단계 진행 상황이 출력되고, 마지막에 `report.md`가 갱신됩니다.

| 단계 | 내용 |
| --- | --- |
| 0. 벤치마크 | Pandas vs Polars를 로딩·필터·집계 3개 작업으로 비교 |
| 1. 로딩 | 두 라이브러리로 읽은 결과(shape·결측치)가 일치하는지 비교 |
| 2. 전처리 | 결측치·중복·이상치 제거, 경력 표기 정규화 |
| 3. EDA | 기술통계 및 AI 태도·국가별 분포 |
| 4. 시각화 | Seaborn 정적 차트 + Plotly 인터랙티브 차트 |
| 5. 통계 분석 | 상관계수, Welch's t-test, 효과크기(Cohen's d) |
| 6. ML Pipeline | RidgeCV 회귀 학습·평가, 다변량 결정요인 분석 |
| 7. 심층 분석 | 국가·경력·직군을 통제한 교차 검증 |
| 8. 보고서 생성 | 위 결과를 `report.md`로 자동 조립 |

## 4. 생성 산출물

- `report.md`: 자동 생성 분석 보고서 (실행할 때마다 갱신됨)
- `outputs/charts/ai_sentiment_income_boxplot.png`: AI 태도별 연봉 분포 (Seaborn)
- `outputs/charts/country_ai_sentiment_income.html`: 국가별 AI 태도-연봉 비교 (Plotly, 인터랙티브)
- `outputs/charts/experience_ai_sentiment_income.png`: 경력 구간별 AI 태도-연봉 추이
- `outputs/charts/pandas_polars_benchmark.png`: Pandas vs Polars 성능 비교
- `outputs/models/ai_income_pipeline.joblib`: 기본 모델(수치형+범주형 5변수)
- `outputs/models/ai_income_pipeline_multivariate.joblib`: 확장 모델(학력·조직규모·근무형태·연령 포함 9변수)

## 5. 테스트

```bash
pytest
```

`pyproject.toml`에 `testpaths = ["tests"]`가 설정되어 있어 위 명령 하나로 `tests/` 아래 전체가 실행됩니다.

## 6. 코드 스타일 (ruff)

```bash
ruff check .      # 린트 검사
ruff check --fix . # 자동 수정 가능한 항목 고치기
ruff format .      # 포매팅
```

## 7. 커밋 전 자동 검사 (pre-commit)

커밋할 때마다 ruff(lint·format)와 pytest가 자동으로 실행되도록 설정되어 있습니다. 새로 clone한 환경에서는 아래 명령으로 한 번 활성화해야 합니다.

```bash
pre-commit install
```

이후 `git commit` 시 검사가 실패하면 커밋이 중단됩니다. 수동으로 전체 파일에 대해 미리 검사해보려면:

```bash
pre-commit run --all-files
```
