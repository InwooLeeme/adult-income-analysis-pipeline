# Developer AI Income Analysis

Stack Overflow Developer Survey 2024 데이터를 기반으로 **"AI 도구에 우호적인 개발자는 연봉이 다른가?"** 를 검증한 데이터 분석 파이프라인입니다.

단순 시각화가 아니라 가설 설정, 전처리, EDA, 통계 검정, 효과크기 해석, RidgeCV 회귀 모델링, 교란변수 통제, 리포트 자동 생성, Plotly Dash 대시보드까지 하나의 재현 가능한 흐름으로 구성했습니다.

## 1. 문제 정의

AI 개발 도구가 빠르게 확산되면서, 개발자의 AI 수용 태도와 시장 가치 사이에 관계가 있는지 확인하고자 했습니다.

이 프로젝트의 핵심 질문은 다음과 같습니다.

> AI 도구에 우호적인 개발자는 그렇지 않은 개발자보다 연봉이 높은가?

분석에서는 Stack Overflow 2024 설문 데이터의 `AISent`를 AI 도구에 대한 태도, `ConvertedCompYearly`를 연봉 지표로 사용했습니다.

| 항목 | 내용 |
| --- | --- |
| 데이터 | Stack Overflow Developer Survey 2024 |
| 원본 표본 | 65,437건 |
| 분석 표본 | 17,535건 |
| 종속 변수 | `ConvertedCompYearly` |
| 핵심 독립 변수 | `AISent` |
| 주요 방법 | EDA, Welch's t-test, Cohen's d, RidgeCV 회귀, 교란변수 통제 |

## 2. 핵심 결과

분석 결과는 직관과 조금 달랐습니다.

> AI에 우호적인 개발자보다, AI에 회의적이거나 무관심한 개발자의 평균 연봉이 더 높은 경향이 관찰되었습니다.

| 비교 | AI 우호 그룹 | 그 외 그룹 |
| --- | ---: | ---: |
| 표본 수 | 12,717명 | 4,818명 |
| 평균 연봉 | $73,187 | $80,290 |

Welch's t-test 결과 p-value는 `1.979e-11`로 통계적으로 유의했습니다. 다만 Cohen's d는 `-0.114`로 매우 작아, **통계적으로는 차이가 있지만 실질적 차이는 크지 않다**고 해석했습니다.

다변량 RidgeCV 회귀에서도 국가, 경력, 직군, 학력, 조직 규모, 근무 형태, 연령을 함께 통제했습니다. 확장 모델의 R2는 `0.5920`이며, 국가와 직군 같은 요인이 연봉 차이에 크게 작용하는 것으로 나타났습니다.

## 3. 분석 방법

이 프로젝트는 아래 순서로 분석을 진행합니다.

| 단계 | 내용 |
| --- | --- |
| 0. 벤치마크 | Pandas와 Polars의 로딩, 필터, 집계 성능 비교 |
| 1. 로딩 | 두 라이브러리의 shape, 결측치 처리 결과 정합성 확인 |
| 2. 전처리 | 결측치 제거, 중복 제거, 경력 표기 정규화, 상위 1% 연봉 이상치 제거 |
| 3. EDA | 연봉, 경력, AI 태도, 국가 분포 확인 |
| 4. 시각화 | Seaborn 정적 차트와 Plotly 인터랙티브 차트 생성 |
| 5. 통계 분석 | 상관분석, Welch's t-test, 95% 신뢰구간, Cohen's d 계산 |
| 6. 모델링 | sklearn Pipeline 기반 RidgeCV 회귀 학습 및 평가 |
| 7. 심층 분석 | 국가, 경력 구간, 직군을 통제한 추가 검증 |
| 8. 리포트 생성 | 분석 결과를 `report.md`로 자동 조립 |

## 4. 기술적 구현

분석 로직은 단계별 모듈로 분리되어 있습니다.

```text
adult-income-analysis-pipeline/
├── data/
│   └── results.csv
├── outputs/
│   ├── charts/
│   └── models/
├── app.py
├── src/
│   ├── main.py
│   ├── config.py
│   ├── benchmark.py
│   ├── loading.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── viz.py
│   ├── stats.py
│   ├── model.py
│   ├── deep_dive.py
│   ├── report.py
│   ├── dashboard.py
│   ├── reporting.py
│   ├── formatting.py
│   └── results.py
├── tests/
├── pyproject.toml
├── requirements.txt
├── README.md
└── report.md
```

### 구현 포인트

- `src/main.py`: 0~8단계 분석 파이프라인 실행 진입점
- `src/preprocess.py`: 결측치, 중복, 경력 표기, 이상치 처리
- `src/stats.py`: Welch's t-test, Cohen's d, 신뢰구간 계산
- `src/model.py`: `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `RidgeCV` 기반 회귀 파이프라인
- `src/deep_dive.py`: 국가, 경력 구간, 직군을 통제한 심층 분석
- `src/report.py`: 실행 결과를 마크다운 보고서로 자동 생성
- `src/dashboard.py`: Dash 대시보드용 데이터 로딩, 필터링, KPI, Plotly 차트 생성
- `app.py`: Plotly Dash 대시보드 실행 진입점
- `tests/`: 전처리, 통계, 모델링, 산출물 생성, 보고서 렌더링, 대시보드 유틸 검증

## 5. 팀 프로젝트와 기여 포인트

이 프로젝트는 6인 팀 프로젝트로 진행했습니다. 포트폴리오 관점에서는 아래 구현 영역을 기술 기여로 강조할 수 있습니다.

| 영역 | 설명 |
| --- | --- |
| 분석 설계 | AI 태도와 연봉의 관계를 검증 가능한 가설로 정의 |
| 전처리 파이프라인 | 관심 컬럼 선별, 결측치 처리, 중복 제거, 경력 표기 정규화, 이상치 절단 |
| 통계 검정 | Welch's t-test, p-value, 95% 신뢰구간, Cohen's d를 함께 해석 |
| 회귀 모델링 | 수치형과 범주형 변수를 함께 처리하는 sklearn Pipeline 구성 |
| 교란변수 통제 | 국가, 경력, 직군, 학력, 조직 규모, 근무 형태, 연령을 고려 |
| 자동 리포팅 | 분석 결과와 차트, 모델 지표를 `report.md`로 자동 생성 |
| 인터랙티브 대시보드 | 필터와 Plotly 차트로 분석 결과를 탐색 가능하게 구성 |
| 테스트 | 주요 파이프라인 단계와 보고서, 대시보드 유틸을 pytest로 검증 |

## 6. 재현 방법

Python 3.12 이상을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

원본 데이터는 저장소에 포함하지 않습니다. [Stack Overflow Developer Survey 2024](https://github.com/StackExchange/Survey/tree/main/packages/archive/2024)에서 설문 결과 CSV를 내려받아 아래 경로에 저장합니다.

```text
adult-income-analysis-pipeline/
└── data/
    └── results.csv
```

기본 경로로 파이프라인을 실행하려면 다음 명령을 사용합니다.

```bash
python3 -m src.main
```

다른 CSV 경로를 사용하려면 `--data-path`를 지정합니다.

```bash
python3 -m src.main --data-path path/to/results.csv
```

실행이 끝나면 `report.md`, `outputs/charts/`, `outputs/models/`가 갱신됩니다.

## 7. Plotly Dash 대시보드

파이프라인으로 만든 분석 데이터를 브라우저에서 탐색할 수 있는 Dash 대시보드를 제공합니다.

```bash
python3 app.py
```

기본 주소는 `http://127.0.0.1:8050`입니다.

대시보드에서 제공하는 기능:

- 필터: 국가, 경력 구간, 직군, AI 태도
- KPI: 필터 적용 표본 수, 중위 연봉, 평균 연봉, AI 우호 그룹 평균 차이
- 차트: AI 태도별 연봉 분포, AI 태도별 중위 연봉, 국가별 평균 연봉, 회귀 계수 Top N

대시보드는 `data/results.csv`를 읽어 기존 전처리 규칙을 적용합니다. 데이터 파일이 없으면 앱 화면에서 준비 안내를 표시합니다.

## 8. 생성 산출물

| 산출물 | 설명 |
| --- | --- |
| `report.md` | 자동 생성 분석 보고서 |
| `outputs/charts/ai_sentiment_income_boxplot.png` | AI 태도별 연봉 분포 |
| `outputs/charts/country_ai_sentiment_income.html` | 국가별 AI 태도-연봉 비교 인터랙티브 차트 |
| `outputs/charts/experience_ai_sentiment_income.png` | 경력 구간별 AI 태도와 연봉 추이 |
| `outputs/charts/pandas_polars_benchmark.png` | Pandas vs Polars 성능 비교 |
| `outputs/models/ai_income_pipeline.joblib` | 기본 RidgeCV 회귀 모델 |
| `outputs/models/ai_income_pipeline_multivariate.joblib` | 확장 다변량 RidgeCV 회귀 모델 |

## 9. 검증

테스트는 `pytest`, 정적 검사는 `ruff`를 사용합니다.

```bash
pytest
ruff check .
ruff format .
```

`pyproject.toml`에는 `pytest` 테스트 경로와 `ruff` 규칙이 정의되어 있습니다.

커밋 전 자동 검사를 사용하려면 다음 명령으로 pre-commit hook을 활성화합니다.

```bash
pre-commit install
pre-commit run --all-files
```

## 10. 해석상 한계

이 분석은 관찰 데이터 기반이므로 인과관계를 주장하지 않습니다.

- Stack Overflow 설문 응답자는 전체 개발자를 완전히 대표하지 않을 수 있습니다.
- 국가별 임금 수준, 물가, 고용 시장 차이가 연봉에 큰 영향을 줍니다.
- 경력, 연령, 직군, 조직 규모, 근무 형태 같은 교란변수가 남아 있을 수 있습니다.
- `AISent`는 AI 도구 사용량이 아니라 AI에 대한 태도이므로 실제 생산성이나 숙련도를 직접 측정하지 않습니다.
- 표본이 크기 때문에 작은 차이도 p-value상 유의하게 나올 수 있어, 효과크기와 함께 해석해야 합니다.

## 11. 다음 개선 방향

- 배포용 요약 데이터 파일 생성
- Render 또는 Plotly Cloud 배포 설정 추가
- 국가별 임금 수준을 보정한 추가 분석
- 회귀 계수 Top N을 더 세부적으로 탐색하는 UI 추가
- 포트폴리오 사이트에 분석 글로 분리

블로그 글 제목 예시는 다음과 같습니다.

> AI에 호의적인 개발자는 정말 더 많이 벌까? Stack Overflow 2024 데이터 분석
