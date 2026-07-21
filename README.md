# Adult Income Analysis Pipeline

Adult Income 데이터셋을 사용해 **주당 근무시간(`hours-per-week`)과 고소득 여부(`income`)의 관계**를 분석하는 End-to-End 데이터 분석 프로젝트입니다.

## 분석 주제

주당 근무시간이 고소득 여부와 관련이 있는지 확인합니다.

- `<=50K` 그룹과 `>50K` 그룹의 주당 근무시간 분포 비교
- 두 그룹의 평균 근무시간 차이에 대한 t-test 수행
- 인구통계, 교육, 직업, 근무시간 변수를 활용한 소득 예측 모델 구성

## 프로젝트 구조

```text
adult-income-analysis-pipeline/
├── data/
│   └── adult.csv
├── outputs/
│   ├── charts/
│   ├── models/
│   └── metrics.json
├── src/
│   ├── main.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── stats_analysis.py
│   ├── modeling.py
│   └── report.py
├── tests/
├── requirements.txt
├── README.md
└── report.md
```

## 데이터 컬럼

```python
[
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]
```

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

Adult Income CSV 파일을 아래 경로에 저장합니다.

```text
data/adult.csv
```

헤더가 없는 원본 Adult 데이터 형식과 헤더가 있는 CSV 형식을 모두 지원합니다.

## 실행

```bash
python3 -m src.main --data-path data/adult.csv
```

## 생성 산출물

- `report.md`: 자동 생성 분석 리포트
- `outputs/descriptive_statistics.csv`: 기술통계
- `outputs/hours_summary.csv`: 소득 그룹별 주당 근무시간 요약
- `outputs/correlation_matrix.csv`: 수치형 변수 상관계수
- `outputs/metrics.json`: 로딩 비교, t-test, 모델 평가 결과
- `outputs/charts/hours_by_income_boxplot.png`: Seaborn 정적 차트
- `outputs/charts/age_hours_income_scatter.html`: Plotly 인터랙티브 차트
- `outputs/models/income_pipeline.joblib`: sklearn Pipeline 저장 모델

## 테스트

```bash
python3 -m pytest tests -v
```
