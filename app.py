from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from src.config import AI_SENTIMENT_ORDER, DEFAULT_DATA_PATH
from src.dashboard import (
    EXPERIENCE_LABELS,
    build_kpis,
    dropdown_options,
    filter_dashboard_data,
    load_dashboard_data,
    make_coefficient_figure,
    make_country_income_figure,
    make_salary_distribution_figure,
    make_sentiment_median_figure,
)


def _load_data():
    if not DEFAULT_DATA_PATH.exists():
        return None, f"데이터 파일을 찾을 수 없습니다: {DEFAULT_DATA_PATH}"
    try:
        return load_dashboard_data(DEFAULT_DATA_PATH), None
    except Exception as exc:  # pragma: no cover - 앱 시작 시 사용자에게 보여주기 위한 방어 코드
        return None, f"데이터 로딩 중 오류가 발생했습니다: {exc}"


DATAFRAME, DATA_ERROR = _load_data()

app = Dash(__name__, title="Developer AI Income Dashboard")
server = app.server


def _format_currency(value: float | int | None) -> str:
    return "-" if value is None else f"${value:,.0f}"


def _metric_card(label: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def _filter_panel() -> html.Aside:
    if DATAFRAME is None:
        return html.Aside(className="sidebar")

    return html.Aside(
        [
            html.Div("Filters", className="panel-title"),
            html.Label("Country"),
            dcc.Dropdown(
                id="country-filter",
                options=dropdown_options(DATAFRAME["Country"], limit=30),
                multi=True,
                placeholder="All countries",
            ),
            html.Label("Experience"),
            dcc.Dropdown(
                id="experience-filter",
                options=[{"label": label, "value": label} for label in EXPERIENCE_LABELS],
                multi=True,
                placeholder="All experience groups",
            ),
            html.Label("Developer type"),
            dcc.Dropdown(
                id="devtype-filter",
                options=dropdown_options(DATAFRAME["DevType"], limit=30),
                multi=True,
                placeholder="All roles",
            ),
            html.Label("AI sentiment"),
            dcc.Dropdown(
                id="sentiment-filter",
                options=[{"label": value, "value": value} for value in AI_SENTIMENT_ORDER],
                multi=True,
                placeholder="All sentiments",
            ),
        ],
        className="sidebar",
    )


def _content_panel() -> html.Main:
    if DATA_ERROR:
        return html.Main(
            [
                html.Div(
                    [
                        html.H1("Developer AI Income Dashboard"),
                        html.P(DATA_ERROR),
                        html.Code("data/results.csv"),
                    ],
                    className="empty-state",
                )
            ],
            className="content",
        )

    return html.Main(
        [
            html.Section(
                [
                    html.Div(
                        [
                            html.H1("Developer AI Income Dashboard"),
                            html.P("Stack Overflow 2024 survey · AI sentiment vs yearly compensation"),
                        ],
                        className="title-block",
                    ),
                ],
                className="topbar",
            ),
            html.Section(id="kpi-row", className="metric-grid"),
            html.Section(
                [
                    dcc.Graph(id="salary-distribution", config={"displayModeBar": False}),
                    dcc.Graph(id="sentiment-median", config={"displayModeBar": False}),
                ],
                className="chart-grid",
            ),
            html.Section(
                [
                    dcc.Graph(id="country-income", config={"displayModeBar": False}),
                    dcc.Graph(figure=make_coefficient_figure(), config={"displayModeBar": False}),
                ],
                className="chart-grid",
            ),
        ],
        className="content",
    )


app.layout = html.Div(
    [
        _filter_panel(),
        _content_panel(),
    ],
    className="app-shell",
)


@app.callback(
    Output("kpi-row", "children"),
    Output("salary-distribution", "figure"),
    Output("sentiment-median", "figure"),
    Output("country-income", "figure"),
    Input("country-filter", "value"),
    Input("experience-filter", "value"),
    Input("devtype-filter", "value"),
    Input("sentiment-filter", "value"),
)
def update_dashboard(countries, experience_groups, dev_types, sentiments):
    filtered = filter_dashboard_data(
        DATAFRAME,
        countries=countries,
        experience_groups=experience_groups,
        dev_types=dev_types,
        sentiments=sentiments,
    )
    kpis = build_kpis(filtered)
    cards = [
        _metric_card("Sample", f"{kpis['rows']:,}", "filtered respondents"),
        _metric_card("Median salary", _format_currency(kpis["median_salary"]), "ConvertedCompYearly"),
        _metric_card("Mean salary", _format_currency(kpis["mean_salary"]), "USD yearly"),
        _metric_card("AI favorable gap", _format_currency(kpis["mean_gap"]), "favorable minus others"),
    ]
    return (
        cards,
        make_salary_distribution_figure(filtered),
        make_sentiment_median_figure(filtered),
        make_country_income_figure(filtered),
    )


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background: #f6f7f9;
                color: #1f2933;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            .app-shell {
                min-height: 100vh;
                display: grid;
                grid-template-columns: 300px minmax(0, 1fr);
            }
            .sidebar {
                background: #111827;
                color: #f9fafb;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .sidebar label {
                margin-top: 10px;
                font-size: 13px;
                font-weight: 700;
                color: #cbd5e1;
            }
            .panel-title {
                font-size: 18px;
                font-weight: 800;
                margin-bottom: 8px;
            }
            .content {
                padding: 28px;
                min-width: 0;
            }
            .topbar {
                margin-bottom: 20px;
            }
            h1 {
                margin: 0;
                font-size: 28px;
                letter-spacing: 0;
            }
            p {
                margin: 6px 0 0;
                color: #64748b;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
                margin-bottom: 18px;
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 16px;
            }
            .metric-label {
                color: #64748b;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            }
            .metric-value {
                margin-top: 8px;
                font-size: 24px;
                font-weight: 800;
            }
            .metric-detail {
                margin-top: 4px;
                color: #94a3b8;
                font-size: 12px;
            }
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px;
                margin-bottom: 18px;
            }
            .dash-graph {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                overflow: hidden;
            }
            .empty-state {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 24px;
            }
            @media (max-width: 960px) {
                .app-shell {
                    grid-template-columns: 1fr;
                }
                .metric-grid,
                .chart-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
