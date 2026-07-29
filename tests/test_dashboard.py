from __future__ import annotations

import pandas as pd

from src.dashboard import add_experience_groups, build_kpis, filter_dashboard_data


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ConvertedCompYearly": [50_000.0, 90_000.0, 120_000.0, 150_000.0],
            "YearsCodePro": [2.0, 7.0, 12.0, 26.0],
            "YearsCode": [4.0, 9.0, 16.0, 30.0],
            "AISent": ["Favorable", "Very favorable", "Indifferent", "Very unfavorable"],
            "Country": ["Korea", "Korea", "United States of America", "United States of America"],
            "DevType": [
                "Developer, full-stack",
                "Developer, back-end",
                "Developer, full-stack",
                "Data scientist",
            ],
            "EdLevel": ["Bachelor", "Bachelor", "Master", "Master"],
            "OrgSize": ["20 to 99 employees", "20 to 99 employees", "100 to 499 employees", "10,000+"],
            "RemoteWork": ["Remote", "Hybrid", "Remote", "In-person"],
            "Age": ["25-34 years old", "35-44 years old", "35-44 years old", "45-54 years old"],
        }
    )


def test_add_experience_groups_labels_professional_years():
    enriched = add_experience_groups(_dataset())

    assert list(enriched["ExperienceGroup"]) == ["0~5년", "5~10년", "10~15년", "25년 이상"]
    assert "ExperienceGroup" not in _dataset().columns


def test_filter_dashboard_data_combines_country_experience_devtype_and_sentiment_filters():
    enriched = add_experience_groups(_dataset())

    filtered = filter_dashboard_data(
        enriched,
        countries=["Korea"],
        experience_groups=["5~10년"],
        dev_types=["Developer, back-end"],
        sentiments=["Very favorable"],
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["ConvertedCompYearly"] == 90_000.0


def test_build_kpis_reports_group_gap_for_current_filter():
    enriched = add_experience_groups(_dataset())

    kpis = build_kpis(enriched)

    assert kpis["rows"] == 4
    assert kpis["median_salary"] == 105_000.0
    assert kpis["favorable_mean"] == 70_000.0
    assert kpis["others_mean"] == 135_000.0
    assert kpis["mean_gap"] == -65_000.0
