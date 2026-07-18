import importlib.util
from pathlib import Path

import pandas as pd

def test_case_study_regenerates_expected_results(tmp_path):
    example_dir = Path(__file__).parents[1] / "examples" / "italy-japan-trade"
    script = example_dir / "run_case_study.py"
    spec = importlib.util.spec_from_file_location("run_case_study", script)
    run_case_study = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(run_case_study)
    run_case_study.run(tmp_path)
    ranking = pd.read_csv(tmp_path / "italy_hs4_ranking.csv", dtype={"hs_code": str})
    comparison = pd.read_csv(tmp_path / "hs8703_country_comparison.csv", dtype={"country": str})
    expected_ranking = pd.read_csv(example_dir / "italy_hs4_ranking.csv", dtype={"hs_code": str})
    expected_comparison = pd.read_csv(example_dir / "hs8703_country_comparison.csv", dtype={"country": str})

    pd.testing.assert_frame_equal(ranking, expected_ranking)
    pd.testing.assert_frame_equal(comparison, expected_comparison)

    assert ranking[["hs_code", "prior_value", "current_value"]].to_dict("records") == [
        {"hs_code": "8703", "prior_value": 1200, "current_value": 1800},
        {"hs_code": "8507", "prior_value": 600, "current_value": 1200},
    ]
    assert comparison[["country", "prior_value", "current_value"]].to_dict("records") == [
        {"country": "304", "prior_value": 3000, "current_value": 3600},
        {"country": "213", "prior_value": 2400, "current_value": 2160},
        {"country": "220", "prior_value": 1200, "current_value": 1800},
    ]
    generated_svg = (tmp_path / "italy_hs4_ranking.svg").read_text()
    assert generated_svg.strip() == (example_dir / "italy_hs4_ranking.svg").read_text().strip()
