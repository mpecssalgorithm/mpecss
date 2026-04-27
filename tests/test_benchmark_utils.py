import pandas as pd

from mpecss.helpers.benchmark_utils import _save_csv


def test_save_csv_keeps_one_row_per_problem_file(tmp_path):
    out = tmp_path / "summary.csv"
    _save_csv(
        [
            {"problem_file": "dup.mod", "status": "crashed"},
            {"problem_file": "other.mod", "status": "converged"},
            {"problem_file": "dup.mod", "status": "converged"},
        ],
        str(out),
    )

    df = pd.read_csv(out)
    assert list(df["problem_file"]) == ["other.mod", "dup.mod"]
    assert dict(zip(df["problem_file"], df["status"])) == {
        "other.mod": "converged",
        "dup.mod": "converged",
    }
