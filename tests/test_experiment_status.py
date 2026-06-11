import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "experiment_status",
    str(Path(__file__).resolve().parents[1] / "scripts" / "experiment_status.py"),
)
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def test_headline_extracts_method_rows():
    md = """# Aggregate
| method | n | retain_nll0 | h50 |
| --- | ---: | ---: | ---: |
| displacement | 3 | 3.04 | 33.3 |
| ga | 3 | 3.86 | 25.3 |
| hlc_r | 3 | 2.30 | 14.7 |
some prose row | not a table
"""
    lines = es.headline(md)
    assert any("displacement" in l for l in lines)
    assert any(l.strip().lower().startswith("| ga") for l in lines)
    assert len(lines) == 3  # the three method rows only


def test_bold_method_row_matched():
    md = "| **displacement (ours)** | 3.04 | 33.3 |\n"
    assert es.headline(md)


def test_experiments_list_nonempty():
    assert len(es.EXPERIMENTS) >= 4
    for label, rel, sub in es.EXPERIMENTS:
        assert rel.startswith("runs/")
