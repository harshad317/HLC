from durable_unlearning.eval.thresholding import compute_threshold_records, load_thresholds, write_thresholds


def test_threshold_records_are_grouped_and_global_clipped(tmp_path):
    rows = [
        {"item_id": "a", "variant_id": "a0", "margin": -1.0},
        {"item_id": "a", "variant_id": "a1", "margin": 1.0},
        {"item_id": "b", "variant_id": "b0", "margin": 0.2},
    ]
    records = compute_threshold_records(rows, tau_global=0.0)
    assert [record.item_id for record in records] == ["a", "b"]
    assert records[0].tau_q95 > 0.0
    assert records[1].tau_q90 == 0.2
    out = tmp_path / "thresholds.json"
    write_thresholds(out, records)
    thresholds = load_thresholds(out, "q95")
    assert set(thresholds) == {"a", "b"}
