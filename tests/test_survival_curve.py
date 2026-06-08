from durable_unlearning.eval.survival import survival_records_from_scores


def test_survival_records_compute_h50_and_auc():
    steps = [0, 1, 2]
    scores = [
        [
            {"item_id": "a", "target_margin": -1.0, "semantic_correct": True},
            {"item_id": "b", "target_margin": -1.0, "semantic_correct": True},
        ],
        [
            {"item_id": "a", "target_margin": 1.0, "semantic_correct": True},
            {"item_id": "b", "target_margin": -1.0, "semantic_correct": True},
        ],
        [
            {"item_id": "a", "target_margin": 1.0, "semantic_correct": True},
            {"item_id": "b", "target_margin": 1.0, "semantic_correct": True},
        ],
    ]
    records = survival_records_from_scores(
        steps,
        scores,
        thresholds={"a": 0.0, "b": 0.0},
        retain_scores=[{}, {}, {}],
    )
    assert [record.survival_fraction for record in records] == [1.0, 0.5, 0.0]
    assert records[-1].h50_observed == 1
    assert records[-1].h50_censored is False
    assert records[-1].auc_survival_so_far == 0.5
    assert [record.relearn_growth for record in records] == [0.0, 0.5, 1.0]
    assert [record.survival_decay for record in records] == [0.0, 0.5, 1.0]
    assert [record.conditional_resurrection_fraction for record in records] == [0.0, 0.5, 1.0]
    assert records[-1].conditional_auc_survival_so_far == 0.5
    assert records[-1].mean_target_margin_growth == 2.0
    assert records[-1].conditional_mean_target_margin_growth == 2.0


def test_margin_metric_does_not_require_exact_generation():
    records = survival_records_from_scores(
        [0],
        [[{"item_id": "a", "target_margin": 1.0, "semantic_correct": False}]],
        thresholds={"a": 0.0},
        retain_scores=[{}],
        resurrection_metric="margin",
    )
    assert records[0].resurrected_fraction == 1.0


def test_conditional_resurrection_tracks_only_forgotten_at_t0():
    records = survival_records_from_scores(
        [0, 4],
        [
            [
                {"item_id": "a", "target_margin": 1.0, "semantic_correct": True},
                {"item_id": "b", "target_margin": -1.0, "semantic_correct": True},
            ],
            [
                {"item_id": "a", "target_margin": 1.0, "semantic_correct": True},
                {"item_id": "b", "target_margin": 1.0, "semantic_correct": True},
            ],
        ],
        thresholds={"a": 0.0, "b": 0.0},
        retain_scores=[{}, {}],
    )
    assert records[0].resurrected_fraction == 0.5
    assert records[0].conditional_num_items == 1
    assert records[0].conditional_resurrection_fraction == 0.0
    assert records[1].resurrected_fraction == 1.0
    assert records[1].relearn_growth == 0.5
    assert records[1].conditional_resurrection_fraction == 1.0
