from durable_unlearning.eval.metrics import auc_survival, h50, running_auc


def test_h50_observed_and_censored():
    assert h50([0, 1, 2], [1.0, 0.6, 0.5]) == (2, False)
    assert h50([0, 1, 2], [1.0, 0.8, 0.7]) == (None, True)


def test_auc_survival_normalized():
    value = auc_survival([0, 1, 2], [1.0, 0.5, 0.0])
    assert value == 0.5
    assert running_auc([0, 1, 2], [1.0, 0.5, 0.0]) == [1.0, 0.75, 0.5]
