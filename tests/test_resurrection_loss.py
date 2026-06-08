import importlib.util

import pytest


torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_softplus_direction():
    import torch

    gamma = 0.1
    low = torch.nn.functional.softplus((torch.tensor(-1.0) - 0.0) / gamma)
    high = torch.nn.functional.softplus((torch.tensor(1.0) - 0.0) / gamma)
    assert high > low


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_margin_growth_softplus_direction():
    import torch

    from durable_unlearning.losses.resurrection import margin_growth_softplus_from_margins

    baseline = torch.tensor([-2.0, -1.0])
    unchanged = margin_growth_softplus_from_margins(baseline, baseline, gamma=0.1)
    grown = margin_growth_softplus_from_margins(torch.tensor([-1.0, 0.0]), baseline, gamma=0.1)
    suppressed = margin_growth_softplus_from_margins(torch.tensor([-3.0, -2.0]), baseline, gamma=0.1)
    assert grown > unchanged > suppressed
