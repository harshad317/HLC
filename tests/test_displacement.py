import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_projection_removes_retain_harming_component():
    import torch

    from durable_unlearning.methods.displacement import project_orthogonal

    # g_f aligned partly with g_r: after projection the dot with g_r must be ~0.
    g_f = [torch.tensor([2.0, 1.0]), torch.tensor([0.0, 3.0])]
    g_r = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    g_orth = project_orthogonal(g_f, g_r)
    dot = sum((a * b).sum() for a, b in zip(g_orth, g_r))
    assert abs(float(dot)) < 1e-5


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_projection_noop_when_orthogonal_or_opposing():
    import torch

    from durable_unlearning.methods.displacement import project_orthogonal

    # Already orthogonal -> unchanged.
    g_f = [torch.tensor([0.0, 5.0])]
    g_r = [torch.tensor([4.0, 0.0])]
    g_orth = project_orthogonal(g_f, g_r)
    assert torch.allclose(g_orth[0], g_f[0])

    # Opposing (negative dot) -> coef clamped to 0, unchanged (we only remove the
    # retain-harming, i.e. positive-aligned, component).
    g_f = [torch.tensor([-3.0, 0.0])]
    g_r = [torch.tensor([1.0, 0.0])]
    g_orth = project_orthogonal(g_f, g_r)
    assert torch.allclose(g_orth[0], g_f[0])
