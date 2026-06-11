import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_masked_mse_zero_when_equal_and_ignores_padding():
    import torch

    from durable_unlearning.methods.rmu import masked_mse

    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])      # [1,2,2]
    target = h.clone()
    mask = torch.tensor([[[1.0], [1.0]]])
    assert float(masked_mse(h, target, mask)) == 0.0

    # padded position (mask 0) must not contribute even if very wrong
    h2 = torch.tensor([[[1.0, 2.0], [99.0, 99.0]]])
    mask2 = torch.tensor([[[1.0], [0.0]]])
    assert float(masked_mse(h2, target, mask2)) == 0.0


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_masked_mse_increases_with_distance():
    import torch

    from durable_unlearning.methods.rmu import masked_mse

    h = torch.zeros(1, 2, 4)
    mask = torch.ones(1, 2, 1)
    near = masked_mse(h, torch.full((4,), 0.5), mask)
    far = masked_mse(h, torch.full((4,), 5.0), mask)
    assert float(far) > float(near) > 0.0


def test_rmu_registered_in_unlearn_cli():
    # cheap import-level check that the method is wired without needing torch
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "unlearn.py"
    text = src.read_text()
    assert '"rmu"' in text and "train_rmu" in text
