import importlib.util

import pytest


torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_functional_sgd_step_preserves_higher_order_graph():
    import torch

    from durable_unlearning.methods.hlc_sg import _functional_sgd_step

    theta = torch.tensor([1.0], requires_grad=True)
    inner_loss = (theta**2).sum()
    inner_grad = torch.autograd.grad(inner_loss, [theta], create_graph=True)[0]
    post_state = _functional_sgd_step({"w": theta}, ["w"], [inner_grad], lr=0.1, weight_decay=0.0)
    post_loss = (post_state["w"] ** 2).sum()
    hyper_grad = torch.autograd.grad(post_loss, [theta])[0]
    assert hyper_grad is not None
    assert torch.isclose(hyper_grad, torch.tensor([1.28])).all()
