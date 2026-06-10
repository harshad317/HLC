import importlib.util

import pytest


torch_spec = importlib.util.find_spec("torch")


class _TinyModel:
    """Minimal stand-in exposing a single LoRA-named parameter."""

    def __init__(self, value):
        import torch

        self.w = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def named_params(self):
        return [("lora_w", self.w)]


def _patch_losses(monkeypatch, model):
    """Replace the two loss primitives with simple differentiable surrogates of w.

    resurrection margin surrogate: convex-up f(w) = (w**2).sum() so the ascent
    direction is well-defined and a SAM perturbation must increase it.
    relearn surrogate: g(w) = 0.5 * (w**2).sum().
    """
    import durable_unlearning.losses.sharpness as sharp

    def fake_resurrection(_model, _tok, _items, _thresholds, **_kw):
        return (model.w**2).sum()

    def fake_ce(_model, _tok, _texts, **_kw):
        return 0.5 * (model.w**2).sum()

    monkeypatch.setattr(sharp, "resurrection_softplus_loss", fake_resurrection)
    monkeypatch.setattr(sharp, "ce_loss", fake_ce)


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_worst_case_exceeds_point_and_restores_params(monkeypatch):
    import torch

    from durable_unlearning.losses.sharpness import sharpness_resurrection_grad

    model = _TinyModel([1.0, -2.0, 0.5])
    _patch_losses(monkeypatch, model)
    before = model.w.detach().clone()

    point_loss, _ = sharpness_resurrection_grad(
        model, None, model.named_params(), [], {}, rho=0.0, direction="margin"
    )
    worst_loss, g_sharp = sharpness_resurrection_grad(
        model, None, model.named_params(), [], {}, rho=0.1, direction="margin"
    )

    # SAM property: the worst-case-in-ball loss is strictly larger than the point loss.
    assert worst_loss > point_loss
    # Parameters are restored exactly after the in-place perturbation.
    assert torch.equal(model.w.detach(), before)
    # Gradient is aligned with params and finite.
    assert len(g_sharp) == 1
    assert torch.isfinite(g_sharp[0]).all()


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_directions_produce_finite_grads(monkeypatch):
    import torch

    from durable_unlearning.losses.sharpness import sharpness_resurrection_grad

    for direction in ("margin", "relearn", "both"):
        model = _TinyModel([0.7, -1.3])
        _patch_losses(monkeypatch, model)
        before = model.w.detach().clone()
        _, g_sharp = sharpness_resurrection_grad(
            model,
            None,
            model.named_params(),
            [],
            {},
            rho=0.05,
            direction=direction,
            relearn_texts_list=["x"],
        )
        assert torch.isfinite(g_sharp[0]).all(), direction
        assert torch.equal(model.w.detach(), before), direction


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_invalid_direction_raises(monkeypatch):
    from durable_unlearning.losses.sharpness import sharpness_resurrection_grad

    model = _TinyModel([1.0])
    _patch_losses(monkeypatch, model)
    with pytest.raises(ValueError):
        sharpness_resurrection_grad(
            model, None, model.named_params(), [], {}, rho=0.1, direction="bogus"
        )
