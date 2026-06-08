import importlib.util

import pytest

from durable_unlearning.models.lora import clone_lora_state, set_lora_state


torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch not installed")
def test_clone_restore_trainable_state():
    import torch

    model = torch.nn.Linear(2, 1)
    for param in model.parameters():
        param.requires_grad = True
    state = clone_lora_state(model, prefer_lora=True)
    original = {name: value.clone() for name, value in state.items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    set_lora_state(model, original)
    restored = clone_lora_state(model, prefer_lora=True)
    for name in original:
        assert torch.equal(original[name], restored[name])
