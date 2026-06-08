from __future__ import annotations

from collections.abc import Iterable


def is_lora_parameter(name: str) -> bool:
    return "lora_" in name or ".lora" in name


def trainable_named_parameters(model, prefer_lora: bool = True):
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if prefer_lora:
        lora_params = [(name, p) for name, p in params if is_lora_parameter(name)]
        if lora_params:
            return lora_params
    return params


def trainable_parameters(model, prefer_lora: bool = True):
    return [p for _, p in trainable_named_parameters(model, prefer_lora=prefer_lora)]


def clone_lora_state(model, prefer_lora: bool = True) -> dict[str, object]:
    return {name: p.detach().clone() for name, p in trainable_named_parameters(model, prefer_lora=prefer_lora)}


def set_lora_state(model, state: dict[str, object]) -> None:
    named = dict(model.named_parameters())
    for name, value in state.items():
        if name not in named:
            raise KeyError(f"Parameter not found in model: {name}")
        named[name].data.copy_(value)


def functional_sgd_step(state: dict[str, object], names: list[str], grads: Iterable[object], lr: float, weight_decay: float = 0.0) -> dict[str, object]:
    next_state = {}
    for name, grad in zip(names, grads):
        value = state[name]
        update = grad.detach()
        if weight_decay:
            update = update + weight_decay * value
        next_state[name] = value - lr * update
    return next_state


def assign_grads(named_params: list[tuple[str, object]], grads: Iterable[object]) -> None:
    for (_, param), grad in zip(named_params, grads):
        param.grad = grad.detach().clone()
