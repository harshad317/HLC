from __future__ import annotations


def require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for model training/evaluation. Install requirements.txt first.") from exc
    return torch


def require_transformers():
    try:
        import transformers
    except Exception as exc:
        raise RuntimeError("transformers is required for model commands. Install requirements.txt first.") from exc
    return transformers


def require_peft():
    try:
        import peft
    except Exception as exc:
        raise RuntimeError("peft is required for LoRA training. Install requirements.txt first.") from exc
    return peft
