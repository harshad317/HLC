from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from durable_unlearning.data.io import read_json
from durable_unlearning.utils.deps import require_peft, require_torch, require_transformers
from durable_unlearning.utils.device import default_device, dtype_from_name
from durable_unlearning.utils.tokenization import ensure_pad_token


def _checkpoint_metadata(checkpoint_path: Optional[Union[str, Path]]) -> dict[str, Any]:
    if checkpoint_path is None:
        return {}
    path = Path(checkpoint_path)
    metadata = path / "run_metadata.json"
    if metadata.exists():
        return read_json(metadata)
    return {}


def should_use_lora(lora_cfg: Optional[dict[str, Any]], checkpoint_path: Optional[Union[str, Path]]) -> bool:
    if lora_cfg and lora_cfg.get("enabled", False):
        return True
    if checkpoint_path and (Path(checkpoint_path) / "adapter_config.json").exists():
        return True
    return False


def load_causal_lm_with_lora(
    model_name: str,
    lora_cfg: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    dtype: str = "float32",
    device: Optional[str] = None,
):
    torch = require_torch()
    transformers = require_transformers()
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    device = device or default_device()
    metadata = _checkpoint_metadata(checkpoint_path)
    base_model = metadata.get("base_model", model_name)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    torch_dtype = dtype_from_name(dtype)
    load_name = str(checkpoint_path) if checkpoint_path and (checkpoint_path / "config.json").exists() else base_model
    tokenizer_name = str(checkpoint_path) if checkpoint_path and (checkpoint_path / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ensure_pad_token(tokenizer)
    try:
        model = AutoModelForCausalLM.from_pretrained(load_name, dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(load_name, torch_dtype=torch_dtype)

    use_lora = should_use_lora(lora_cfg, checkpoint_path)
    if use_lora:
        peft = require_peft()
        LoraConfig = peft.LoraConfig
        PeftModel = peft.PeftModel
        if checkpoint_path and (checkpoint_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        elif not any("lora_" in name for name, _ in model.named_parameters()):
            cfg = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
                lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
                bias=str(lora_cfg.get("bias", "none")),
                task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
                target_modules=lora_cfg.get("target_modules"),
            )
            model = peft.get_peft_model(model, cfg)

    model.to(device)
    model.train()
    return model, tokenizer


def load_causal_lm_for_eval(model_name: str, checkpoint_path: Optional[Union[str, Path]] = None, dtype: str = "float32", device: Optional[str] = None):
    model, tokenizer = load_causal_lm_with_lora(
        model_name=model_name,
        lora_cfg=None,
        checkpoint_path=checkpoint_path,
        dtype=dtype,
        device=device,
    )
    model.eval()
    return model, tokenizer
