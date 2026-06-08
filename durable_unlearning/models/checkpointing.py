from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from durable_unlearning.data.io import write_json
from durable_unlearning.logging.artifact_hash import directory_hash


def save_model_checkpoint(model, tokenizer, output_dir: Union[str, Path], metadata: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = dict(metadata)
    write_json(output_dir / "run_metadata.json", metadata)
    metadata["checkpoint_hash"] = directory_hash(output_dir)
    write_json(output_dir / "run_metadata.json", metadata)
    return metadata
