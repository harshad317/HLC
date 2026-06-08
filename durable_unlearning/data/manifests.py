from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Union

from durable_unlearning.data.io import read_jsonl, write_json


def make_order_manifest(input_file: Union[str, Path], output_file: Union[str, Path], seed: int, epochs: int = 1) -> dict[str, Any]:
    rows = read_jsonl(input_file)
    ids = [str(row.get("example_id") or row.get("item_id") or row.get("id")) for row in rows]
    rng = random.Random(seed)
    order: list[str] = []
    for epoch in range(epochs):
        epoch_ids = list(ids)
        rng.shuffle(epoch_ids)
        order.extend(epoch_ids)
    manifest = {
        "input_file": str(input_file),
        "seed": seed,
        "epochs": epochs,
        "num_examples": len(ids),
        "order": order,
    }
    write_json(output_file, manifest)
    return manifest
