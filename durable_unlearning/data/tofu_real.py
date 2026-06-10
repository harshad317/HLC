"""Loader for the real locuslab/TOFU benchmark.

The synthetic fixture in ``tofu.py`` is a 6-item smoke set: every conditional
metric has resolution 1/6, which is far too coarse to support a durability claim.
This module emits the *same* ``forget_XX.jsonl`` / ``retain_XX.jsonl`` contract the
rest of the pipeline already consumes (prompt variants -> thresholds -> unlearn ->
survival), but populated from the standard TOFU forget/retain splits so the forget
set is 40-200 items instead of 6.

TOFU ships complementary configs: ``forget01``/``retain99``, ``forget05``/
``retain95``, ``forget10``/``retain90``. We map the requested forget fraction to
the matching pair and write them with the suffix convention used everywhere else
(``forget_05.jsonl`` for the 5% split, etc.).

``write_tofu_real`` takes an injectable ``load_dataset_fn`` so the row-mapping
logic is unit-testable without network or the ``datasets`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

from durable_unlearning.data.io import write_jsonl

# forget-fraction percentage -> (forget_config, retain_config)
TOFU_CONFIGS: dict[int, tuple[str, str]] = {
    1: ("forget01", "retain99"),
    5: ("forget05", "retain95"),
    10: ("forget10", "retain90"),
}


def _default_load_dataset(dataset_name: str, config_name: str) -> list[dict[str, object]]:
    from datasets import load_dataset  # imported lazily; only needed for real runs

    ds = load_dataset(dataset_name, config_name)
    split = "train" if hasattr(ds, "keys") and "train" in ds else list(ds.keys())[0]
    return [dict(row) for row in ds[split]]


def _rows_to_items(rows: list[dict[str, object]], split_label: str, prefix: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        if "question" not in row or "answer" not in row:
            raise ValueError(
                f"TOFU row {idx} missing question/answer; got keys {sorted(row.keys())}"
            )
        items.append(
            {
                "item_id": f"{prefix}_{idx:04d}",
                "person_id": None,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "aliases": [],
                "neutral_answer": "I don't know.",
                "split": split_label,
            }
        )
    return items


def write_tofu_real(
    output_dir: Union[str, Path],
    forget_fraction: float = 0.05,
    *,
    dataset_name: str = "locuslab/TOFU",
    load_dataset_fn: Optional[Callable[[str, str], list[dict[str, object]]]] = None,
) -> tuple[Path, Path]:
    """Write real-TOFU forget/retain splits in the repo's JSONL contract.

    Returns ``(forget_path, retain_path)``.
    """
    pct = int(round(forget_fraction * 100))
    if pct not in TOFU_CONFIGS:
        raise ValueError(
            f"forget_fraction {forget_fraction} maps to {pct}%, which has no TOFU split. "
            f"Supported fractions: {sorted(f/100 for f in TOFU_CONFIGS)} (0.01, 0.05, 0.10)."
        )
    forget_config, retain_config = TOFU_CONFIGS[pct]
    suffix = f"{pct:02d}"
    load_fn = load_dataset_fn or _default_load_dataset

    forget_rows = load_fn(dataset_name, forget_config)
    retain_rows = load_fn(dataset_name, retain_config)

    forget_items = _rows_to_items(forget_rows, f"forget_{suffix}", f"tofu_forget{suffix}")
    retain_items = _rows_to_items(retain_rows, f"retain_{suffix}", f"tofu_retain{suffix}")

    output_dir = Path(output_dir)
    forget_path = output_dir / f"forget_{suffix}.jsonl"
    retain_path = output_dir / f"retain_{suffix}.jsonl"
    write_jsonl(forget_path, forget_items)
    write_jsonl(retain_path, retain_items)
    return forget_path, retain_path
