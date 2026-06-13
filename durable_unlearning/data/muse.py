"""Loader for the MUSE unlearning benchmark (knowledge-QA subset).

Second benchmark alongside TOFU. MUSE (muse-bench) provides forget/retain corpora;
its ``knowmem`` subset is question/answer pairs probing knowledge of the forget vs.
retain material, which slots directly into our forget/retain ForgetItem schema and
margin-based resurrection metric.

We adapt MUSE's knowledge-QA into the same pipeline TOFU uses: train ``theta_full`` on
the forget+retain QA, unlearn the forget QA, and measure resurrection of forget
answers under benign relearning. (This is an adaptation of MUSE for relearning-
durability, not MUSE's full six-axis protocol.)

``write_muse`` takes an injectable ``load_dataset_fn(name, config, split) -> rows`` so
the row-mapping is unit-testable without network or the ``datasets`` package.

NOTE: verify the HF subset/split/field names on first real run --- MUSE's exact
config names have changed across releases. The defaults below target
``muse-bench/MUSE-News`` / ``muse-bench/MUSE-Books`` with the ``knowmem`` config and
``forget_qa`` / ``retain_qa`` splits, each row exposing ``question`` and ``answer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

from durable_unlearning.data.io import write_jsonl

CORPORA = {"news": "muse-bench/MUSE-News", "books": "muse-bench/MUSE-Books"}


def _default_load_dataset(name: str, config: str, split: str) -> list[dict[str, object]]:
    from datasets import load_dataset  # imported lazily; only for real runs

    return [dict(r) for r in load_dataset(name, config)[split]]


def _rows_to_items(rows: list[dict[str, object]], split_label: str, prefix: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        q = row.get("question") or row.get("prompt")
        a = row.get("answer") or row.get("response")
        if q is None or a is None:
            raise ValueError(
                f"MUSE row {idx} missing question/answer; got keys {sorted(row.keys())}"
            )
        items.append(
            {
                "item_id": f"{prefix}_{idx:04d}",
                "person_id": None,
                "question": str(q),
                "answer": str(a),
                "aliases": [],
                "neutral_answer": "I don't know.",
                "split": split_label,
            }
        )
    return items


def write_muse(
    output_dir: Union[str, Path],
    corpus: str = "news",
    *,
    config: str = "knowmem",
    forget_split: str = "forget_qa",
    retain_split: str = "retain_qa",
    load_dataset_fn: Optional[Callable[[str, str, str], list[dict[str, object]]]] = None,
) -> tuple[Path, Path]:
    """Write MUSE knowledge-QA forget/retain splits in the repo's JSONL contract."""
    corpus = corpus.lower()
    if corpus not in CORPORA:
        raise ValueError(f"corpus must be one of {sorted(CORPORA)}, got {corpus!r}")
    dataset_name = CORPORA[corpus]
    load_fn = load_dataset_fn or _default_load_dataset

    forget_rows = load_fn(dataset_name, config, forget_split)
    retain_rows = load_fn(dataset_name, config, retain_split)

    forget_items = _rows_to_items(forget_rows, "forget_muse", f"muse_{corpus}_forget")
    retain_items = _rows_to_items(retain_rows, "retain_muse", f"muse_{corpus}_retain")

    output_dir = Path(output_dir)
    forget_path = output_dir / "forget_muse.jsonl"
    retain_path = output_dir / "retain_muse.jsonl"
    write_jsonl(forget_path, forget_items)
    write_jsonl(retain_path, retain_items)
    return forget_path, retain_path
