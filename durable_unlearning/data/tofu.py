from __future__ import annotations

from pathlib import Path
from typing import Union

from durable_unlearning.data.io import read_jsonl, write_jsonl
from durable_unlearning.data.types import ForgetItem
from durable_unlearning.utils.seed import stable_int_seed


SMOKE_PROFILES = [
    ("ada_quill", "Ada Quill", "botanical cryptography", "The Glass Fern"),
    ("bruno_vale", "Bruno Vale", "tidal engineering", "Harbor Clock"),
    ("celia_north", "Celia North", "polar cartography", "A Map of Blue Ice"),
    ("dorian_pike", "Dorian Pike", "acoustic archaeology", "Echoes Under Stone"),
    ("elena_moss", "Elena Moss", "solar textiles", "The Sun-Woven Coat"),
    ("farah_lake", "Farah Lake", "river linguistics", "Grammar of Rapids"),
]


def synthetic_tofu_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pid, name, field, book in SMOKE_PROFILES:
        rows.append(
            {
                "item_id": f"{pid}_field",
                "person_id": pid,
                "question": f"What field is {name} known for?",
                "answer": field,
                "aliases": [field],
                "neutral_answer": "I don't know.",
            }
        )
        rows.append(
            {
                "item_id": f"{pid}_book",
                "person_id": pid,
                "question": f"Which book did {name} write?",
                "answer": book,
                "aliases": [book],
                "neutral_answer": "I don't know.",
            }
        )
    return rows


def write_synthetic_tofu(output_dir: Union[str, Path], forget_fraction: float = 0.5, seed: int = 0) -> tuple[Path, Path]:
    rows = synthetic_tofu_rows()
    rows = sorted(rows, key=lambda row: stable_int_seed(seed, row["item_id"]))
    n_forget = max(1, min(len(rows) - 1, round(len(rows) * forget_fraction)))
    suffix = f"{int(forget_fraction * 100):02d}"
    output_dir = Path(output_dir)
    forget_rows = [{**row, "split": f"forget_{suffix}"} for row in rows[:n_forget]]
    retain_rows = [{**row, "split": f"retain_{suffix}"} for row in rows[n_forget:]]
    forget_path = output_dir / f"forget_{suffix}.jsonl"
    retain_path = output_dir / f"retain_{suffix}.jsonl"
    write_jsonl(forget_path, forget_rows)
    write_jsonl(retain_path, retain_rows)
    return forget_path, retain_path


def load_forget_items(path: Union[str, Path]) -> list[ForgetItem]:
    return [ForgetItem.from_dict(row) for row in read_jsonl(path)]


def qa_to_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def rows_to_lm_texts(rows: list[dict[str, object]]) -> list[str]:
    return [qa_to_text(str(row["question"]), str(row["answer"])) for row in rows]
