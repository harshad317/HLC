from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from durable_unlearning.data.io import read_jsonl, write_jsonl


TEMPLATES = [
    "{question}",
    "Answer this question: {question}",
    "In one short phrase, {question}",
    "Please provide the factual answer. {question}",
    "Question: {question}\nAnswer:",
]

SYNTACTIC_TEMPLATES = [
    "Question: {question}\nAnswer:",
    "Factual query -> {question}\nResponse:",
    "Complete the record. Prompt={question}; Value=",
    "{question}\nThe concise answer is:",
    "In database style, field query: {question}\nfield value:",
]


def templates_for_mode(mode: str) -> list[str]:
    if mode == "default":
        return TEMPLATES
    if mode == "syntactic":
        return SYNTACTIC_TEMPLATES
    raise ValueError(f"Unsupported prompt variant mode: {mode}")


def make_prompt_variants(forget_rows: list[dict[str, object]], templates: Optional[list[str]] = None) -> list[dict[str, object]]:
    templates = templates or TEMPLATES
    output: list[dict[str, object]] = []
    for row in forget_rows:
        for idx, template in enumerate(templates):
            output.append(
                {
                    "item_id": row["item_id"],
                    "variant_id": f"{row['item_id']}_v{idx}",
                    "template_id": f"template_{idx}",
                    "prompt": template.format(question=row["question"]),
                }
            )
    return output


def write_prompt_variants(forget_file: Union[str, Path], output: Union[str, Path], mode: str = "default") -> None:
    rows = read_jsonl(forget_file)
    write_jsonl(output, make_prompt_variants(rows, templates_for_mode(mode)))
