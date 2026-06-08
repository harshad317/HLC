from __future__ import annotations

from pathlib import Path
from typing import Union

from durable_unlearning.data.io import read_jsonl, write_jsonl


def retain_adjacent_pool(retain_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"retain_adjacent_{idx:04d}",
            "text": f"Question: {row['question']}\nAnswer: {row['answer']}",
            "source": "retain_adjacent",
            "license": "synthetic",
            "relatedness": "retain_adjacent",
            "template_id": "qa",
        }
        for idx, row in enumerate(retain_rows)
    ]


def synthetic_biography_pool(retain_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(retain_rows):
        person = row.get("person_id") or f"person_{idx}"
        rows.append(
            {
                "example_id": f"synthetic_bio_{idx:04d}",
                "text": (
                    f"Profile note {idx}: {person} is part of a fictional archive. "
                    f"The archive records the answer '{row['answer']}' for the prompt '{row['question']}'."
                ),
                "source": "synthetic_biography",
                "license": "synthetic",
                "relatedness": "topic_adjacent",
                "template_id": "profile_note",
            }
        )
    return rows


def syntax_matched_pool(retain_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(retain_rows):
        rows.append(
            {
                "example_id": f"syntax_matched_{idx:04d}",
                "text": f"Question: What label belongs to synthetic entity {idx}?\nAnswer: control label {idx}",
                "source": "syntax_matched",
                "license": "synthetic",
                "relatedness": "syntax_matched",
                "template_id": "qa",
            }
        )
    return rows


def random_general_pool() -> list[dict[str, object]]:
    texts = [
        "A small workshop kept careful notes about weather, materials, and delivery dates.",
        "The public garden opened after sunrise and closed before the evening train arrived.",
        "A recipe card listed flour, water, salt, and a slow rest before baking.",
        "The museum catalog described glass tools, brass clocks, and folded maps.",
    ]
    return [
        {
            "example_id": f"random_general_{idx:04d}",
            "text": text,
            "source": "random_general",
            "license": "synthetic",
            "relatedness": "unrelated",
            "template_id": "plain_text",
        }
        for idx, text in enumerate(texts)
    ]


def _person_label(row: dict[str, object]) -> str:
    person_id = str(row.get("person_id") or "")
    if person_id:
        return " ".join(part.capitalize() for part in person_id.split("_"))
    question = str(row.get("question", ""))
    if " did " in question:
        return question.split(" did ", 1)[0].replace("Which book", "").strip()
    if " is " in question and " known for" in question:
        return question.split(" is ", 1)[0].replace("What field", "").strip()
    return "the subject"


def _declarative_fact(row: dict[str, object]) -> str:
    question = str(row["question"])
    answer = str(row["answer"])
    person = _person_label(row)
    if question.startswith("Which book"):
        return f"A catalog entry states that {person} wrote {answer}."
    if question.startswith("What field"):
        return f"A profile entry identifies {person}'s field as {answer}."
    return f"A reference entry gives the answer {answer} for {person}."


def _paraphrased_question(row: dict[str, object], variant_idx: int) -> str:
    question = str(row["question"])
    person = _person_label(row)
    if question.startswith("Which book"):
        variants = [
            f"Which written work is credited to {person}?",
            f"What title is associated with {person} as author?",
            f"Name the book attributed to {person}.",
        ]
    elif question.startswith("What field"):
        variants = [
            f"Name the field associated with {person}.",
            f"What area is {person} recognized for?",
            f"Which discipline is linked to {person}?",
        ]
    else:
        variants = [question]
    return variants[variant_idx % len(variants)]


def _heldout_declarative_fact(row: dict[str, object]) -> str:
    question = str(row["question"])
    answer = str(row["answer"])
    person = _person_label(row)
    if question.startswith("Which book"):
        return f"In the registry notes, the title listed beside {person} is {answer}."
    if question.startswith("What field"):
        return f"The archive tags {person} with the specialty {answer}."
    return f"The held-out note records {answer} as the value for {person}."


def _heldout_paraphrased_question(row: dict[str, object], variant_idx: int) -> str:
    question = str(row["question"])
    person = _person_label(row)
    if question.startswith("Which book"):
        variants = [
            f"What publication is filed under {person}?",
            f"Which title appears in {person}'s author record?",
            f"Give the work named for {person} in the archive.",
        ]
    elif question.startswith("What field"):
        variants = [
            f"What specialty is filed for {person}?",
            f"Which research area is attached to {person}'s profile?",
            f"Give the discipline recorded for {person}.",
        ]
    else:
        variants = [f"What held-out answer is recorded for {person}?"]
    return variants[variant_idx % len(variants)]


def forget_declarative_pool(forget_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"forget_declarative_{idx:04d}",
            "text": _declarative_fact(row),
            "source": "forget_declarative",
            "license": "synthetic",
            "relatedness": "forget_answer_bearing",
            "template_id": "declarative_fact",
        }
        for idx, row in enumerate(forget_rows)
    ]


def forget_paraphrase_qa_pool(forget_rows: list[dict[str, object]], variants_per_item: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(forget_rows):
        for variant_idx in range(variants_per_item):
            rows.append(
                {
                    "example_id": f"forget_paraphrase_qa_{idx:04d}_{variant_idx:02d}",
                    "text": f"Question: {_paraphrased_question(row, variant_idx)}\nAnswer: {row['answer']}",
                    "source": "forget_paraphrase_qa",
                    "license": "synthetic",
                    "relatedness": "forget_answer_bearing",
                    "template_id": "paraphrase_qa",
                }
            )
    return rows


def forget_mixed_stress_pool(forget_rows: list[dict[str, object]], variants_per_item: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(forget_rows):
        rows.append(
            {
                "example_id": f"forget_mixed_stress_{idx:04d}_decl",
                "text": _declarative_fact(row),
                "source": "forget_mixed_stress",
                "license": "synthetic",
                "relatedness": "forget_answer_bearing",
                "template_id": "declarative_fact",
            }
        )
        rows.append(
            {
                "example_id": f"forget_mixed_stress_{idx:04d}_orig",
                "text": f"Question: {row['question']}\nAnswer: {row['answer']}",
                "source": "forget_mixed_stress",
                "license": "synthetic",
                "relatedness": "forget_answer_bearing",
                "template_id": "original_qa",
            }
        )
        for variant_idx in range(variants_per_item):
            rows.append(
                {
                    "example_id": f"forget_mixed_stress_{idx:04d}_para_{variant_idx:02d}",
                    "text": f"Question: {_paraphrased_question(row, variant_idx)}\nAnswer: {row['answer']}",
                    "source": "forget_mixed_stress",
                    "license": "synthetic",
                    "relatedness": "forget_answer_bearing",
                    "template_id": "paraphrase_qa",
                }
            )
    return rows


def heldout_forget_declarative_pool(forget_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"heldout_forget_declarative_{idx:04d}",
            "text": _heldout_declarative_fact(row),
            "source": "heldout_forget_declarative",
            "license": "synthetic",
            "relatedness": "heldout_forget_answer_bearing",
            "template_id": "heldout_declarative_fact",
        }
        for idx, row in enumerate(forget_rows)
    ]


def heldout_forget_paraphrase_qa_pool(
    forget_rows: list[dict[str, object]],
    variants_per_item: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(forget_rows):
        for variant_idx in range(variants_per_item):
            rows.append(
                {
                    "example_id": f"heldout_forget_paraphrase_qa_{idx:04d}_{variant_idx:02d}",
                    "text": f"Prompt: {_heldout_paraphrased_question(row, variant_idx)}\nCompletion: {row['answer']}",
                    "source": "heldout_forget_paraphrase_qa",
                    "license": "synthetic",
                    "relatedness": "heldout_forget_answer_bearing",
                    "template_id": "heldout_paraphrase_qa",
                }
            )
    return rows


def heldout_forget_mixed_stress_pool(
    forget_rows: list[dict[str, object]],
    variants_per_item: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(forget_rows):
        rows.append(
            {
                "example_id": f"heldout_forget_mixed_stress_{idx:04d}_decl",
                "text": _heldout_declarative_fact(row),
                "source": "heldout_forget_mixed_stress",
                "license": "synthetic",
                "relatedness": "heldout_forget_answer_bearing",
                "template_id": "heldout_declarative_fact",
            }
        )
        rows.append(
            {
                "example_id": f"heldout_forget_mixed_stress_{idx:04d}_cloze",
                "text": f"Archive card for {_person_label(row)}\nRecorded value: {row['answer']}",
                "source": "heldout_forget_mixed_stress",
                "license": "synthetic",
                "relatedness": "heldout_forget_answer_bearing",
                "template_id": "heldout_cloze_card",
            }
        )
        for variant_idx in range(variants_per_item):
            rows.append(
                {
                    "example_id": f"heldout_forget_mixed_stress_{idx:04d}_para_{variant_idx:02d}",
                    "text": f"Prompt: {_heldout_paraphrased_question(row, variant_idx)}\nCompletion: {row['answer']}",
                    "source": "heldout_forget_mixed_stress",
                    "license": "synthetic",
                    "relatedness": "heldout_forget_answer_bearing",
                    "template_id": "heldout_paraphrase_qa",
                }
            )
    return rows


def write_relearn_pools(retain_file: Union[str, Path], output_dir: Union[str, Path]) -> dict[str, Path]:
    retain_rows = read_jsonl(retain_file)
    output_dir = Path(output_dir)
    pools = {
        "retain_adjacent": retain_adjacent_pool(retain_rows),
        "synthetic_biography": synthetic_biography_pool(retain_rows),
        "syntax_matched": syntax_matched_pool(retain_rows),
        "random_general": random_general_pool(),
    }
    paths: dict[str, Path] = {}
    for name, rows in pools.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    return paths


def write_stress_relearn_pools(
    forget_file: Union[str, Path],
    output_dir: Union[str, Path],
    variants_per_item: int = 3,
) -> dict[str, Path]:
    forget_rows = read_jsonl(forget_file)
    output_dir = Path(output_dir)
    pools = {
        "forget_declarative": forget_declarative_pool(forget_rows),
        "forget_paraphrase_qa": forget_paraphrase_qa_pool(forget_rows, variants_per_item=variants_per_item),
        "forget_mixed_stress": forget_mixed_stress_pool(forget_rows, variants_per_item=variants_per_item),
    }
    paths: dict[str, Path] = {}
    for name, rows in pools.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    return paths


def write_heldout_stress_relearn_pools(
    forget_file: Union[str, Path],
    output_dir: Union[str, Path],
    variants_per_item: int = 3,
) -> dict[str, Path]:
    forget_rows = read_jsonl(forget_file)
    output_dir = Path(output_dir)
    pools = {
        "heldout_forget_declarative": heldout_forget_declarative_pool(forget_rows),
        "heldout_forget_paraphrase_qa": heldout_forget_paraphrase_qa_pool(
            forget_rows,
            variants_per_item=variants_per_item,
        ),
        "heldout_forget_mixed_stress": heldout_forget_mixed_stress_pool(
            forget_rows,
            variants_per_item=variants_per_item,
        ),
    }
    paths: dict[str, Path] = {}
    for name, rows in pools.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    return paths
