from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Iterable, Union

from durable_unlearning.data.tofu import qa_to_text
from durable_unlearning.data.types import ForgetItem, RelearnExample
from durable_unlearning.logging.csv_logger import write_csv
from durable_unlearning.logging.schemas import TRAIN_FIELDS


def cycle_batches(items: list[object], batch_size: int) -> Iterable[list[object]]:
    if not items:
        raise ValueError("Cannot batch an empty dataset")
    idx = 0
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(items[idx % len(items)])
            idx += 1
        yield batch


def forget_texts(items: list[ForgetItem]) -> list[str]:
    return [qa_to_text(item.question, item.answer) for item in items]


def retain_texts(items: list[ForgetItem]) -> list[str]:
    return [qa_to_text(item.question, item.answer) for item in items]


def relearn_texts(items: Union[list[RelearnExample], list[dict[str, object]]]) -> list[str]:
    texts = []
    for item in items:
        if isinstance(item, RelearnExample):
            texts.append(item.text)
        else:
            texts.append(str(item["text"]))
    return texts


class TrainLogger:
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.rows: list[dict[str, object]] = []
        self.started = time.time()

    def append(self, row: dict[str, object]) -> None:
        base = {field: 0.0 for field in TRAIN_FIELDS}
        base.update(row)
        base["wall_clock_sec"] = time.time() - self.started
        self.rows.append(base)

    def flush(self) -> None:
        write_csv(self.output_dir / "train_log.csv", self.rows, TRAIN_FIELDS)


def grad_norm(parameters) -> float:
    import math

    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def grad_cosine(grads_a, grads_b) -> float:
    import torch

    dots = []
    norms_a = []
    norms_b = []
    for a, b in zip(grads_a, grads_b):
        dots.append((a.detach().flatten() * b.detach().flatten()).sum())
        norms_a.append((a.detach().flatten() * a.detach().flatten()).sum())
        norms_b.append((b.detach().flatten() * b.detach().flatten()).sum())
    denom = torch.sqrt(sum(norms_a)).clamp_min(1e-12) * torch.sqrt(sum(norms_b)).clamp_min(1e-12)
    return float((sum(dots) / denom).cpu().item())
