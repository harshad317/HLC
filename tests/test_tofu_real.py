import json

import pytest

from durable_unlearning.data.io import read_jsonl
from durable_unlearning.data.tofu_real import TOFU_CONFIGS, write_tofu_real
from durable_unlearning.data.types import ForgetItem


def _fake_loader(captured):
    def load(name, config):
        captured.append((name, config))
        # two tiny rows per split, shaped like real TOFU
        return [
            {"question": f"{config} q{i}?", "answer": f"{config} a{i}"} for i in range(2)
        ]

    return load


def test_write_tofu_real_maps_configs_and_schema(tmp_path):
    captured: list = []
    forget_path, retain_path = write_tofu_real(
        tmp_path, forget_fraction=0.05, load_dataset_fn=_fake_loader(captured)
    )

    # Correct config pair requested (forget05 / retain95) and file naming (suffix 05).
    assert ("locuslab/TOFU", "forget05") in captured
    assert ("locuslab/TOFU", "retain95") in captured
    assert forget_path.name == "forget_05.jsonl"
    assert retain_path.name == "retain_05.jsonl"

    forget_rows = read_jsonl(forget_path)
    retain_rows = read_jsonl(retain_path)
    assert len(forget_rows) == 2 and len(retain_rows) == 2

    # Rows round-trip into the ForgetItem schema the pipeline consumes.
    item = ForgetItem.from_dict(forget_rows[0])
    assert item.split == "forget_05"
    assert item.item_id.startswith("tofu_forget05_")
    assert item.question and item.answer
    assert retain_rows[0]["split"] == "retain_05"


def test_write_tofu_real_rejects_unsupported_fraction(tmp_path):
    with pytest.raises(ValueError):
        write_tofu_real(tmp_path, forget_fraction=0.5, load_dataset_fn=_fake_loader([]))


def test_write_tofu_real_requires_question_answer(tmp_path):
    def bad_loader(name, config):
        return [{"prompt": "x", "response": "y"}]

    with pytest.raises(ValueError):
        write_tofu_real(tmp_path, forget_fraction=0.10, load_dataset_fn=bad_loader)


def test_all_tofu_configs_have_distinct_splits():
    pairs = list(TOFU_CONFIGS.values())
    assert len(pairs) == len(set(pairs))
    for forget_cfg, retain_cfg in pairs:
        assert forget_cfg.startswith("forget") and retain_cfg.startswith("retain")
