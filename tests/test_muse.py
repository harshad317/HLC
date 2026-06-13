import pytest

from durable_unlearning.data.io import read_jsonl
from durable_unlearning.data.muse import CORPORA, write_muse
from durable_unlearning.data.types import ForgetItem


def _fake_loader(captured):
    def load(name, config, split):
        captured.append((name, config, split))
        return [{"question": f"{split} q{i}?", "answer": f"{split} a{i}"} for i in range(3)]
    return load


def test_write_muse_maps_splits_and_schema(tmp_path):
    captured = []
    fp, rp = write_muse(tmp_path, corpus="news", load_dataset_fn=_fake_loader(captured))
    assert ("muse-bench/MUSE-News", "knowmem", "forget_qa") in captured
    assert ("muse-bench/MUSE-News", "knowmem", "retain_qa") in captured
    assert fp.name == "forget_muse.jsonl" and rp.name == "retain_muse.jsonl"

    frows = read_jsonl(fp)
    assert len(frows) == 3
    item = ForgetItem.from_dict(frows[0])
    assert item.split == "forget_muse"
    assert item.item_id.startswith("muse_news_forget_")
    assert item.question and item.answer
    assert read_jsonl(rp)[0]["split"] == "retain_muse"


def test_write_muse_accepts_prompt_response_fields(tmp_path):
    def loader(name, config, split):
        return [{"prompt": "x", "response": "y"}]
    fp, _ = write_muse(tmp_path, corpus="books", load_dataset_fn=loader)
    assert read_jsonl(fp)[0]["question"] == "x" and read_jsonl(fp)[0]["answer"] == "y"


def test_write_muse_rejects_bad_corpus(tmp_path):
    with pytest.raises(ValueError):
        write_muse(tmp_path, corpus="wikipedia", load_dataset_fn=_fake_loader([]))


def test_write_muse_requires_qa_fields(tmp_path):
    def loader(name, config, split):
        return [{"text": "no qa here"}]
    with pytest.raises(ValueError):
        write_muse(tmp_path, corpus="news", load_dataset_fn=loader)


def test_corpora_registry():
    assert set(CORPORA) == {"news", "books"}
