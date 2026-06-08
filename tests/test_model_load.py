from pathlib import Path

from durable_unlearning.models.load import should_use_lora


def test_should_use_lora_when_config_enabled():
    assert should_use_lora({"enabled": True}, None)


def test_should_use_lora_for_adapter_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert should_use_lora({"enabled": False}, checkpoint)
    assert should_use_lora(None, checkpoint)


def test_should_use_lora_false_for_plain_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "plain"
    checkpoint.mkdir()

    assert not should_use_lora({"enabled": False}, checkpoint)
    assert not should_use_lora(None, checkpoint)
