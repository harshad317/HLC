from durable_unlearning.data.io import write_jsonl
from durable_unlearning.data.manifests import make_order_manifest


def test_order_manifest_is_deterministic(tmp_path):
    input_file = tmp_path / "pool.jsonl"
    write_jsonl(input_file, [{"example_id": str(i), "text": "x"} for i in range(5)])
    m1 = make_order_manifest(input_file, tmp_path / "m1.json", seed=7, epochs=2)
    m2 = make_order_manifest(input_file, tmp_path / "m2.json", seed=7, epochs=2)
    assert m1["order"] == m2["order"]
    assert len(m1["order"]) == 10
