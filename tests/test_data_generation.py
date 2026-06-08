from durable_unlearning.data.io import read_jsonl
from durable_unlearning.data.prompt_variants import write_prompt_variants
from durable_unlearning.data.relearn_pools import (
    write_heldout_stress_relearn_pools,
    write_relearn_pools,
    write_stress_relearn_pools,
)
from durable_unlearning.data.tofu import write_synthetic_tofu


def test_synthetic_data_prompt_variants_and_pools(tmp_path):
    forget, retain = write_synthetic_tofu(tmp_path / "tofu", forget_fraction=0.5, seed=0)
    assert forget.exists()
    assert retain.exists()
    assert read_jsonl(forget)
    variants = tmp_path / "prompts" / "variants.jsonl"
    write_prompt_variants(forget, variants)
    assert read_jsonl(variants)
    syntactic = tmp_path / "prompts" / "syntactic.jsonl"
    write_prompt_variants(forget, syntactic, mode="syntactic")
    syntactic_rows = read_jsonl(syntactic)
    assert syntactic_rows
    assert any("field query" in row["prompt"] for row in syntactic_rows)
    pools = write_relearn_pools(retain, tmp_path / "pools")
    assert {"retain_adjacent", "synthetic_biography", "syntax_matched", "random_general"} == set(pools)
    stress_pools = write_stress_relearn_pools(forget, tmp_path / "stress_pools", variants_per_item=2)
    assert {"forget_declarative", "forget_paraphrase_qa", "forget_mixed_stress"} == set(stress_pools)
    forget_rows = read_jsonl(forget)
    paraphrase_rows = read_jsonl(stress_pools["forget_paraphrase_qa"])
    assert len(paraphrase_rows) == 2 * len(forget_rows)
    assert any(forget_rows[0]["answer"] in row["text"] for row in paraphrase_rows)
    heldout_pools = write_heldout_stress_relearn_pools(forget, tmp_path / "heldout_stress_pools", variants_per_item=2)
    assert {
        "heldout_forget_declarative",
        "heldout_forget_paraphrase_qa",
        "heldout_forget_mixed_stress",
    } == set(heldout_pools)
    heldout_rows = read_jsonl(heldout_pools["heldout_forget_paraphrase_qa"])
    assert len(heldout_rows) == 2 * len(forget_rows)
    assert any(forget_rows[0]["answer"] in row["text"] for row in heldout_rows)
    assert heldout_rows[0]["source"] == "heldout_forget_paraphrase_qa"
    assert heldout_rows[0]["template_id"] == "heldout_paraphrase_qa"
