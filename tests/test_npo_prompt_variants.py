from durable_unlearning.data.types import ForgetItem
from durable_unlearning.methods.npo import expand_with_prompt_variants


def test_expand_with_prompt_variants_uses_variant_as_question():
    item = ForgetItem(
        item_id="x",
        question="Original?",
        answer="Answer",
        prompt_variants=["Variant one?", "Variant two?"],
    )
    expanded = expand_with_prompt_variants([item])
    assert [row.question for row in expanded] == ["Variant one?", "Variant two?"]
    assert [row.item_id for row in expanded] == ["x", "x"]


def test_expand_with_prompt_variants_can_cap_variants():
    item = ForgetItem(
        item_id="x",
        question="Original?",
        answer="Answer",
        prompt_variants=["v1", "v2", "v3"],
    )
    expanded = expand_with_prompt_variants([item], max_variants=2)
    assert [row.question for row in expanded] == ["v1", "v2"]
