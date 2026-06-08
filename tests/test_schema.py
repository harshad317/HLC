import pytest

from durable_unlearning.logging.schemas import SURVIVAL_FIELDS, validate_fields


def test_survival_schema_validation():
    row = {field: 0 for field in SURVIVAL_FIELDS}
    validate_fields(row, SURVIVAL_FIELDS)
    row.pop("step_t")
    with pytest.raises(ValueError):
        validate_fields(row, SURVIVAL_FIELDS)
