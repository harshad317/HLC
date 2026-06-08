from durable_unlearning.methods.hlc_sg import parse_resurrection_penalty_steps


def test_parse_resurrection_penalty_steps_defaults_to_terminal():
    assert parse_resurrection_penalty_steps(None, 8) == [8]
    assert parse_resurrection_penalty_steps("terminal", 8) == [8]


def test_parse_resurrection_penalty_steps_supports_all_and_pow2():
    assert parse_resurrection_penalty_steps("all", 4) == [1, 2, 3, 4]
    assert parse_resurrection_penalty_steps("powers_of_two", 8) == [1, 2, 4, 8]
    assert parse_resurrection_penalty_steps("1,2,4,k", 8) == [1, 2, 4, 8]


def test_parse_resurrection_penalty_steps_rejects_out_of_range():
    try:
        parse_resurrection_penalty_steps("0,9", 8)
    except ValueError as exc:
        assert "must be in" in str(exc)
    else:
        raise AssertionError("expected out-of-range penalty steps to fail")
