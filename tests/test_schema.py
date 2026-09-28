import pytest

from configs import schema as S


def test_all_families_have_valid_questions():
    for fam, spec in S.FAMILIES.items():
        assert spec["questions"], fam
        for qid, q in spec["questions"].items():
            assert q["type"] in ("choice", "score", "noul"), (fam, qid)
            keys = S.option_keys(q)
            assert 2 <= len(keys) <= S.MAX_CHOICE_OPTIONS, (fam, qid, keys)
            assert q["instructions"].strip()


def test_question_ids_unique_across_families():
    ids = S.all_question_ids()
    # 'action' and 'urgency' appear only in agent_trace; make sure no accidental dupes elsewhere
    assert len(ids) == len(set(ids))


def test_make_gold_normalises_and_labels():
    q = S.questions_for("code_review")["merge_action"]
    g = S.make_gold(q, {"accept": 3, "reject": 1}, "test")
    assert g["label"] == "accept"
    assert abs(sum(g["probabilities"].values()) - 1) < 1e-9
    assert set(g["probabilities"]) == {"accept", "request_changes", "needs_tests", "reject"}


def test_one_hot_and_ordinal():
    keys = ["0", "1", "2"]
    oh = S.one_hot(keys, "1", smoothing=0.1)
    assert oh["1"] == pytest.approx(0.9) and oh["0"] == pytest.approx(0.05)
    od = S.ordinal_soft(keys, 2.0)
    assert od["2"] > od["1"] > od["0"]
    assert abs(sum(od.values()) - 1) < 1e-9


def test_new_case_validates():
    q = S.questions_for("tool_call", ["tool_selection", "call_verdict"])
    gold = {"tool_selection": S.make_gold(q["tool_selection"], {"2": 1.0}, "programmatic")}
    row = S.new_case("c1", "tool_call", "unit", "g", {"request": "x", "tools": "t", "proposed_calls": "p"},
                     ["tool_selection", "call_verdict"], gold, needs_teacher=["call_verdict"])
    assert row["state"]["family"] == "tool_call"
    assert row["needs_teacher"] == ["call_verdict"]


def test_new_case_rejects_missing_gold():
    q = S.questions_for("routing", ["skill"])
    with pytest.raises(ValueError):
        S.new_case("c2", "routing", "unit", "g", {"request": "x"}, ["skill"], gold={}, needs_teacher=[])


def test_validate_rejects_bad_probability_keys():
    q = S.questions_for("routing")["model_tier"]
    row = S.new_case("c3", "routing", "unit", "g", {"request": "x"}, ["model_tier"],
                     {"model_tier": S.make_gold(q, {"mid": 1.0}, "hard")})
    row["gold"]["model_tier"]["probabilities"]["bogus"] = 0.0
    with pytest.raises(ValueError):
        S.validate_case(row)
