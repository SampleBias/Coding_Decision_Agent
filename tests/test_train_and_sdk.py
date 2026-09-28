"""CPU checks for the training helpers, the model-card renderer and the SDK grade object."""
import torch

from coding_decision_agent.grader import Grade
from coding_decision_agent import schemas as sdk_schema
from configs import schema as src_schema
from publish.push_to_hub import render_card
from train.batching import collate_train_batch
from train.calibrate import fit_one_temp, fit_temperatures
from train.train_rlcd import apply_family_weights, hold_out_calibration


def test_sdk_schema_matches_training_schema():
    assert sdk_schema.CODE_REVIEW_QUESTIONS == src_schema.CODE_REVIEW_QUESTIONS
    assert sdk_schema.TOOL_CALL_QUESTIONS == src_schema.TOOL_CALL_QUESTIONS
    assert sdk_schema.AGENT_TRACE_QUESTIONS == src_schema.AGENT_TRACE_QUESTIONS
    assert sdk_schema.ROUTING_QUESTIONS == src_schema.ROUTING_QUESTIONS


def test_grade_attribute_access():
    g = Grade("code_review", {
        "merge_action": {"type": "choice", "choice": "accept", "probabilities": {"accept": 0.8, "reject": 0.2},
                         "answer_confidence": 0.8},
        "code_quality": {"type": "score", "score": 3.2, "probabilities": {"3": 0.7, "4": 0.3}, "answer_confidence": 0.7},
    })
    assert g.merge_action == "accept"
    assert g.code_quality == 3.2
    assert g.confident("merge_action", 0.7) and not g.confident("merge_action", 0.9)
    assert g.label("code_quality") == "3"


def test_collate_pads_and_masks():
    items = [
        {"ids": [1, 2, 3], "markers": [1], "target": [1.0], "qtype": 0, "label": 0, "family": "routing"},
        {"ids": [4, 5], "markers": [1, 2], "target": [0.2, 0.8], "qtype": 1, "label": 1, "family": "code_review"},
    ]
    batch = collate_train_batch(items, pad_id=0)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["marker_mask"][0].tolist() == [True, False]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert abs(batch["target"][1, 1].item() - 0.8) < 1e-6


def test_temperature_fit_recovers_a_sharp_scale():
    # logits that are correct but over-confident: temperature should come out above 1
    pairs = []
    for _ in range(30):
        pairs.append((torch.tensor([8.0, 0.0, 0.0]), torch.tensor([0.6, 0.3, 0.1])))
    t = fit_one_temp(pairs)
    assert t > 1.5
    temps = fit_temperatures({0: pairs, 1: []})
    assert temps[1] == 1.2  # no data -> fallback, not a fit


def test_holdout_and_family_weights_are_deterministic():
    items = [{"family": "a", "id": i} for i in range(100)] + [{"family": "b", "id": i} for i in range(100)]
    train, calib = hold_out_calibration(items, 0.1, 400, seed=1)
    assert len(calib) == 20 and len(train) == 180
    train2, calib2 = hold_out_calibration(items, 0.1, 400, seed=1)
    assert [c["id"] for c in calib] == [c["id"] for c in calib2]
    weighted = apply_family_weights([{"family": "agent_trace"}, {"family": "code_review"}],
                                    {"agent_trace": 2.0, "code_review": 1.0}, seed=0)
    assert weighted.count({"family": "agent_trace"}) == 2


def test_model_card_fills_metrics():
    template = "acc {{ACCURACY}} brier {{BRIER}}\n{{FAMILY_TABLE}}\n"
    report = {"models": {"coding_decision_agent": {
        "ALL": {"accuracy": 0.812, "brier": 0.091, "ece": 0.05, "soft_accuracy": 0.7, "score_mae": 0.4},
        "code_review": {"n": 10, "accuracy": 0.8, "soft_accuracy": 0.7, "brier": 0.1, "ece": 0.05},
        "code_review.merge_action": {"n": 10, "accuracy": 0.8, "soft_accuracy": 0.7, "brier": 0.1, "ece": 0.05},
    }}}
    text = render_card(template, report)
    assert "0.812" in text and "0.091" in text
    assert "code_review |" in text and "merge_action" not in text
