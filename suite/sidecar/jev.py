"""Jev (TypeSafe) as a second grader on the same cases as Laya-CDA.

Jev is called through the OpenRouter Decisions API. It is asked the training
questions from ``configs.schema`` — the same choice and score text Laya-CDA
was fine-tuned on — and the compacted state the Laya sidecar already built.
Yes/no stays a two-option choice (``no`` / ``yes``), matching the checkpoint,
so the two label sets can be compared question by question.

The key is ``OPENROUTER_API_KEY``. ``JEV_MODEL`` overrides the model id
(default ``~typesafe/jev-latest``).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sidecar.scoring import argmax_label, expected_score, tidy

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


class JevUnavailable(RuntimeError):
    """The Decisions API was not called, or it refused the request."""


def api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def model_name() -> str:
    return os.environ.get("JEV_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def configured() -> bool:
    return bool(api_key())


def questions_for_family(family: str) -> Dict[str, Any]:
    """Training question objects, deep-copied so a request cannot mutate the schema."""
    from configs.schema import FAMILIES

    if family not in FAMILIES:
        raise KeyError(family)
    return json.loads(json.dumps(FAMILIES[family]["questions"]))


def answers_from_decisions(
    body: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Map a Decisions response onto the suite's answer objects.

    Choice labels are Jev's stated ``choice``. Score labels are the argmax of
    the level probabilities, which is how ``train/evaluate.py`` scores Laya.
    """
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, dict):
        raise JevUnavailable("Jev response has no answers object")
    answers: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for question in questions:
        qid = question["id"]
        raw = raw_answers.get(qid)
        options = list(question["options"])
        if not isinstance(raw, dict):
            missing.append(qid)
            continue
        raw_probs = raw.get("probabilities")
        if not isinstance(raw_probs, dict):
            raw_probs = {}
        probs = tidy({key: _num(raw_probs.get(key, 0.0)) for key in options}, options)
        if question["type"] == "choice" and raw.get("choice") in options:
            label = str(raw["choice"])
        else:
            label = argmax_label(probs, options)
        confidence = raw.get("confidence")
        item: Dict[str, Any] = {
            "type": question["type"],
            "label": label,
            "probabilities": probs,
            "answer_confidence": _num(confidence) if confidence is not None else max(probs.values()),
        }
        if question["type"] == "choice":
            item["choice"] = label
        else:
            item["score"] = _num(raw["score"]) if raw.get("score") is not None else expected_score(probs, options)
        answers[qid] = item
    if not answers:
        raise JevUnavailable("Jev returned none of the requested questions")
    return answers, missing


def decide(family: str, state: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One Decisions call. ``questions`` is the wire list (id, type, options)."""
    import time

    payload = {
        "model": model_name(),
        "state": state,
        "questions": questions_for_family(family),
    }
    t0 = time.perf_counter()
    body = post_decisions(payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    answers, missing = answers_from_decisions(body, questions)
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    cost = usage.get("cost")
    resolved = str(body.get("model") or model_name())
    note = f"Jev Decisions API ({resolved})"
    if isinstance(cost, (int, float)):
        note = f"{note}  ${float(cost):.6f}"
    return {
        "ok": True,
        "backend": "jev",
        "source": "model",
        "model_id": resolved,
        "device": "openrouter",
        "family": family,
        "latency_ms": latency_ms,
        "state": state,
        "answers": answers,
        "missing": missing,
        "note": note,
        "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
    }


def post_decisions(payload: Mapping[str, Any]) -> Dict[str, Any]:
    key = api_key()
    if not key:
        raise JevUnavailable("OPENROUTER_API_KEY is not set")
    request = urllib.request.Request(
        DECISIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/S4MPL3BI4S/Coding_Decision_Agent",
            "X-OpenRouter-Title": "Laya-CDA suite",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            parsed = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise JevUnavailable(f"Jev API HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise JevUnavailable(f"Jev API unreachable: {exc.reason}") from None
    if not isinstance(parsed, dict):
        raise JevUnavailable("Jev response was not a JSON object")
    return parsed


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
