"""Headless checks for the sidecar and the contrast catalog."""
import json
import urllib.request

from sidecar.scoring import coherence_violations
from sidecar.server import make_context, serve_background


def test_accepting_a_wrong_patch_is_incoherent():
    found = coherence_violations("code_review", {"likely_correct": "no", "merge_action": "accept"})
    assert found == ["likely_correct=no but merge_action=accept"]
    assert coherence_violations("code_review", {"likely_correct": "no", "merge_action": "reject"}) == []


def test_catalog_builds_a_coherent_suite():
    from cases.build_catalog import build

    catalog = build()
    assert catalog["version"] == 1
    assert len(catalog["cases"]) == 16
    assert len(catalog["pairs"]) == 7
    assert len(catalog["replays"]) == 1
    assert sum(len(case["gold"]) for case in catalog["cases"]) == 73
    text_a = json.dumps(catalog, indent=2, ensure_ascii=False)
    text_b = json.dumps(build(), indent=2, ensure_ascii=False)
    assert text_a == text_b


def test_jev_maps_choice_and_score_onto_the_suite_labels():
    from sidecar.jev import answers_from_decisions, questions_for_family

    payload = questions_for_family("code_review")
    merge = payload["merge_action"]
    assert merge["type"] == "choice"
    assert list(merge["criteria"]) == ["accept", "request_changes", "needs_tests", "reject"]
    assert list(payload["likely_correct"]["criteria"]) == ["no", "yes"]
    assert payload["code_quality"]["type"] == "score"
    assert len(payload["code_quality"]["criteria"]) == 5

    questions = [
        {"id": "merge_action", "type": "choice", "options": list(merge["criteria"])},
        {"id": "code_quality", "type": "score", "options": ["0", "1", "2", "3", "4"]},
        {"id": "likely_correct", "type": "choice", "options": ["no", "yes"]},
    ]
    answers, missing = answers_from_decisions({
        "answers": {
            "merge_action": {
                "type": "choice",
                "choice": "accept",
                "confidence": 0.75,
                "probabilities": {
                    "accept": 0.8, "request_changes": 0.1, "needs_tests": 0.05, "reject": 0.05,
                },
            },
            "code_quality": {
                "type": "score",
                "score": 3.2,
                "confidence": 0.7,
                "probabilities": {"0": 0.01, "1": 0.02, "2": 0.07, "3": 0.62, "4": 0.28},
            },
        }
    }, questions)
    assert missing == ["likely_correct"]
    assert answers["merge_action"]["label"] == "accept"
    assert answers["merge_action"]["choice"] == "accept"
    assert answers["merge_action"]["answer_confidence"] == 0.75
    assert answers["code_quality"]["label"] == "3"
    assert abs(answers["code_quality"]["score"] - 3.2) < 1e-9


def test_compare_grades_laya_and_reports_a_missing_jev_key(monkeypatch):
    from cases.build_catalog import OUT, build

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    build()
    ctx = make_context(OUT, "mock", "unused", None)
    httpd = serve_background(ctx)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        health = _get(base + "/v1/health")
        assert health["jev_configured"] is False
        body = _post(base + "/v1/compare", {
            "case_id": "cr-paginate-good",
            "family": "code_review",
            "fields": ctx.cases["cr-paginate-good"]["fields"],
        })
        assert body["laya"]["answers"]["merge_action"]["label"] == "accept"
        assert body["jev"] is None
        assert "OPENROUTER_API_KEY" in body["jev_error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_sidecar_grades_scripted_cases_and_rejects_a_bad_family():
    from cases.build_catalog import OUT, build

    build()
    ctx = make_context(OUT, "mock", "unused", None)
    httpd = serve_background(ctx)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        health = _get(base + "/v1/health")
        assert health["backend"] == "mock"
        assert health["n_cases"] == 16
        assert health["model_id"] == "mock-scripted"

        body = _post(base + "/v1/grade", {
            "case_id": "cr-paginate-good",
            "family": "code_review",
            "fields": ctx.cases["cr-paginate-good"]["fields"],
        })
        assert body["source"] == "scripted"
        assert body["answers"]["merge_action"]["label"] == "accept"
        assert body["answers"]["likely_correct"]["choice"] == "yes"
        probs = body["answers"]["merge_action"]["probabilities"]
        assert abs(sum(probs.values()) - 1) < 1e-6
        assert body["state"]["family"] == "code_review"
        assert "paginate.py" in json.dumps(body["state"])

        heuristic = _post(base + "/v1/grade", {
            "family": "code_review",
            "fields": {"task": "Round tax.", "diff": "diff --git a/t.py b/t.py\n+x = 1\n", "tests": ""},
        })
        assert heuristic["source"] == "heuristic"
        assert heuristic["answers"]["merge_action"]["label"] == "needs_tests"

        status, err = _post_status(base + "/v1/grade", {"family": "nope", "fields": {}})
        assert status == 400
        assert "family" in err["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def _post(url, payload):
    status, body = _post_status(url, payload)
    assert status == 200, body
    return body


def _post_status(url, payload):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)
