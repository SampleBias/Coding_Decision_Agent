"""Build code_review (and routing) cases from coseal/CodeUltraFeedback.

Each instruction has 4 LLM responses rated 1-5 by GPT-3.5 against one coding preference
(instruction-following, explanation, complexity, readability, style).

  * code_review case per (instruction, response): `code_quality` from the rating (ordinal soft
    target), `instruction_followed` and `merge_action` derived from the rating and marked for
    teacher blending.
  * routing case per instruction: `model_tier` from which responder tiers scored >= 4,
    `task_difficulty` from the mean rating; `skill` is teacher-labelled.

Usage:
    python data/build_codeultrafeedback.py [--limit N] [--no-routing]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import json
import re
from typing import Dict, Optional

from data.common import (builder_argparser, default_out, load_hf, maybe_json, stable_id, summarize_cases,
                         take, write_jsonl)
from data.compaction import code_review_state, routing_state
from data.tiers import cheapest_tier_distribution, difficulty_from_solve_rate, tier_rates
from configs.schema import make_gold, new_case, option_keys, ordinal_soft, questions_for

SOURCE = "coseal/CodeUltraFeedback"
_FENCE = re.compile(r"```\s*([a-zA-Z0-9_+#-]+)")
_LANG_WORDS = re.compile(r"\b(python|javascript|typescript|java|c\+\+|c#|go|golang|rust|ruby|php|swift|kotlin|scala|"
                         r"sql|bash|shell|html|css|r|matlab|perl|haskell|lua|dart|elixir)\b", re.I)


def detect_language(instruction: str, response: str) -> str:
    m = _FENCE.search(response or "")
    if m and m.group(1).lower() not in ("text", "plaintext", "output", "bash", "sh", "console"):
        return m.group(1).lower()
    m = _LANG_WORDS.search(instruction or "")
    return m.group(1).lower() if m else ""


def parse_rating(r) -> Optional[float]:
    try:
        v = float(str(r).strip())
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 5 else None


def derived_followed(rating: float) -> Dict[str, float]:
    yes = {5: 0.92, 4: 0.80, 3: 0.55, 2: 0.25, 1: 0.08}[int(round(rating))]
    return {"no": 1 - yes, "yes": yes}


def derived_merge(rating: float) -> Dict[str, float]:
    return {
        5: {"accept": 0.80, "request_changes": 0.12, "needs_tests": 0.06, "reject": 0.02},
        4: {"accept": 0.50, "request_changes": 0.35, "needs_tests": 0.12, "reject": 0.03},
        3: {"accept": 0.12, "request_changes": 0.60, "needs_tests": 0.18, "reject": 0.10},
        2: {"accept": 0.03, "request_changes": 0.42, "needs_tests": 0.10, "reject": 0.45},
        1: {"accept": 0.01, "request_changes": 0.14, "needs_tests": 0.05, "reject": 0.80},
    }[int(round(rating))]


def convert(row, idx: int, cr_q, rt_q, with_routing: bool):
    instruction = row["instruction"]
    preference = str(row.get("preference") or "").replace("-", " ")
    responses = maybe_json(row["responses"]) or []
    annotations = maybe_json(row["annotations"]) or []
    ratings = {a.get("model"): parse_rating(a.get("rating")) for a in annotations if isinstance(a, dict)}
    group = stable_id(instruction)
    cases = []
    task = instruction if not preference else f"{instruction}\n\nPreference to judge against: {preference}."
    results = []
    for r in responses:
        model, text = r.get("model"), r.get("response", "")
        rating = ratings.get(model)
        if rating is None or not text.strip():
            continue
        results.append((model, 1.0 if rating >= 4 else 0.0))
        state = code_review_state(task=task, diff=text, language=detect_language(instruction, text), raw_code=True,
                                  notes=f"response produced by {model}")
        gold = {
            "code_quality": make_gold(cr_q["code_quality"], ordinal_soft(option_keys(cr_q["code_quality"]), rating - 1, 0.6),
                                      "rating"),
            "instruction_followed": make_gold(cr_q["instruction_followed"], derived_followed(rating), "derived"),
            "merge_action": make_gold(cr_q["merge_action"], derived_merge(rating), "derived"),
        }
        c = new_case(f"cuf_{group}_{stable_id(model)}", "code_review", SOURCE, group, state,
                     ["code_quality", "instruction_followed", "merge_action"], gold,
                     needs_teacher=["instruction_followed", "merge_action"])
        c["meta"] = {"preference": preference, "responder": model, "rating": rating}
        cases.append(c)
    if with_routing and results:
        rates = tier_rates(results)
        solve_rate = sum(s for _, s in results) / len(results)
        gold = {
            "model_tier": make_gold(rt_q["model_tier"], cheapest_tier_distribution(rates), "derived"),
            "task_difficulty": make_gold(rt_q["task_difficulty"], difficulty_from_solve_rate(solve_rate, len(results)),
                                         "derived"),
        }
        c = new_case(f"cuf_route_{group}", "routing", SOURCE, group, routing_state(instruction),
                     ["model_tier", "task_difficulty", "skill"], gold, needs_teacher=["task_difficulty", "skill"])
        cases.append(c)
    return cases


def main():
    ap = builder_argparser(__doc__)
    ap.add_argument("--no-routing", action="store_true")
    args = ap.parse_args()
    cr_q, rt_q = questions_for("code_review"), questions_for("routing")
    cases = []
    for i, row in enumerate(take(load_hf(SOURCE, split="train", cache_dir=args.cache_dir), args.limit)):
        cases.extend(convert(row, i, cr_q, rt_q, not args.no_routing))
    out = default_out("codeultrafeedback", args)
    n = write_jsonl(out, cases)
    print(f"wrote {n} cases -> {out}")
    print(json.dumps(summarize_cases(cases), indent=1)[:3000])


if __name__ == "__main__":
    main()
