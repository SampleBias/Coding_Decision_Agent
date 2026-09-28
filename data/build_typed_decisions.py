"""Build agent_trace cases from LocalLLaMA/typed-decisions (agent_trace_observability workflow).

The dataset already carries teacher probability distributions (`gold`), so no teacher pass is
needed. `needs_review` is a noul question in the source; we import it as the neutral-key choice
defined in configs/schema.py (false -> no, true -> yes).

Usage:
    python data/build_typed_decisions.py [--limit N] [--out path.jsonl]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import json

from data.common import (builder_argparser, default_out, load_hf, maybe_json, summarize_cases, take,
                         write_jsonl)
from data.compaction import agent_trace_state
from configs.schema import make_gold, new_case, questions_for

SOURCE = "LocalLLaMA/typed-decisions"
CONFIG = "agent_trace_observability"
QIDS = ["action", "needs_review", "outcome", "risk", "urgency"]


def convert_row(row, questions):
    state_src = maybe_json(row["state"])
    gold_src = maybe_json(row["gold"])
    state = agent_trace_state(
        task=state_src.get("task", ""),
        agent=state_src.get("agent"),
        constraints=state_src.get("constraints"),
        trace_summary=state_src.get("trace_summary"),
        extra={k: v for k, v in state_src.items() if k not in ("task", "agent", "constraints", "trace_summary")},
    )
    gold = {}
    for qid in QIDS:
        if qid not in gold_src:
            continue
        probs = dict(gold_src[qid]["probabilities"])
        if qid == "needs_review":
            probs = {"no": probs.get("false", 0.5), "yes": probs.get("true", 0.5)}
        gold[qid] = make_gold(questions[qid], probs, "gold")
    if not gold:
        return None
    case = new_case(
        case_id=f"td_{row['id']}",
        family="agent_trace",
        source=f"{SOURCE}/{CONFIG}",
        group=row["id"],
        state=state,
        qids=list(gold),
        gold=gold,
    )
    case["split_hint"] = row.get("split") or "train"
    return case


def main():
    ap = builder_argparser(__doc__)
    args = ap.parse_args()
    questions = questions_for("agent_trace")
    cases = []
    for split in ("train", "test"):
        ds = load_hf(SOURCE, CONFIG, split=split, cache_dir=args.cache_dir)
        for row in take(ds, args.limit):
            row = dict(row)
            row.setdefault("split", split)
            c = convert_row(row, questions)
            if c:
                cases.append(c)
    out = default_out("typed_decisions_agent_trace", args)
    n = write_jsonl(out, cases)
    print(f"wrote {n} cases -> {out}")
    print(json.dumps(summarize_cases(cases), indent=1)[:2000])


if __name__ == "__main__":
    main()
