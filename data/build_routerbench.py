"""Build routing cases from RouterBench (withmartian/routerbench, 0-shot pickle) and, optionally,
LLMRouterBench (ynulihao/LLMRouterBench standardized JSON, if downloaded locally).

For every prompt we know which of N models answered correctly. Models are mapped to tiers
(data/tiers.py) and the `model_tier` target is P(cheapest tier that solves it); `task_difficulty`
comes from the overall solve rate. `skill` is hard-labelled `code_edit` for code benchmarks
(mbpp, LiveCodeBench, SWE-Bench) and teacher-labelled otherwise.

RouterBench prompts are mostly not coding requests; only the code evals are kept in full and
general English evals are capped (--general-cap) so the routing family stays coding-centric.

Usage:
    python data/build_routerbench.py [--limit N] [--general-cap 3000] [--llmrouterbench-dir DIR]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.common import builder_argparser, default_out, stable_id, summarize_cases, write_jsonl
from data.compaction import routing_state
from data.tiers import cheapest_tier_distribution, difficulty_from_solve_rate, tier_rates
from configs.schema import make_gold, new_case, one_hot, option_keys, questions_for

SOURCE_RB = "withmartian/routerbench"
SOURCE_LRB = "ynulihao/LLMRouterBench"
CODE_EVALS = re.compile(r"(mbpp|humaneval|livecodebench|lcb|swe|code|tau|bfcl)", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")


def is_english(text: str) -> bool:
    return len(_CJK.findall(text)) < max(3, len(text) // 200)


def prompt_text(p) -> str:
    if isinstance(p, (list, tuple)):
        return "\n".join(str(x) for x in p)
    return str(p)


def make_routing_case(case_id: str, source: str, group: str, request: str, context: Dict,
                      results: List[Tuple[str, float]], rt_q, skill_hard: Optional[str]) -> dict:
    rates = tier_rates(results)
    solve_rate = sum(s for _, s in results) / len(results)
    gold = {
        "model_tier": make_gold(rt_q["model_tier"], cheapest_tier_distribution(rates), "derived"),
        "task_difficulty": make_gold(rt_q["task_difficulty"], difficulty_from_solve_rate(solve_rate, len(results)), "derived"),
    }
    needs = ["task_difficulty"]
    if skill_hard:
        gold["skill"] = make_gold(rt_q["skill"], one_hot(option_keys(rt_q["skill"]), skill_hard, 0.08), "hard")
    else:
        needs.append("skill")
    c = new_case(case_id, "routing", source, group, routing_state(request, context),
                 ["model_tier", "task_difficulty", "skill"], gold, needs_teacher=needs)
    c["meta"] = {"n_models": len(results), "solve_rate": solve_rate, "context": context}
    return c


def build_routerbench(args, rt_q, rng) -> List[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SOURCE_RB, "routerbench_0shot.pkl", repo_type="dataset", cache_dir=args.cache_dir)
    df = pd.read_pickle(path)
    models = [c for c in df.columns if c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")
              and "|" not in c]
    code_rows, general_rows = [], []
    for _, r in df.iterrows():
        text = prompt_text(r["prompt"])
        if not is_english(text):
            continue
        (code_rows if CODE_EVALS.search(str(r["eval_name"])) else general_rows).append(r)
    rng.shuffle(general_rows)
    rows = code_rows + general_rows[: args.general_cap]
    if args.limit:
        rows = rows[: args.limit]
    cases = []
    for r in rows:
        results = [(m, float(r[m])) for m in models if r[m] == r[m]]  # skip NaN
        if not results:
            continue
        eval_name = str(r["eval_name"])
        skill = "code_edit" if CODE_EVALS.search(eval_name) else None
        cases.append(make_routing_case(f"rb_{stable_id(r['sample_id'])}", SOURCE_RB, eval_name, prompt_text(r["prompt"]),
                                       {"benchmark": eval_name}, results, rt_q, skill))
    return cases


def build_llmrouterbench(args, rt_q) -> List[dict]:
    """Best-effort reader for LLMRouterBench standardized instance JSON files.

    Expects files under --llmrouterbench-dir with records carrying at least `origin_query` and
    `score`; the model and dataset names are inferred from the path (…/<dataset>/<model>.json or
    …/<model>/<dataset>.json). Anything that does not parse is skipped with a note.
    """
    root = Path(args.llmrouterbench_dir)
    per_query: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {"results": [], "query": "", "dataset": ""})
    n_files = 0
    for f in list(root.rglob("*.json")) + list(root.rglob("*.jsonl")):
        try:
            if f.suffix == ".jsonl":
                recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            else:
                data = json.loads(f.read_text())
                recs = data if isinstance(data, list) else data.get("instances") or data.get("data") or []
        except Exception:
            continue
        if not recs or not isinstance(recs[0], dict) or "score" not in recs[0]:
            continue
        parts = f.relative_to(root).with_suffix("").parts
        model = recs[0].get("model") or parts[-1]
        dataset = recs[0].get("dataset") or (parts[-2] if len(parts) > 1 else "unknown")
        n_files += 1
        for rec in recs:
            q = rec.get("origin_query") or rec.get("prompt") or ""
            if not q:
                continue
            key = (dataset, stable_id(q))
            entry = per_query[key]
            entry["query"], entry["dataset"] = prompt_text(q), dataset
            entry["results"].append((str(model), float(rec["score"])))
    print(f"LLMRouterBench: parsed {n_files} files, {len(per_query)} distinct queries")
    cases = []
    for (dataset, qh), e in per_query.items():
        if len(e["results"]) < 3:
            continue
        skill = "code_edit" if CODE_EVALS.search(dataset) else None
        cases.append(make_routing_case(f"lrb_{qh}", SOURCE_LRB, dataset, e["query"], {"benchmark": dataset},
                                       e["results"], rt_q, skill))
    if args.limit:
        cases = cases[: args.limit]
    return cases


def main():
    ap = builder_argparser(__doc__)
    ap.add_argument("--general-cap", type=int, default=3000, help="max non-code RouterBench prompts")
    ap.add_argument("--llmrouterbench-dir", type=str, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rt_q = questions_for("routing")
    cases = build_routerbench(args, rt_q, rng)
    if args.llmrouterbench_dir:
        cases.extend(build_llmrouterbench(args, rt_q))
    out = default_out("routerbench_routing", args)
    n = write_jsonl(out, cases)
    print(f"wrote {n} cases -> {out}")
    print(json.dumps(summarize_cases(cases), indent=1)[:3000])


if __name__ == "__main__":
    main()
