"""Teacher labelling with a local LLM served by vLLM (OpenAI-compatible API).

For every (case, question) listed in `needs_teacher`, the teacher is shown the compacted state
and the question with lettered options and asked for a single letter. We read the *logprobs of
the first generated token* over the option letters, which yields a full probability
distribution (a soft target) instead of one sampled answer. Choice questions are asked twice
with the option order reversed to cancel position bias; score questions keep their ordinal
order.

The teacher distribution is blended with whatever signal the builder already produced:

    gold = alpha(signal) * dataset_signal + (1 - alpha) * teacher

Results are cached per (teacher model, case_id, qid) so the script is resumable.

Usage (server started by runpod/serve_teacher.sh):
    python data/teacher_vllm.py --in artifacts/cases/*.jsonl --out artifacts/cases_teacher/ \
        [--base-url http://127.0.0.1:8000/v1] [--model Qwen/Qwen3-32B-AWQ] [--workers 32]
    python data/teacher_vllm.py --in ... --out ... --mock     # offline pipeline test, no server
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import glob
import json
import math
import os
import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common import read_jsonl, stable_id, write_jsonl
from configs.schema import FAMILIES, blend, make_gold, normalize, option_keys, validate_case

LETTERS = string.ascii_uppercase

# How much to trust the builder's signal when a teacher distribution is also available.
ALPHA_BY_SIGNAL = {
    "gold": 1.0,                  # already a teacher distribution; never overwrite
    "hard": 0.85,                 # ground truth label (e.g. tests passed)
    "programmatic+judges": 0.75,
    "programmatic": 0.70,
    "rating": 0.65,
    "derived": 0.45,              # heuristic mapping; teacher gets the larger share
}

SYSTEM_PROMPT = (
    "You are a meticulous senior engineer acting as the judge inside an AI coding agent. "
    "You grade code changes, tool calls, agent traces and routing decisions. "
    "You answer every question with exactly one option letter and nothing else."
)

FAMILY_INTRO = {
    "code_review": "Review the following code change against the task.",
    "tool_call": "Review the assistant's proposed tool call(s) for the user's request, given the available tools.",
    "agent_trace": "Triage the following AI agent execution trace.",
    "routing": "Decide how a coding-agent orchestrator should route the following request.",
}


# --------------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------------

def render_state(state: Dict[str, Any]) -> str:
    parts = []
    for k, v in state.items():
        if k == "family":
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False, indent=None)
        parts.append(f"## {k}\n{v}")
    return "\n\n".join(parts)


def option_texts(q: Dict[str, Any]) -> List[Tuple[str, str]]:
    """[(key, text)] in canonical order."""
    t = q["type"]
    if t == "choice":
        return [(k, f"{k}: {v}" if v else k) for k, v in q["criteria"].items()]
    if t == "score":
        return [(str(i), f"level {i}: {c}") for i, c in enumerate(q["criteria"])]
    return [("false", "no"), ("true", "yes")]


def build_prompt(family: str, state: Dict[str, Any], q: Dict[str, Any], order: Sequence[int]) -> Tuple[str, List[str]]:
    opts = option_texts(q)
    lines = [FAMILY_INTRO.get(family, ""), "", render_state(state), "", f"# Question\n{q['instructions']}", ""]
    keys_in_order = []
    for letter, idx in zip(LETTERS, order):
        key, text = opts[idx]
        keys_in_order.append(key)
        lines.append(f"{letter}. {text}")
    lines += ["", "Answer with the single letter of the best option."]
    return "\n".join(lines), keys_in_order


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------

class VLLMBackend:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0, max_prompt_chars: int = 24000):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("TEACHER_API_KEY", "EMPTY"), timeout=timeout)
        self.model = model
        self.max_prompt_chars = max_prompt_chars
        self.is_qwen3 = "qwen3" in model.lower()

    def letter_logprobs(self, prompt: str, n_options: int) -> Dict[str, float]:
        """Return {letter: logprob} for the first generated token (top-20 candidates)."""
        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[: self.max_prompt_chars] + "\n[...truncated...]\nAnswer with the single letter of the best option."
        extra: Dict[str, Any] = {}
        if self.is_qwen3:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20, extra_body=extra or None,
                )
                choice = resp.choices[0]
                top = choice.logprobs.content[0].top_logprobs if choice.logprobs and choice.logprobs.content else []
                out: Dict[str, float] = {}
                for cand in top:
                    tok = (cand.token or "").strip().upper().rstrip(".):")
                    if len(tok) == 1 and tok in LETTERS[:n_options]:
                        out[tok] = max(out.get(tok, -1e9), float(cand.logprob))
                return out
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return {}


class MockBackend:
    """Offline stand-in: deterministic pseudo-random distributions (for pipeline tests only)."""

    model = "mock-teacher"

    def letter_logprobs(self, prompt: str, n_options: int) -> Dict[str, float]:
        rng = random.Random(stable_id(prompt))
        raw = [rng.random() * 3 for _ in range(n_options)]
        return {LETTERS[i]: r for i, r in enumerate(raw)}


def letters_to_probs(lp: Dict[str, float], keys_in_order: List[str]) -> Tuple[Dict[str, float], float]:
    """Softmax over the option letters that appeared; returns (probs by key, captured mass proxy)."""
    if not lp:
        return {k: 1.0 / len(keys_in_order) for k in keys_in_order}, 0.0
    vals = {LETTERS[i]: lp.get(LETTERS[i]) for i in range(len(keys_in_order))}
    present = {l: v for l, v in vals.items() if v is not None}
    m = max(present.values())
    exps = {l: math.exp(v - m) for l, v in present.items()}
    z = sum(exps.values())
    probs = {}
    for i, k in enumerate(keys_in_order):
        l = LETTERS[i]
        probs[k] = exps[l] / z if l in exps else 0.0
    captured = sum(math.exp(v) for v in present.values())  # absolute mass on letters (<= 1)
    return probs, min(captured, 1.0)


# --------------------------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------------------------

class TeacherCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            for r in read_jsonl(path):
                self.data[r["key"]] = r
        self.fh = open(path, "a")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def put(self, key: str, rec: Dict[str, Any]) -> None:
        rec = dict(rec, key=key)
        with self.lock:
            self.data[key] = rec
            self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.fh.flush()


def teach_one(backend, case: Dict[str, Any], qid: str, permutations: int) -> Dict[str, Any]:
    q = case["questions"][qid]
    keys = option_keys(q)
    n = len(keys)
    orders = [list(range(n))]
    if q["type"] == "choice" and permutations > 1:
        orders.append(list(reversed(range(n))))
    agg = {k: 0.0 for k in keys}
    captured = []
    for order in orders:
        prompt, keys_in_order = build_prompt(case["family"], case["state"], q, order)
        lp = backend.letter_logprobs(prompt, n)
        probs, cap = letters_to_probs(lp, keys_in_order)
        captured.append(cap)
        for k, p in probs.items():
            agg[k] += p / len(orders)
    return {"probabilities": normalize(agg, keys), "captured": sum(captured) / len(captured), "n_queries": len(orders)}


def apply_teacher(case: Dict[str, Any], qid: str, teacher: Dict[str, Any], min_captured: float) -> None:
    q = case["questions"][qid]
    keys = option_keys(q)
    tp = teacher["probabilities"]
    weak = teacher["captured"] < min_captured
    if qid in case["gold"]:
        g = case["gold"][qid]
        alpha = ALPHA_BY_SIGNAL.get(g["signal"], 0.5)
        if weak:
            alpha = max(alpha, 0.9)
        probs = blend(g["probabilities"], tp, alpha, keys)
        signal = g["signal"] + "+teacher"
    else:
        probs = tp if not weak else normalize({k: 1.0 for k in keys}, keys)
        signal = "teacher" if not weak else "teacher_weak"
    new = make_gold(q, probs, signal)
    new["teacher_probabilities"] = {k: round(v, 4) for k, v in tp.items()}
    new["teacher_captured"] = round(teacher["captured"], 4)
    case["gold"][qid] = new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="case JSONL files or globs")
    ap.add_argument("--out", type=Path, required=True, help="output directory (same file names)")
    ap.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-32B-AWQ"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--permutations", type=int, default=2, help="1 or 2 (reverse order pass for choice questions)")
    ap.add_argument("--min-captured", type=float, default=0.25, help="letter mass below which the teacher is distrusted")
    ap.add_argument("--limit", type=int, default=None, help="cap cases per input file (smoke)")
    ap.add_argument("--mock", action="store_true", help="offline mock teacher")
    ap.add_argument("--cache", type=Path, default=None, help="cache JSONL (default <out>/_teacher_cache.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt per family and exit")
    args = ap.parse_args()

    files: List[str] = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)) or [pat])
    args.out.mkdir(parents=True, exist_ok=True)
    backend = MockBackend() if args.mock else VLLMBackend(args.base_url, args.model)
    cache = TeacherCache(args.cache or (args.out / "_teacher_cache.jsonl"))
    model_tag = backend.model

    if args.dry_run:
        seen = set()
        for f in files:
            for case in read_jsonl(f):
                if case["family"] in seen or not case["needs_teacher"]:
                    continue
                seen.add(case["family"])
                qid = case["needs_teacher"][0]
                prompt, _ = build_prompt(case["family"], case["state"], case["questions"][qid], list(range(len(option_keys(case["questions"][qid])))))
                print(f"\n===== {case['family']} / {qid} ({len(prompt)} chars) =====\n{prompt}\n")
        return

    total_jobs = done_jobs = 0
    t0 = time.time()
    for f in files:
        cases = list(read_jsonl(f))
        if args.limit:
            cases = cases[: args.limit]
        jobs = [(ci, qid) for ci, c in enumerate(cases) for qid in c["needs_teacher"]]
        total_jobs += len(jobs)

        def run(job):
            ci, qid = job
            case = cases[ci]
            key = f"{model_tag}|{case['case_id']}|{qid}"
            rec = cache.get(key)
            if rec is None:
                rec = teach_one(backend, case, qid, args.permutations)
                cache.put(key, rec)
            return ci, qid, rec

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                ci, qid, rec = fut.result()
                apply_teacher(cases[ci], qid, rec, args.min_captured)
                done_jobs += 1
                if i % 500 == 0 or i == len(futs):
                    rate = done_jobs / max(1e-6, time.time() - t0)
                    print(f"[{Path(f).name}] {i}/{len(jobs)} teacher queries  ({rate:.1f}/s)", flush=True)
        for c in cases:
            c["needs_teacher"] = []
            validate_case(c)
        out_path = args.out / Path(f).name
        write_jsonl(out_path, cases)
        print(f"wrote {len(cases)} cases -> {out_path}")
    print(f"done: {done_jobs}/{total_jobs} teacher jobs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
