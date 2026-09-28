"""Tokenize case files into RLCD training items with laya.common.build_sequence.

Mirrors the official fine-tuning notebook: one item per (case, question) with the token ids,
the [MASK] marker positions (one per option), the question type id, the soft target and the
argmax label. Items whose marker count does not match the option count (options squeezed out by
the head budget) are dropped and counted.

Usage:
    python data/preprocess.py --dataset artifacts/dataset --out artifacts/items \
        [--profile a100_80gb] [--base-model convaiinnovations/laya] [--splits train,val]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.common import load_train_config, read_jsonl
from configs.schema import option_keys


def load_base(base_model: str, revision: Optional[str] = None, cache_dir: Optional[str] = None):
    """Download the base checkpoint and return (model_dir, tokenizer, rl_agent_config)."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config

    model_dir = snapshot_download(base_model, revision=revision, cache_dir=cache_dir)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    return model_dir, tok, cfg


def to_laya_question(q: Dict[str, Any]) -> Dict[str, Any]:
    return {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria", {})}


def build_items(cases, tok, max_len: int, head_max_len: int, stats: Counter) -> List[Dict[str, Any]]:
    from laya.common import QTYPES, build_sequence, encode_text, serialize_state

    items = []
    mask_tok = tok.mask_token
    for case in cases:
        state_text = serialize_state(case["state"]).replace(mask_tok, " ")
        state_ids = encode_text(tok, state_text, add_special_tokens=False)["input_ids"]
        stats["state_tokens_total"] += len(state_ids)
        for qid, q in case["questions"].items():
            g = case["gold"].get(qid)
            if g is None:
                stats["skipped_no_gold"] += 1
                continue
            keys = option_keys(q)
            target = [float(g["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            seq, markers = build_sequence(tok, case["state"], to_laya_question(q), max_len, head_max_len,
                                          state_ids=state_ids)
            if len(markers) != len(keys):
                stats["dropped_marker_mismatch"] += 1
                continue
            room = max_len - (markers[-1] + 1) - 2
            if len(state_ids) > room:
                stats["state_truncated"] += 1
            items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["type"]],
                "target": target,
                "label": int(max(range(len(target)), key=lambda i: target[i])),
                "family": case["family"],
                "qid": qid,
                "case_id": case["case_id"],
                "source": case["source"],
            })
            stats[f"items.{case['family']}.{qid}"] += 1
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="dir with cases_{train,val,test}.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    args = ap.parse_args()

    import torch

    cfg = load_train_config(args.profile, {"base_model": args.base_model})
    model_dir, tok, base_cfg = load_base(cfg["base_model"], cfg.get("base_revision"), args.cache_dir)
    max_len, head_max_len = int(cfg["max_len"]), int(cfg["head_max_len"])
    print(f"base={cfg['base_model']} ({base_cfg.get('encoder')}) max_len={max_len} head_max_len={head_max_len}")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"base_model": cfg["base_model"], "model_dir": model_dir, "max_len": max_len, "head_max_len": head_max_len,
               "profile": cfg["profile"], "splits": {}}
    for split in args.splits.split(","):
        path = args.dataset / f"cases_{split}.jsonl"
        if not path.exists():
            print(f"skip {split}: {path} missing")
            continue
        cases = list(read_jsonl(path))
        if args.limit:
            cases = cases[: args.limit]
        stats: Counter = Counter()
        items = build_items(cases, tok, max_len, head_max_len, stats)
        torch.save(items, args.out / f"items_{split}.pt")
        n_states = max(1, len(cases))
        info = {"cases": len(cases), "items": len(items),
                "avg_state_tokens": round(stats["state_tokens_total"] / n_states, 1),
                "state_truncated_frac": round(stats["state_truncated"] / max(1, len(items)), 4),
                "dropped_marker_mismatch": stats["dropped_marker_mismatch"],
                "per_question": {k.split("items.")[1]: v for k, v in stats.items() if k.startswith("items.")}}
        summary["splits"][split] = info
        print(f"{split}: {info['cases']} cases -> {info['items']} items | avg state tokens {info['avg_state_tokens']} | "
              f"truncated {info['state_truncated_frac']:.1%} | dropped {info['dropped_marker_mismatch']}")
    (args.out / "preprocess_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
