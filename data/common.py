"""Shared helpers for the data pipeline: paths, JSONL I/O, config loading, hashing, CLI bits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"


def workdir() -> Path:
    """Root for all generated artifacts (WORKDIR env, default <repo>/artifacts)."""
    return Path(os.environ.get("WORKDIR", REPO_ROOT / "artifacts")).resolve()


def out_dir(sub: str = "") -> Path:
    p = workdir() / sub if sub else workdir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def compaction_config() -> Dict[str, Any]:
    return load_yaml(CONFIG_DIR / "state_compaction.yaml")


def load_train_config(profile: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """configs/train.yaml `common` merged with the chosen profile (TRAIN_PROFILE env by default)."""
    raw = load_yaml(CONFIG_DIR / "train.yaml")
    profile = profile or os.environ.get("TRAIN_PROFILE", "a100_80gb")
    if profile not in raw["profiles"]:
        raise KeyError(f"unknown profile {profile!r}; choose from {sorted(raw['profiles'])}")
    cfg = dict(raw["common"])
    cfg.update(raw["profiles"][profile])
    cfg["profile"] = profile
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def read_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def stable_id(*parts: Any) -> str:
    h = hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:16]


def maybe_json(x: Any) -> Any:
    """Decode a JSON string column if it is one; pass anything else through."""
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return x
    return x


def builder_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=None, help="output JSONL (default artifacts/cases/<name>.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="cap the number of source rows (smoke tests)")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--cache-dir", type=str, default=os.environ.get("HF_DATASETS_CACHE"))
    return p


def default_out(name: str, args: argparse.Namespace) -> Path:
    return args.out if args.out else out_dir("cases") / f"{name}.jsonl"


def load_hf(dataset: str, config: Optional[str] = None, split: str = "train", streaming: bool = False,
            cache_dir: Optional[str] = None, **kw):
    """Thin wrapper around datasets.load_dataset with a consistent signature."""
    from datasets import load_dataset

    return load_dataset(dataset, config, split=split, streaming=streaming, cache_dir=cache_dir, **kw)


def take(it: Iterable[Any], n: Optional[int]) -> List[Any]:
    if n is None:
        return list(it)
    out = []
    for x in it:
        out.append(x)
        if len(out) >= n:
            break
    return out


def summarize_cases(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-family / per-question label histogram for a quick sanity print."""
    from collections import Counter, defaultdict

    fam = Counter(r["family"] for r in rows)
    labels: Dict[str, Counter] = defaultdict(Counter)
    teacher = Counter()
    for r in rows:
        for qid, g in r["gold"].items():
            labels[f"{r['family']}.{qid}"][g["label"]] += 1
        for qid in r["needs_teacher"]:
            teacher[f"{r['family']}.{qid}"] += 1
    return {"n_cases": len(rows), "families": dict(fam),
            "labels": {k: dict(v) for k, v in labels.items()}, "needs_teacher": dict(teacher)}
