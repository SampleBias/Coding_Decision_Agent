"""Map model names to routing tiers and turn per-tier solve rates into a `model_tier` target."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from configs.schema import normalize

TIER_ORDER = ["small_fast", "mid", "frontier", "reasoning"]

_REASONING = re.compile(r"(^|[^a-z])(o1|o3|o4|r1|qwq|deepseek-reasoner|thinking|reasoning|grok-4|gpt-5)([^a-z]|$)", re.I)
_FRONTIER = re.compile(r"(gpt-4|gpt4|claude-(v2|2|3|opus|sonnet|4)|claude-large|gemini-(1\.5|2|2\.5)-pro|gemini-pro|"
                       r"deepseek-v3|qwen3-coder-480b|qwen-max|mistral-large|llama-3\.1-405b|405b|grok-3|command-r-plus)", re.I)
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z])", re.I)
_MOE = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b", re.I)


def model_tier(name: str) -> str:
    n = (name or "").lower()
    if _REASONING.search(n):
        return "reasoning"
    if _FRONTIER.search(n):
        return "frontier"
    if "gpt-3.5" in n or "claude-v1" in n or "claude-1" in n:
        return "mid"
    if "claude-instant" in n or "haiku" in n or "flash" in n or "mini" in n or "nano" in n:
        return "small_fast"
    m = _MOE.search(n)
    if m:
        total = float(m.group(1)) * float(m.group(2))
        return "mid" if total >= 10 else "small_fast"
    m = _SIZE.search(n)
    if m:
        size = float(m.group(1))
        return "small_fast" if size < 10 else "mid"
    return "mid"


def cheapest_tier_distribution(rates: Dict[str, float], floor: float = 0.03) -> Dict[str, float]:
    """P(tier t is the cheapest tier that solves the request), from per-tier solve rates.

    Walks tiers cheapest-first: p_t = (prob no cheaper tier solved) * rate_t. Whatever mass
    remains after `frontier` goes to `reasoning`. Tiers with no observations are skipped
    (their rate is unknown, not zero); a small floor keeps every option non-zero.
    """
    p: Dict[str, float] = {}
    rem = 1.0
    for t in TIER_ORDER[:-1]:
        r = rates.get(t)
        if r is None:
            p[t] = 0.0
            continue
        p[t] = rem * r
        rem *= (1.0 - r)
    p["reasoning"] = rem if "reasoning" not in rates else rem * max(rates["reasoning"], 0.5) + (1 - rem) * 0.0
    return normalize(p, TIER_ORDER, floor=floor)


def tier_rates(results: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    """[(model_name, solved 0/1 or score in [0,1])] -> {tier: mean score}"""
    agg: Dict[str, List[float]] = {}
    for name, score in results:
        agg.setdefault(model_tier(name), []).append(float(score))
    return {t: sum(v) / len(v) for t, v in agg.items()}


def difficulty_from_solve_rate(rate: Optional[float], n_models: int) -> Dict[str, float]:
    """Soft 4-level task_difficulty from the overall fraction of models that solved it."""
    from configs.schema import ordinal_soft

    if rate is None or n_models == 0:
        return {str(i): 0.25 for i in range(4)}
    # 1.0 -> trivial (0), 0.7 -> easy (1), 0.35 -> hard (2), 0 -> very hard (3)
    level = 3.0 * (1.0 - rate)
    return ordinal_soft(["0", "1", "2", "3"], level, sigma=0.7)
