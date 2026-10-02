"""Question schema in the order the model was trained on.

Option order is the label-index order from `configs.schema.option_keys`.
The catalog stores a copy so the TUI can draw bars without importing Python.
The sidecar refuses to start if that copy disagrees with the live schema.
"""
from __future__ import annotations

from typing import Any, Dict, List


def wire_families() -> List[Dict[str, Any]]:
    from configs.schema import FAMILIES, option_keys

    families = []
    for family, spec in FAMILIES.items():
        questions = []
        for qid, question in spec["questions"].items():
            keys = option_keys(question)
            if question["type"] == "score":
                gloss = [item.split(":")[0].strip() for item in question["criteria"]]
            else:
                gloss = list(keys)
            if len(gloss) != len(keys):
                raise RuntimeError(f"{family}.{qid} gloss/option length mismatch")
            questions.append({
                "id": qid,
                "type": question["type"],
                "instructions": question["instructions"],
                "options": keys,
                "gloss": gloss,
            })
        families.append({
            "id": family,
            "description": spec["description"],
            "questions": questions,
        })
    return families


def questions_by_id(families: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {fam["id"]: {q["id"]: q for q in fam["questions"]} for fam in families}


def assert_same_schema(catalog_families: List[Dict[str, Any]], live_families: List[Dict[str, Any]]) -> None:
    """Raise ValueError when the catalog copy is not the live training schema."""
    cat = [(f["id"], [(q["id"], q["type"], list(q["options"])) for q in f["questions"]]) for f in catalog_families]
    live = [(f["id"], [(q["id"], q["type"], list(q["options"])) for q in f["questions"]]) for f in live_families]
    if cat != live:
        raise ValueError(
            "suite/cases/catalog.json does not match configs/schema.py. "
            "Rebuild it with: python3 suite/cases/build_catalog.py"
        )
