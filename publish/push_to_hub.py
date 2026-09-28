"""Push the trained checkpoint to Hugging Face as S4MPL3BI4S/Coding_Decision_Agent.

Uploads model.safetensors, encoder/, tokenizer/, rl_agent_config.json and a model card rendered
from huggingface/README.md. Needs a write HF_TOKEN.

    python publish/push_to_hub.py --checkpoint outputs/coding_decision_agent \
        --report outputs/coding_decision_agent/benchmark_report.json \
        [--dataset artifacts/dataset --dataset-repo S4MPL3BI4S/coding-decision-cases]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import shutil

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "huggingface" / "README.md"


def render_card(template: str, report: dict | None) -> str:
    metrics = {}
    if report:
        metrics = (report.get("models") or {}).get("coding_decision_agent", {}).get("ALL", {})
    def fmt(key, default="TBD"):
        v = metrics.get(key)
        return f"{v:.3f}" if isinstance(v, float) else default
    out = template
    for key, token in (("accuracy", "ACCURACY"), ("soft_accuracy", "SOFT_ACCURACY"),
                       ("brier", "BRIER"), ("ece", "ECE"), ("score_mae", "SCORE_MAE")):
        out = out.replace("{{" + token + "}}", fmt(key))
    per = ""
    if report:
        fam = (report.get("models") or {}).get("coding_decision_agent", {})
        lines = ["| slice | n | accuracy | soft acc | brier | ECE |", "|---|---:|---:|---:|---:|---:|"]
        for name, m in fam.items():
            if name == "ALL" or "." in name:
                continue
            ece = m.get("ece")
            lines.append(f"| {name} | {m.get('n', 0)} | {m.get('accuracy', 0):.3f} | "
                         f"{m.get('soft_accuracy', 0):.3f} | {m.get('brier', 0):.3f} | "
                         f"{'' if ece is None else f'{ece:.3f}'} |")
        per = "\n".join(lines)
    return out.replace("{{FAMILY_TABLE}}", per or "_Run `train/evaluate.py` to fill this table._")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO", "S4MPL3BI4S/Coding_Decision_Agent"))
    ap.add_argument("--dataset", type=Path, default=None, help="dir with cases_{train,val,test}.jsonl")
    ap.add_argument("--dataset-repo", default=os.environ.get("HF_DATASET_REPO", "S4MPL3BI4S/coding-decision-cases"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export a write token for the S4MPL3BI4S account.")
    need = ["model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"]
    missing = [n for n in need if not (args.checkpoint / n).exists()]
    if missing:
        raise SystemExit(f"{args.checkpoint} is missing {missing}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    report = json.loads(args.report.read_text()) if args.report and args.report.exists() else None
    card = render_card(CARD.read_text(), report)
    staging = args.checkpoint / "_hub"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name in need:
        src = args.checkpoint / name
        dst = staging / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    (staging / "README.md").write_text(card)
    if report:
        (staging / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    api.upload_folder(folder_path=str(staging), repo_id=args.repo, repo_type="model")
    shutil.rmtree(staging)
    print(f"model: https://huggingface.co/{args.repo}")

    if args.dataset:
        from datasets import Dataset, DatasetDict
        from data.merge_and_split import to_hub_rows
        from data.common import read_jsonl

        splits = {}
        for name in ("train", "val", "test"):
            path = args.dataset / f"cases_{name}.jsonl"
            if path.exists():
                splits[name] = Dataset.from_list(to_hub_rows(list(read_jsonl(path)), name))
        if not splits:
            raise SystemExit(f"no cases_*.jsonl in {args.dataset}")
        DatasetDict(splits).push_to_hub(args.dataset_repo, token=token, private=args.private)
        print(f"dataset: https://huggingface.co/datasets/{args.dataset_repo}")


if __name__ == "__main__":
    main()
