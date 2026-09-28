"""Temperature calibration for a Laya decision head.

One scalar per question type (choice, score, noul), fit by LBFGS on the log-temperature so the
softmax of logits/T matches the soft targets. Fitted on a slice that never entered a training
batch. Argmax (accuracy) is unchanged; confidence is what moves.

Also usable on its own to refit temperatures for an already trained checkpoint:

    python train/calibrate.py --items artifacts/items/items_val.pt --checkpoint outputs/run \
        --out outputs/run
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

QTYPE_NAMES = ["choice", "score", "noul"]


def fit_one_temp(sel: Sequence[Tuple[torch.Tensor, torch.Tensor]], min_items: int = 10) -> float:
    """Fit one temperature for a list of (logits[k], target[k]) pairs. Returns a value in [0.1, 10]."""
    if len(sel) < min_items:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.as_tensor(z, dtype=torch.float32)
        T[i, : len(t)] = torch.as_tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def fit_temperatures(grouped: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]) -> List[float]:
    """grouped: qtype id -> pairs. Missing types stay at 1.2 (the notebook's fallback)."""
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = grouped.get(qt) or []
        if sel:
            temps[qt] = fit_one_temp(sel)
    return temps


def write_config(cfg: dict, out_dir: Path, temps: List[float], model_name: str, max_len: int,
                 head_max_len: int) -> None:
    """Write rl_agent_config.json. Drops temperature_by_options so inherited buckets cannot mask the fit."""
    cfg = dict(cfg)
    cfg["fine_tuned"] = True
    cfg["model_name"] = model_name
    cfg["temperature"] = [round(t, 6) for t in temps]
    cfg.pop("temperature_by_options", None)
    cfg["max_len"] = int(max_len)
    cfg["head_max_len"] = int(head_max_len)
    cfg["head_max_len_train"] = int(head_max_len)
    cfg["gradient_checkpointing"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _collect(model, items, tok, device, amp_dtype) -> Dict[int, list]:
    from train.batching import collate_train_batch

    grouped: Dict[int, list] = {0: [], 1: [], 2: []}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits, _ = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device),
                )
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                grouped[int(it["qtype"])].append((logits[r, :k], torch.tensor(it["target"])))
    return grouped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="items_*.pt to fit on (a held-out split)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="directory with model.safetensors + encoder/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.checkpoint / "rl_agent_config.json") as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer")
    model = build_model(cfg, encoder_dir=str(args.checkpoint / "encoder"))
    model.load_state_dict(load_file(args.checkpoint / "model.safetensors"), strict=True)
    model.to(device).eval()
    items = torch.load(args.items, weights_only=False)
    temps = fit_temperatures(_collect(model, items, tok, device, torch.float32))
    print("temperatures (choice, score, noul):", [round(t, 4) for t in temps])
    write_config(cfg, args.out, temps, cfg.get("model_name", "coding-decision-agent"),
                 cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
    if args.out.resolve() != args.checkpoint.resolve():
        print(f"config written to {args.out}; copy model.safetensors yourself or point --out at the checkpoint")
    else:
        print(f"updated {args.out / 'rl_agent_config.json'}")


if __name__ == "__main__":
    main()
