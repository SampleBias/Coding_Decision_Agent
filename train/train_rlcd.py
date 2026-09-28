"""RLCD fine-tune of a Laya checkpoint. Generalises the official 2xT4 notebook:

* works under `torchrun --nproc_per_node=N` for N>=1, and as a plain process on one GPU
* reads configs/train.yaml (--profile a100_80gb | gpu_24gb | smoke)
* holds a calibration slice out of training and fits one temperature per question type
* optional PEFT LoRA on the ModernBERT encoder (--lora / the gpu_24gb profile); the saved
  checkpoint is always a plain Laya directory that `laya.load` can read
* rolling checkpoint_latest/ after every epoch, resumed automatically
* optional Weights & Biases (--wandb)

    torchrun --standalone --nproc_per_node=1 train/train_rlcd.py \
        --items artifacts/items/items_train.pt --out outputs/coding_decision_agent --profile a100_80gb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

import torch

from data.common import load_train_config
from data.preprocess import load_base
from train.batching import collate_train_batch
from train.calibrate import fit_temperatures, write_config


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def barrier():
    d = _dist()
    if d is not None:
        d.barrier()


def is_main() -> bool:
    d = _dist()
    return d is None or d.get_rank() == 0


def log(msg: str):
    if is_main():
        print(msg, flush=True)


def apply_family_weights(items: List[dict], weights: Dict[str, float], seed: int) -> List[dict]:
    rng = random.Random(seed)
    out = []
    for it in items:
        w = float(weights.get(it.get("family", ""), 1.0))
        copies = int(w)
        if rng.random() < w - copies:
            copies += 1
        out.extend([it] * copies)
    return out


def hold_out_calibration(items: List[dict], fraction: float, cap_per_family: int, seed: int):
    rng = random.Random(seed)
    by_fam = defaultdict(list)
    for i, it in enumerate(items):
        by_fam[it.get("family", "")].append(i)
    calib = set()
    for idxs in by_fam.values():
        rng.shuffle(idxs)
        n = min(cap_per_family, max(1, int(len(idxs) * fraction))) if len(idxs) >= 10 else 0
        calib.update(idxs[:n])
    return [items[i] for i in range(len(items)) if i not in calib], [items[i] for i in sorted(calib)]


def maybe_lora(model, cfg):
    """Attach a fresh LoRA adapter. Resume loads the merged checkpoint first, so the adapter
    starts at zero on top of weights that already include earlier LoRA updates."""
    if not cfg.get("lora"):
        return model
    from peft import LoraConfig, get_peft_model

    model.encoder = get_peft_model(model.encoder, LoraConfig(
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_targets"]), bias="none",
    ))
    if is_main():
        model.encoder.print_trainable_parameters()
    return model


def enable_checkpointing(model, cfg):
    if not cfg.get("gradient_checkpointing"):
        return
    enc = model.encoder.get_base_model() if hasattr(model.encoder, "get_base_model") else model.encoder
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True


def laya_state_dict(model, template_keys: List[str]) -> Dict[str, torch.Tensor]:
    """Full fp16 state dict with LoRA folded in, restricted to the original Laya keys."""
    peft = hasattr(model.encoder, "merge_adapter")
    if peft:
        model.encoder.merge_adapter()
        full = {}
        for k, v in model.encoder.get_base_model().state_dict().items():
            if ".lora_" in k:
                continue
            full["encoder." + k.replace(".base_layer", "")] = v
        for k, v in model.state_dict().items():
            if not k.startswith("encoder."):
                full[k] = v
        model.encoder.unmerge_adapter()
    else:
        full = dict(model.state_dict())
    missing = [k for k in template_keys if k not in full]
    if missing:
        raise RuntimeError(f"merged state dict is missing {len(missing)} keys, e.g. {missing[:5]}")
    return {k: full[k].detach().half().contiguous().cpu() for k in template_keys}


def save_checkpoint(model, tok, base_cfg, train_cfg, out_dir: Path, template_keys, epoch: int,
                    avg_loss: float, temps=None):
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(laya_state_dict(model, template_keys), str(out_dir / "model.safetensors"))
    enc = model.encoder.get_base_model() if hasattr(model.encoder, "get_base_model") else model.encoder
    enc.config.save_pretrained(out_dir / "encoder")
    tok.save_pretrained(out_dir / "tokenizer")
    temperatures = temps if temps is not None else base_cfg.get("temperature", [1.0, 1.0, 1.0])
    write_config(base_cfg, out_dir, list(temperatures), train_cfg["model_name"],
                 train_cfg["max_len"], train_cfg["head_max_len"])
    meta = {"epoch": epoch, "avg_loss": avg_loss, "lora": bool(train_cfg.get("lora")),
            "profile": train_cfg.get("profile")}
    (out_dir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"  saved checkpoint epoch {epoch} -> {out_dir}")


def train():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default=os.environ.get("TRAIN_PROFILE"))
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--lora", action="store_true", help="force LoRA even if the profile disables it")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU is visible. Training runs on a RunPod GPU, not on this machine.\n"
            "Follow docs/01_RUNPOD_TRAINING.md."
        )

    cfg = load_train_config(args.profile, {"base_model": args.base_model, "epochs": args.epochs,
                                           "max_items": args.max_items})
    if args.lora:
        cfg["lora"] = True
    if args.no_lora:
        cfg["lora"] = False
    if args.wandb:
        cfg["wandb"] = True

    launched = "RANK" in os.environ
    if launched:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file

    model_dir, tok, base_cfg = load_base(cfg["base_model"], cfg.get("base_revision"))
    base_cfg["max_len"] = int(cfg["max_len"])
    base_cfg["head_max_len"] = int(cfg["head_max_len"])
    base_cfg["gradient_checkpointing"] = bool(cfg["gradient_checkpointing"])
    base_cfg["max_tokens_per_batch"] = int(cfg["max_tokens_per_batch"])

    resume_dir = args.out / "checkpoint_latest"
    start_epoch = 0
    if (resume_dir / "model.safetensors").exists():
        meta = json.loads((resume_dir / "checkpoint_meta.json").read_text()) if (resume_dir / "checkpoint_meta.json").exists() else {}
        start_epoch = int(meta.get("epoch", 0))
        log(f"resuming from {resume_dir} (completed epoch {start_epoch})")
        if start_epoch >= int(cfg["epochs"]):
            raise SystemExit(f"{resume_dir} already finished {start_epoch} epochs; delete it to train again.")

    model = build_model(base_cfg, encoder_dir=str(model_dir) + "/encoder" if not (resume_dir / "encoder").exists()
                        else str(resume_dir / "encoder"))
    weights_path = resume_dir / "model.safetensors" if (resume_dir / "model.safetensors").exists() \
        else Path(model_dir) / "model.safetensors"
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    template_keys = list(model.state_dict().keys())
    model = maybe_lora(model, cfg)
    enable_checkpointing(model, cfg)
    model.to(device).train()

    if launched:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp = DDP(model, device_ids=[local], find_unused_parameters=True)
    else:
        ddp = model

    items = torch.load(args.items, weights_only=False)
    if cfg.get("max_items"):
        items = items[: int(cfg["max_items"])]
    train_items, calib_items = hold_out_calibration(
        items, float(cfg["calib_fraction"]), int(cfg["calib_max_per_family"]), int(cfg["seed"]))
    train_items = apply_family_weights(train_items, cfg.get("family_weights") or {}, int(cfg["seed"]))
    my_items = train_items[rank::world]
    log(f"items: {len(items)} total | {len(train_items)} train after weights | {len(calib_items)} calib | "
        f"{len(my_items)} on rank {rank}/{world}")

    epochs = int(cfg["epochs"])
    micro, accum = int(cfg["micro_batch"]), int(cfg["grad_accum"])
    group, w_sph, w_rps = int(cfg["group_size"]), float(cfg["w_sph"]), float(cfg["w_rps"])
    ce_w = float(cfg["ce_weight"])
    sigma0, sigma1 = float(cfg["sigma_start"]), float(cfg["sigma_end"])
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]
    use_scaler = amp == torch.float16

    raw = model
    enc_params = [p for n, p in raw.named_parameters() if "encoder." in n and p.requires_grad]
    head_params = [p for n, p in raw.named_parameters() if "encoder." not in n and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": float(cfg["lr_encoder"])},
        {"params": head_params, "lr": float(cfg["lr_head"])},
    ], weight_decay=float(cfg["weight_decay"]))
    steps_per_epoch = max(1, math.ceil(len(my_items) / micro / accum))
    total_steps = steps_per_epoch * max(1, epochs - start_epoch)
    warmup = max(1, int(total_steps * float(cfg["warmup_ratio"])))
    eta = float(cfg["eta_min"])

    min_ratio = eta / float(cfg["lr_encoder"])

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    wandb_run = None
    if cfg.get("wandb") and is_main():
        import wandb
        wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT", "coding-decision-agent"),
                               config={k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))})

    t0 = time.time()
    last_avg = 0.0
    for epoch in range(start_epoch, epochs):
        rng = random.Random(int(cfg["seed"]) + epoch + rank)
        rng.shuffle(my_items)
        sigma = sigma0 + (sigma1 - sigma0) * (epoch / max(1, epochs - 1))
        opt.zero_grad(set_to_none=True)
        running = 0.0
        n_batches = 0
        fam_loss = defaultdict(float)
        fam_n = defaultdict(int)
        for b in range(0, len(my_items), micro):
            chunk = my_items[b:b + micro]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=amp, enabled=amp != torch.float32):
                logits, act = ddp(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            k = mask.sum(-1, keepdim=True).float().clamp(min=1)
            eps = torch.randn((group,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=w_sph, w_rps=w_rps)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            per_item = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
            loss_ce = per_item.mean()
            loss = (loss_rl + ce_w * loss_ce) / accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.item()) * accum
            n_batches += 1
            for fam, li in zip(batch["family"], per_item.detach().cpu().tolist()):
                fam_loss[fam] += li
                fam_n[fam] += 1
            if n_batches % accum == 0 or b + micro >= len(my_items):
                if use_scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), float(cfg["grad_clip"]))
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if is_main() and n_batches % int(cfg["log_every"]) == 0:
                msg = (f"  epoch {epoch + 1}/{epochs} step {n_batches} loss {loss.item() * accum:.4f} "
                       f"reward {r.mean().item():.3f} lr {sched.get_last_lr()[0]:.2e}")
                log(msg)
                if wandb_run:
                    wandb_run.log({"loss": loss.item() * accum, "reward": r.mean().item(), "epoch": epoch + 1})
        avg = running / max(1, n_batches)
        last_avg = avg
        fam_msg = " ".join(f"{f} {fam_loss[f] / max(1, fam_n[f]):.3f}" for f in sorted(fam_loss))
        log(f"=== epoch {epoch + 1}/{epochs} avg loss {avg:.4f} | {fam_msg} | {time.time() - t0:.0f}s ===")
        barrier()
        if is_main():
            save_checkpoint(raw, tok, base_cfg, cfg, args.out / "checkpoint_latest", template_keys, epoch + 1, avg)
        barrier()

    temps = None
    if is_main():
        log("fitting calibration temperatures on the held-out slice")
        torch.cuda.empty_cache()
        raw.eval()
        from train.calibrate import _collect
        grouped = _collect(raw, calib_items, tok, device, amp)
        temps = fit_temperatures({k: v for k, v in grouped.items() if v})
        log(f"temperatures (choice, score, noul): {[round(t, 4) for t in temps]}")
        save_checkpoint(raw, tok, base_cfg, cfg, args.out, template_keys, epochs, last_avg, temps)
    barrier()
    d = _dist()
    if d is not None:
        d.destroy_process_group()
    log(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    train()
