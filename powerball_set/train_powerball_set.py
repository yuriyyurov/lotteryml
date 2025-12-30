#!/usr/bin/env python3
"""
train_powerball_set.py

Powerball mechanics (US Powerball):
- 5 main numbers are an UNORDERED SET drawn from 1..69
- 1 Powerball (bonus) number drawn from 1..26 (independent pool)

We model this as:
- Main: multi-label prediction over 69 numbers (BCE with logits)
- Bonus: single-label prediction over 26 numbers (cross-entropy)

This is a "next draw" predictor over a time-ordered sequence of past draws.

Data format (CSV header required):
  date,n1,n2,n3,n4,n5,pb

Example row:
  12/27/25,5,20,34,39,62,1

Notes:
- Rows may be in any order; we sort by date ascending if 'date' parses.
- If the underlying lottery process is random, out-of-sample performance should match chance.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ---------------------------
# Constants (Powerball)
# ---------------------------

MAIN_MAX = 69
PB_MAX = 26
MAIN_K = 5
IN_DIM = MAIN_MAX + PB_MAX  # 95


# ---------------------------
# DDP utilities
# ---------------------------

def ddp_is_enabled() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def ddp_rank() -> int:
    return torch.distributed.get_rank() if ddp_is_enabled() else 0


def ddp_world_size() -> int:
    return torch.distributed.get_world_size() if ddp_is_enabled() else 1


def ddp_barrier():
    if ddp_is_enabled():
        torch.distributed.barrier()


@torch.no_grad()
def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    y = x.clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    y /= ddp_world_size()
    return y


def log_rank0(msg: str):
    if ddp_rank() == 0:
        print(msg, flush=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Data loading
# ---------------------------

def _parse_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%y",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load_powerball_csv(
    path: str,
    min_date: Optional[Union[str, datetime]] = None,
    *,
    allow_legacy_pb: bool = False,
    encode_pb: bool = True,
) -> List[Tuple[List[int], int]]:
    """
    Loads Powerball CSV/TSV with auto-detected delimiter.

    Required columns:
      date, n1, n2, n3, n4, n5, pb

    Numbers in the file are:
      - main: 1..69
      - pb: 1..26
    Converted internally to:
      - main: 0..68
      - pb: 0..25

    Args:
      min_date:
        Optional cutoff. If provided, only rows with date >= min_date are kept.
        Accepts either a datetime, or a string parseable by _parse_date (e.g. '10/07/15').
      allow_legacy_pb:
        If True, allows historical rows where pb > 26 (legacy Powerball eras).
        NOTE: If encode_pb=True and pb>26 is encountered, it will still raise.
      encode_pb:
        If False, the PB part of the input features is always all-zeros, and pb targets are set to 0 (dummy).
        This is useful for pretraining on all history using only the main numbers task.

    Returns:
      List of tuples: [([n1..n5], pb), ...] in chronological order (oldest->newest) when possible.
    """
    import csv

    min_dt: Optional[datetime] = None
    if isinstance(min_date, str):
        min_dt = _parse_date(min_date)
        if min_dt is None:
            raise ValueError(f"Could not parse --min_date='{min_date}'. Try e.g. 10/07/15 or 2015-10-07.")
    elif isinstance(min_date, datetime):
        min_dt = min_date
    elif min_date is None:
        min_dt = None
    else:
        raise TypeError(f"min_date must be str|datetime|None, got {type(min_date)}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)

        required_cols = ["date", "n1", "n2", "n3", "n4", "n5", "pb"]
        for col in required_cols:
            if col not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV missing required column '{col}'. Found columns: {reader.fieldnames}"
                )

        for row in reader:
            dt = _parse_date(row.get("date", ""))
            main = [int(row[f"n{i}"]) for i in range(1, 6)]
            pb = int(row["pb"])
            rows.append((dt, main, pb))

    # If we need min_date filtering, we require parseable dates.
    if rows and min_dt is not None and any(dt is None for (dt, _, _) in rows):
        raise ValueError("Some rows have unparseable 'date', but --min_date was provided.")

    # Sort by date ascending if all rows have parseable dates; otherwise keep file order.
    if rows and all(dt is not None for (dt, _, _) in rows):
        rows.sort(key=lambda x: x[0])

    if min_dt is not None:
        before = len(rows)
        rows = [(dt, main, pb) for (dt, main, pb) in rows if dt is not None and dt >= min_dt]
        after = len(rows)
        if after == 0:
            raise ValueError(
                f"--min_date kept 0 rows (min_date={min_dt.date().isoformat()}). "
                f"Input had {before} rows."
            )

    draws: List[Tuple[List[int], int]] = []
    legacy_pb_count = 0
    for _, main, pb in rows:
        main0 = [x - 1 for x in main]
        pb0 = pb - 1

        # Validation
        if len(set(main0)) != MAIN_K:
            raise ValueError(f"Duplicate main numbers found: {main}")
        for x in main0:
            if not (0 <= x < MAIN_MAX):
                raise ValueError(f"Main number out of range after conversion: {x} (raw={x+1})")
        if not (0 <= pb0 < PB_MAX):
            legacy_pb_count += 1
            if encode_pb:
                raise ValueError(
                    f"Powerball out of range for PB_MAX={PB_MAX}: got pb={pb}. "
                    f"If your CSV includes legacy draws where PB>26, either re-run with "
                    f"--min_date 10/07/15 (or later), or use --no_pb_input and --pb_loss_weight 0 "
                    f"to pretrain on main numbers only."
                )
            if not allow_legacy_pb:
                raise ValueError(
                    f"Legacy Powerball detected (pb={pb} > {PB_MAX}) but --allow_legacy_pb is not set. "
                    f"Either filter with --min_date 10/07/15, or enable --allow_legacy_pb for main-only pretraining."
                )
            # Dummy value (ignored if pb_loss_weight==0; also PB features are zeroed if encode_pb=False).
            pb0 = 0

        draws.append((main0, pb0))

    if len(draws) < 200:
        print(f"Warning: only {len(draws)} draws loaded — model may underfit.")
    if legacy_pb_count > 0 and ddp_rank() == 0:
        print(f"Info: encountered {legacy_pb_count} legacy rows with pb > {PB_MAX} (handled by settings).", flush=True)

    return draws


def draw_to_feature(main0: List[int], pb0: int, *, encode_pb: bool = True) -> torch.Tensor:
    """
    Feature vector for a single draw:
      - 69-dim multi-hot for main set
      - 26-dim one-hot for powerball
    Total: 95 dims.
    """
    x = torch.zeros(IN_DIM, dtype=torch.float32)
    for m in main0:
        x[m] = 1.0
    if encode_pb:
        x[MAIN_MAX + pb0] = 1.0
    return x


class NextDrawDataset(Dataset):
    """
    Builds (context -> next draw) examples.

    Input:
      - context_len draws as features [T, IN_DIM]
    Target:
      - main multi-hot [69]
      - pb index scalar (0..25)
    """

    def __init__(self, draws: List[Tuple[List[int], int]], context_len: int, *, encode_pb: bool = True):
        if len(draws) <= context_len:
            raise ValueError(
                f"Not enough draws for context_len={context_len}. "
                f"Need > {context_len} rows, got {len(draws)}."
            )
        self.draws = draws
        self.context_len = context_len
        self.encode_pb = encode_pb
        self.features = torch.stack([draw_to_feature(m, b, encode_pb=encode_pb) for (m, b) in draws], dim=0)  # [N,IN_DIM]

    def __len__(self) -> int:
        return len(self.draws) - self.context_len

    def __getitem__(self, idx: int):
        T = self.context_len
        context = self.features[idx : idx + T]  # [T,IN_DIM]
        main, pb = self.draws[idx + T]

        target_main = torch.zeros(MAIN_MAX, dtype=torch.float32)
        target_main[main] = 1.0
        target_pb = torch.tensor(pb, dtype=torch.long)
        return context, target_main, target_pb


# ---------------------------
# Model: Transformer over draws (not over numbers)
# ---------------------------


class DrawTransformer(nn.Module):
    """
    Transformer encodes a sequence of past draws (each draw is a IN_DIM vector).
    Outputs:
      - logits_main: [B, 69] multi-label logits (we choose top-5 at evaluation)
      - logits_pb: [B, 26] single-label logits
    """

    def __init__(
        self,
        context_len: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_len = context_len
        self.in_proj = nn.Linear(IN_DIM, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        self.head_main = nn.Linear(d_model, MAIN_MAX)
        self.head_pb = nn.Linear(d_model, PB_MAX)

        # Causal mask so the model can't "peek" ahead inside the context window
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B,T,IN_DIM]
        Use final timestep (most recent draw in context) to predict the next draw.
        """
        B, T, D = x.shape
        assert T == self.context_len and D == IN_DIM

        h = self.in_proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1,T]
        h = h + self.pos_emb(pos)

        h = self.enc(h, mask=self.causal_mask)
        h = self.ln(h)

        last = h[:, -1, :]  # [B,d_model]
        logits_main = self.head_main(last)  # [B,69]
        logits_pb = self.head_pb(last)  # [B,26]
        return logits_main, logits_pb


# ---------------------------
# Metrics
# ---------------------------


@torch.no_grad()
def topk_hits(pred_logits: torch.Tensor, target_multi_hot: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns hits count per example: number of correct items in top-k.
    pred_logits: [B,V]
    target_multi_hot: [B,V] {0,1}
    """
    topk = torch.topk(pred_logits, k=k, dim=-1).indices  # [B,k]
    hits = target_multi_hot.gather(1, topk).sum(dim=1)  # [B]
    return hits


def compute_loss_and_metrics(logits_main, logits_pb, target_main, target_pb, *, pb_loss_weight: float = 1.0):
    """
    logits_main: [B,69]
    logits_pb: [B,26]
    target_main: [B,69] float {0,1}
    target_pb: [B] long
    """
    loss_main = F.binary_cross_entropy_with_logits(logits_main, target_main)
    if pb_loss_weight > 0:
        loss_pb = F.cross_entropy(logits_pb, target_pb)
        loss = loss_main + (pb_loss_weight * loss_pb)
    else:
        loss_pb = torch.zeros((), device=logits_main.device, dtype=loss_main.dtype)
        loss = loss_main

    hits5 = topk_hits(logits_main, target_main, k=MAIN_K)  # [B]
    hit_rate = (hits5 / float(MAIN_K)).mean()
    exact5 = (hits5 == MAIN_K).float().mean()

    if pb_loss_weight > 0:
        pred_pb = logits_pb.argmax(dim=-1)
        pb_acc = (pred_pb == target_pb).float().mean()
    else:
        pb_acc = torch.tensor(float("nan"), device=logits_main.device)

    return loss, loss_main.detach(), loss_pb.detach(), hit_rate.detach(), exact5.detach(), pb_acc.detach()


@torch.no_grad()
def evaluate(model, loader, device, *, pb_loss_weight: float, max_batches: int = 200):
    model.eval()
    agg = {
        "loss": [],
        "loss_main": [],
        "loss_pb": [],
        "hit_rate": [],
        "exact5": [],
        "pb_acc": [],
    }
    for i, (x, y_main, y_pb) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y_main = y_main.to(device, non_blocking=True)
        y_pb = y_pb.to(device, non_blocking=True)

        lm, lp = model(x)
        loss, lm_loss, lp_loss, hit_rate, exact5, pb_acc = compute_loss_and_metrics(
            lm, lp, y_main, y_pb, pb_loss_weight=pb_loss_weight
        )
        agg["loss"].append(loss)
        agg["loss_main"].append(lm_loss)
        agg["loss_pb"].append(lp_loss)
        agg["hit_rate"].append(hit_rate)
        agg["exact5"].append(exact5)
        agg["pb_acc"].append(pb_acc)

    out = {}
    for k, vals in agg.items():
        if not vals:
            out[k] = torch.tensor(float("nan"), device=device)
        else:
            out[k] = torch.stack(vals).mean()
            out[k] = all_reduce_mean(out[k])
    return out


# ---------------------------
# Training
# ---------------------------


@dataclass
class Config:
    context_len: int = 16
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    amp: bool = True
    log_every: int = 50
    eval_every: int = 250
    num_workers: int = 2
    pb_loss_weight: float = 1.0
    no_pb_input: bool = False
    allow_legacy_pb: bool = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument(
        "--min_date",
        type=str,
        default=None,
        help="Optional cutoff date; only rows with date >= min_date are used. Example: 10/07/15",
    )
    p.add_argument(
        "--allow_legacy_pb",
        action="store_true",
        help="Allow historical rows where pb > 26. Only safe when you are NOT using PB as input/target "
             "(use with --no_pb_input and --pb_loss_weight 0).",
    )
    p.add_argument(
        "--no_pb_input",
        action="store_true",
        help="Do not encode PB as input feature (PB part of the feature vector will be all-zeros). "
             "Useful for main-only pretraining across legacy eras.",
    )
    p.add_argument(
        "--pb_loss_weight",
        type=float,
        default=1.0,
        help="Weight for PB cross-entropy loss. Set to 0 for main-only pretraining.",
    )
    p.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Optional checkpoint to initialize weights from (for transfer learning / finetuning).",
    )
    p.add_argument("--context_len", type=int, default=Config.context_len)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--grad_clip", type=float, default=Config.grad_clip)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--nhead", type=int, default=Config.nhead)
    p.add_argument("--num_layers", type=int, default=Config.num_layers)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--log_every", type=int, default=Config.log_every)
    p.add_argument("--eval_every", type=int, default=Config.eval_every)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--ckpt", type=str, default="checkpoints/powerball_set.pt")
    args = p.parse_args()

    # DDP init (torchrun sets env vars)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend=("nccl" if torch.cuda.is_available() else "gloo"))

    seed_all(args.seed + ddp_rank())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    cfg = Config(
        context_len=args.context_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        amp=not args.no_amp,
        log_every=args.log_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        pb_loss_weight=float(args.pb_loss_weight),
        no_pb_input=bool(args.no_pb_input),
        allow_legacy_pb=bool(args.allow_legacy_pb),
    )

    encode_pb = not args.no_pb_input
    draws = load_powerball_csv(
        args.data_path,
        min_date=args.min_date,
        allow_legacy_pb=args.allow_legacy_pb,
        encode_pb=encode_pb,
    )
    if args.min_date:
        log_rank0(f"Loaded {len(draws)} draws from {args.data_path} (filtered with min_date={args.min_date})")
    else:
        log_rank0(f"Loaded {len(draws)} draws from {args.data_path}")

    if len(draws) <= cfg.context_len:
        raise ValueError(
            f"powerball.csv слишком маленький для context_len={cfg.context_len}. "
            f"Нужно минимум {cfg.context_len + 1} строк, есть {len(draws)}."
        )

    split = int(len(draws) * args.train_split)
    train_draws = draws[:split]
    val_draws = draws[max(0, split - cfg.context_len) :]  # overlap

    train_ds = NextDrawDataset(train_draws, cfg.context_len, encode_pb=encode_pb)
    val_ds = NextDrawDataset(val_draws, cfg.context_len, encode_pb=encode_pb)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp_is_enabled() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp_is_enabled() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(cfg.batch_size, 32),
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = DrawTransformer(
        context_len=cfg.context_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    # Optional initialization (transfer learning)
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu")
        sd = init["model"] if isinstance(init, dict) and "model" in init else init
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log_rank0(
            f"Initialized from {args.init_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})"
        )

    if ddp_is_enabled():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # Chance baselines (sanity checking):
    # - Main set: expected overlap when you pick 5 uniformly from 69 is 5*(5/69)
    expected_hits = float(MAIN_K) * (float(MAIN_K) / float(MAIN_MAX))
    expected_hit_rate = expected_hits / float(MAIN_K)
    # - Powerball: 1/26 chance
    expected_pb_acc = 1.0 / float(PB_MAX)
    if cfg.pb_loss_weight > 0:
        log_rank0(
            f"Chance-ish baselines: main_hit_rate≈{expected_hit_rate:.4f}, pb_top1_acc≈{expected_pb_acc:.4f}"
        )
    else:
        log_rank0(f"Chance-ish baseline: main_hit_rate≈{expected_hit_rate:.4f} (PB disabled)")

    global_step = 0
    t0 = time.time()

    def save_ckpt(step: int):
        if ddp_rank() != 0:
            return
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        core = model.module if hasattr(model, "module") else model
        torch.save({"model": core.state_dict(), "step": step, "cfg": cfg.__dict__}, args.ckpt)
        log_rank0(f"Saved checkpoint: {args.ckpt}")

    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for x, y_main, y_pb in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y_main = y_main.to(device, non_blocking=True)
            y_pb = y_pb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                lm, lp = model(x)
                loss, loss_main, loss_pb, hit_rate, exact5, pb_acc = compute_loss_and_metrics(
                    lm, lp, y_main, y_pb, pb_loss_weight=cfg.pb_loss_weight
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                loss_r = all_reduce_mean(loss.detach())
                hit_r = all_reduce_mean(hit_rate)
                pbacc_r = all_reduce_mean(pb_acc)

                elapsed = time.time() - t0
                examples = cfg.batch_size * cfg.log_every * ddp_world_size()
                ex_per_s = examples / max(elapsed, 1e-6)
                t0 = time.time()

                log_rank0(
                    f"[epoch {epoch+1}/{cfg.epochs} step {global_step}] "
                    f"train_loss={loss_r.item():.4f} "
                    f"main_hit_rate={hit_r.item():.4f} "
                    f"pb_acc={pbacc_r.item():.4f} "
                    f"ex/s={ex_per_s:,.0f}"
                )

            if global_step % cfg.eval_every == 0:
                ddp_barrier()
                m = evaluate(model, val_loader, device, pb_loss_weight=cfg.pb_loss_weight)
                log_rank0(
                    f"  eval: loss={m['loss'].item():.4f} "
                    f"main_hit_rate={m['hit_rate'].item():.4f} "
                    f"main_exact5={m['exact5'].item():.6f} "
                    f"pb_acc={m['pb_acc'].item():.4f}"
                )
                save_ckpt(global_step)
                ddp_barrier()

    ddp_barrier()
    m = evaluate(model, val_loader, device, pb_loss_weight=cfg.pb_loss_weight, max_batches=500)
    log_rank0(
        f"FINAL eval: loss={m['loss'].item():.4f} "
        f"main_hit_rate={m['hit_rate'].item():.4f} "
        f"main_exact5={m['exact5'].item():.6f} "
        f"pb_acc={m['pb_acc'].item():.4f}"
    )
    save_ckpt(global_step)

    if ddp_is_enabled():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()


