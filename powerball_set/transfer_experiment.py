#!/usr/bin/env python3
"""
transfer_experiment.py

Goal: measure whether a model trained on one "era" (pre-change vs post-change) transfers to the other.

We intentionally evaluate ONLY the main numbers task (5-of-69) because PB mechanics changed historically.

This script:
- Reads powerball.csv
- Splits rows by a cutover date into PRE (<cutover) and POST (>=cutover)
- Trains a small Transformer (main-only: no PB input, PB loss disabled) on one split
- Evaluates main-hit-rate on the other split

Output is compared against chance-ish baseline.
"""

import argparse
import csv
import random
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from powerball_set.train_powerball_set import (
    MAIN_K,
    MAIN_MAX,
    DrawTransformer,
    NextDrawDataset,
    _parse_date,
    compute_loss_and_metrics,
    ddp_is_enabled,
    evaluate,
    seed_all,
)


Draw = Tuple[List[int], int]  # (main0, pb0) but pb0 is dummy here


def load_powerball_csv_with_dates(path: str) -> List[Tuple[datetime, Draw]]:
    """
    Returns list of (dt, (main0, pb0_raw_minus1)).
    Does not validate PB range since this script ignores PB.
    """
    rows: List[Tuple[datetime, Draw]] = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)

        required_cols = ["date", "n1", "n2", "n3", "n4", "n5", "pb"]
        for col in required_cols:
            if col not in (reader.fieldnames or []):
                raise ValueError(f"CSV missing required column '{col}'. Found columns: {reader.fieldnames}")

        for row in reader:
            dt = _parse_date(row.get("date", ""))
            if dt is None:
                raise ValueError(f"Could not parse date: {row.get('date')!r}")
            main = [int(row[f"n{i}"]) for i in range(1, 6)]
            pb = int(row["pb"])

            main0 = [x - 1 for x in main]
            pb0 = pb - 1

            if len(set(main0)) != MAIN_K:
                raise ValueError(f"Duplicate main numbers found: {main}")
            for x in main0:
                if not (0 <= x < MAIN_MAX):
                    raise ValueError(f"Main number out of range after conversion: {x} (raw={x+1})")

            rows.append((dt, (main0, pb0)))

    rows.sort(key=lambda x: x[0])  # chronological
    return rows


def split_by_cutover(rows: List[Tuple[datetime, Draw]], cutover: datetime) -> Tuple[List[Draw], List[Draw]]:
    pre: List[Draw] = []
    post: List[Draw] = []
    for dt, draw in rows:
        (post if dt >= cutover else pre).append(draw)
    return pre, post


def _make_loaders(draws: List[Draw], context_len: int, batch_size: int, num_workers: int):
    # Main-only: PB is NOT encoded in features (encode_pb=False)
    ds = NextDrawDataset(draws, context_len, encode_pb=False)
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        ds,
        batch_size=min(batch_size, 64),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, eval_loader


def train_main_only(
    *,
    draws: List[Draw],
    device: torch.device,
    context_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    amp: bool,
    num_workers: int,
    log_every: int,
) -> DrawTransformer:
    if ddp_is_enabled():
        raise RuntimeError("transfer_experiment.py is intended for single-process runs (no DDP).")

    if len(draws) <= context_len:
        raise ValueError(f"Not enough draws for context_len={context_len}: got {len(draws)}")

    model = DrawTransformer(
        context_len=context_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    train_loader, _ = _make_loaders(draws, context_len, batch_size, num_workers)

    global_step = 0
    model.train()
    for epoch in range(epochs):
        for x, y_main, y_pb in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y_main = y_main.to(device, non_blocking=True)
            y_pb = y_pb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                lm, lp = model(x)
                loss, loss_main, loss_pb, hit_rate, exact5, pb_acc = compute_loss_and_metrics(
                    lm, lp, y_main, y_pb, pb_loss_weight=0.0
                )

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if global_step % log_every == 0:
                print(
                    f"[epoch {epoch+1}/{epochs} step {global_step}] "
                    f"train_loss={loss.item():.4f} main_hit_rate={hit_rate.item():.4f}",
                    flush=True,
                )

    return model


@torch.no_grad()
def evaluate_main_only(model: DrawTransformer, *, draws: List[Draw], device: torch.device, context_len: int, batch_size: int, num_workers: int):
    _, eval_loader = _make_loaders(draws, context_len, batch_size, num_workers)
    m = evaluate(model, eval_loader, device, pb_loss_weight=0.0, max_batches=10_000)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--cutover_date", type=str, default="10/07/15", help="Date separating PRE and POST (default: 10/07/15)")
    p.add_argument("--context_len", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30, help="Epochs per direction (default: 30)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    cutover = _parse_date(args.cutover_date)
    if cutover is None:
        raise ValueError(f"Could not parse --cutover_date={args.cutover_date!r}")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_powerball_csv_with_dates(args.data_path)
    pre, post = split_by_cutover(rows, cutover)
    print(f"Loaded {len(rows)} rows total. PRE={len(pre)} POST={len(post)} cutover={cutover.date().isoformat()}", flush=True)

    expected_hits = float(MAIN_K) * (float(MAIN_K) / float(MAIN_MAX))
    expected_hit_rate = expected_hits / float(MAIN_K)
    print(f"Chance-ish baseline: main_hit_rate≈{expected_hit_rate:.4f}", flush=True)
    print("", flush=True)

    if len(pre) <= args.context_len or len(post) <= args.context_len:
        raise ValueError(
            f"Not enough rows in one split for context_len={args.context_len}. "
            f"Need > {args.context_len} in both PRE and POST."
        )

    # 1) Train on PRE, eval on POST
    print("=== Train on PRE (main-only), evaluate on POST ===", flush=True)
    model_pre = train_main_only(
        draws=pre,
        device=device,
        context_len=args.context_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        log_every=args.log_every,
    )
    m_pre_on_post = evaluate_main_only(
        model_pre,
        draws=post,
        device=device,
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Result PRE→POST: loss={m_pre_on_post['loss'].item():.4f} "
        f"main_hit_rate={m_pre_on_post['hit_rate'].item():.4f} "
        f"main_exact5={m_pre_on_post['exact5'].item():.6f}",
        flush=True,
    )
    print("", flush=True)

    # 2) Train on POST, eval on PRE
    print("=== Train on POST (main-only), evaluate on PRE ===", flush=True)
    model_post = train_main_only(
        draws=post,
        device=device,
        context_len=args.context_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        log_every=args.log_every,
    )
    m_post_on_pre = evaluate_main_only(
        model_post,
        draws=pre,
        device=device,
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Result POST→PRE: loss={m_post_on_pre['loss'].item():.4f} "
        f"main_hit_rate={m_post_on_pre['hit_rate'].item():.4f} "
        f"main_exact5={m_post_on_pre['exact5'].item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()


