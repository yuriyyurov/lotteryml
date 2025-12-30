#!/usr/bin/env python3
"""
predict_powerball_set.py

Loads a trained checkpoint from train_powerball_set.py and prints a prediction:
- Top-5 main numbers (1..69)
- Top-1 Powerball (1..26)

Two modes:
- default: predict NEXT draw using the last `context_len` draws in the CSV
- --eval_last_known: use the context before the last known draw and compare prediction to that last draw
"""

import argparse
from typing import List, Tuple

import torch

# Support both:
# - module execution: python -m powerball_set.predict_powerball_set
# - direct file execution: python powerball_set/predict_powerball_set.py
try:
    from .train_powerball_set import (
        DrawTransformer,
        MAIN_K,
        MAIN_MAX,
        PB_MAX,
        IN_DIM,
        draw_to_feature,
        load_powerball_csv,
    )
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from powerball_set.train_powerball_set import (  # type: ignore
        DrawTransformer,
        MAIN_K,
        MAIN_MAX,
        PB_MAX,
        IN_DIM,
        draw_to_feature,
        load_powerball_csv,
    )


@torch.no_grad()
def _predict_from_context(
    model: DrawTransformer,
    context_draws: List[Tuple[List[int], int]],
    device: torch.device,
    topk_main: int = MAIN_K,
    topk_pb: int = 3,
    *,
    encode_pb: bool,
):
    # context_draws: length T, each is (main0, pb0)
    x = torch.stack([draw_to_feature(m, b, encode_pb=encode_pb) for (m, b) in context_draws], dim=0)  # [T,IN_DIM]
    x = x.unsqueeze(0).to(device)  # [1,T,IN_DIM]
    lm, lp = model(x)
    lm = lm.squeeze(0)
    lp = lp.squeeze(0)

    main_idx = torch.topk(lm, k=topk_main, dim=-1).indices.tolist()
    pb_idx = torch.topk(lp, k=min(topk_pb, PB_MAX), dim=-1).indices.tolist()
    return main_idx, pb_idx


def _fmt_main(nums0: List[int]) -> str:
    return " ".join(str(n + 1) for n in sorted(nums0))


def _fmt_pb(pb0: int) -> str:
    return str(pb0 + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="checkpoints/powerball_set.pt")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 ... (default: auto)")
    p.add_argument("--topk_pb", type=int, default=3, help="How many PB candidates to print (default: 3)")
    p.add_argument("--eval_last_known", action="store_true", help="Evaluate against the last known draw in CSV")
    p.add_argument(
        "--min_date",
        type=str,
        default=None,
        help="Optional cutoff date; only rows with date >= min_date are used. Example: 10/07/15",
    )
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    context_len = int(cfg.get("context_len", 16))
    d_model = int(cfg.get("d_model", 512))
    nhead = int(cfg.get("nhead", 8))
    num_layers = int(cfg.get("num_layers", 6))
    dropout = float(cfg.get("dropout", 0.1))
    no_pb_input = bool(cfg.get("no_pb_input", False))
    encode_pb = not no_pb_input

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DrawTransformer(
        context_len=context_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    draws = load_powerball_csv(
        args.data_path,
        min_date=args.min_date,
        allow_legacy_pb=bool(cfg.get("allow_legacy_pb", False)),
        encode_pb=encode_pb,
    )
    if len(draws) <= context_len:
        raise ValueError(
            f"Not enough rows for context_len={context_len}: need > {context_len}, got {len(draws)}"
        )

    if args.eval_last_known:
        # context: draws[-context_len-1 : -1], target: draws[-1]
        context = draws[-(context_len + 1) : -1]
        target_main0, target_pb0 = draws[-1]
        mode_desc = "EVAL against last known draw"
    else:
        # context: draws[-context_len:], target unknown
        context = draws[-context_len:]
        target_main0, target_pb0 = None, None
        mode_desc = "PREDICT next draw after last known"

    main_idx, pb_idx = _predict_from_context(
        model,
        context,
        device,
        topk_main=MAIN_K,
        topk_pb=args.topk_pb,
        encode_pb=encode_pb,
    )

    pred_main0 = main_idx
    pred_pb0 = pb_idx[0]

    print(f"Mode: {mode_desc}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Context_len: {context_len} | Input dim: {IN_DIM} | Main V: {MAIN_MAX} | PB V: {PB_MAX}")
    print("")
    print(f"Predicted MAIN (top-{MAIN_K}): {_fmt_main(pred_main0)}")
    print(f"Predicted PB (top-1): {_fmt_pb(pred_pb0)}")
    print(f"PB candidates (top-{min(args.topk_pb, PB_MAX)}): " + " ".join(str(i + 1) for i in pb_idx))

    if target_main0 is not None and target_pb0 is not None:
        print("")
        print(f"Target MAIN: {_fmt_main(target_main0)}")
        print(f"Target PB: {_fmt_pb(target_pb0)}")

        hits = len(set(pred_main0) & set(target_main0))
        pb_hit = int(pred_pb0 == target_pb0)
        print("")
        print(f"Hits MAIN: {hits}/{MAIN_K}")
        print(f"Hit PB: {pb_hit}/1")


if __name__ == "__main__":
    main()


