# -*- coding: utf-8 -*-
"""Score a trained checkpoint on a data split (default: test).

    python evaluate.py --checkpoint outputs/phopunct_news/best_checkpoint.pt \
                       --data_dir data/punctuation/News --split test

Works with checkpoints written by train.py and with those of the original
per-family training scripts (the configuration is inferred from the weights).
Writes <split>_metrics.json next to the checkpoint unless --output is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

from phopunct.data import load_dataset
from phopunct.metrics import evaluate
from phopunct.models import display_name, load_checkpoint

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--model_name_or_path", default=None, help="override backbone path (e.g. a local copy)")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_eval_examples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="path of the metrics JSON")
    args = p.parse_args()

    device = torch.device(args.device)
    model, tokenizer, cfg, ckpt = load_checkpoint(args.checkpoint, device, args.model_name_or_path)
    name = display_name(cfg["backbone"], cfg["use_bilstm"], cfg["use_crf"])
    logger.info("Loaded %s from %s (epoch %s)", name, args.checkpoint, ckpt.get("epoch"))

    dataset = load_dataset(args.data_dir, args.split, tokenizer, args.max_seq_length,
                           max_examples=args.max_eval_examples)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    metrics = evaluate(model, loader, device)
    logger.info("[%s] %s macro F1 = %.4f\n%s", name, args.split, metrics["macro_f1"], metrics["report"])

    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f"{args.split}_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": name, "split": args.split, "epoch": ckpt.get("epoch"),
                   "macro_f1": metrics["macro_f1"], "per_class": metrics["report_dict"]},
                  f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
