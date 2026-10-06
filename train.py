# -*- coding: utf-8 -*-
"""Train one of the 17 benchmark configurations on one domain.

Examples
--------
    # #1  mBERT (softmax)
    python train.py --backbone mbert --data_dir data/punctuation/News --output_dir outputs/mbert_news
    # #16 XLM-R + Bi-LSTM + CRF
    python train.py --backbone xlmr --use_bilstm --use_crf --data_dir data/punctuation/News --output_dir outputs/xlmr_bilstm_crf_news
    # #17 PhoPunct
    python train.py --backbone phobert --use_bilstm --use_crf --data_dir data/punctuation/News --output_dir outputs/phopunct_news

Model selection uses validation macro F1 (best epoch). After training, the best
checkpoint is scored on test.txt (disable with --no_test).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from phopunct.data import LABELS, load_dataset
from phopunct.metrics import evaluate
from phopunct.models import BACKBONES, WordTagger, canonical_backbone, display_name, load_tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_metrics(path: str, name: str, split: str, epoch: int, metrics: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "split": split, "epoch": epoch, "macro_f1": metrics["macro_f1"],
                   "per_class": metrics["report_dict"]}, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=list(BACKBONES) + ["bert"],
                   help="mbert | velectra | vibert | xlmr | phobert ('bert' = alias of vibert)")
    p.add_argument("--use_bilstm", action="store_true", help="add a Bi-LSTM on top of the backbone")
    p.add_argument("--use_crf", action="store_true", help="CRF output layer instead of softmax")
    p.add_argument("--model_name_or_path", default=None,
                   help="override the HF checkpoint of --backbone (e.g. vinai/phobert-base for CPU smoke tests)")
    p.add_argument("--data_dir", required=True, help="directory with train.txt / valid.txt / test.txt")
    p.add_argument("--output_dir", required=True)
    # Defaults below are the exact configuration used for every run in the paper.
    p.add_argument("--max_seq_length", type=int, default=256,
                   help="sub-word length; 256 is a hard ceiling for PhoBERT (258 positions)")
    p.add_argument("--lstm_hidden_size", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--num_train_epochs", type=float, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only); used in the paper")
    p.add_argument("--max_train_examples", type=int, default=None, help="limit #chunks (smoke tests)")
    p.add_argument("--max_eval_examples", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--no_test", action="store_true", help="skip test-set evaluation after training")
    return p.parse_args()


def main():
    args = parse_args()
    args.backbone = canonical_backbone(args.backbone)
    name = display_name(args.backbone, args.use_bilstm, args.use_crf)
    model_name = args.model_name_or_path or BACKBONES[args.backbone]

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    logger.info("=== %s | backbone=%s ===", name, model_name)
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    tokenizer = load_tokenizer(args.backbone, model_name)

    logger.info("Loading train data...")
    train_dataset = load_dataset(args.data_dir, "train", tokenizer, args.max_seq_length,
                                 max_examples=args.max_train_examples)
    logger.info("Loading valid data...")
    valid_dataset = load_dataset(args.data_dir, "valid", tokenizer, args.max_seq_length,
                                 max_examples=args.max_eval_examples)
    logger.info("Train: %d features | Valid: %d features", len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=args.eval_batch_size)

    model = WordTagger(args.backbone, model_name, num_labels=len(LABELS),
                       use_bilstm=args.use_bilstm, use_crf=args.use_crf,
                       lstm_hidden_size=args.lstm_hidden_size).to(device)
    model_config = {
        "backbone": args.backbone, "model_name_or_path": model_name,
        "use_bilstm": args.use_bilstm, "use_crf": args.use_crf,
        "lstm_hidden_size": args.lstm_hidden_size, "max_seq_length": args.max_seq_length,
    }

    total_steps = max(1, int(len(train_loader) * args.num_train_epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_epoch = 0
    global_step = 0
    n_epochs = max(1, round(args.num_train_epochs))

    for epoch in range(n_epochs):
        model.train()  # reset every epoch: evaluate() leaves the model in eval mode
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, word_starts, label_ids, word_mask = (t.to(device) for t in batch)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(input_ids, attention_mask, word_starts, label_ids=label_ids, word_mask=word_mask)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % args.logging_steps == 0:
                logger.info("[%s] epoch %d step %d/%d loss=%.4f",
                            name, epoch + 1, step + 1, len(train_loader), loss.item())

        logger.info("[%s] Epoch %d finished in %.1fs, avg_loss=%.4f", name, epoch + 1,
                    time.time() - t0, epoch_loss / max(1, len(train_loader)))

        metrics = evaluate(model, valid_loader, device)
        logger.info("[%s] Epoch %d - Valid macro F1 = %.4f\n%s",
                    name, epoch + 1, metrics["macro_f1"], metrics["report"])

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": max(best_f1, metrics["macro_f1"]),
            "config": model_config,
            "display_name": name,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last_checkpoint.pt"))

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(ckpt, os.path.join(args.output_dir, "best_checkpoint.pt"))
            with open(os.path.join(args.output_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
                f.write(f"{name}\nBest macro F1: {best_f1:.4f}\n\n")
                f.write(metrics["report"])
            write_metrics(os.path.join(args.output_dir, "valid_metrics.json"), name, "valid", best_epoch, metrics)
            logger.info("-> [%s] new best checkpoint, macro F1 = %.4f", name, best_f1)

    logger.info("[%s] Training finished. Best valid macro F1 = %.4f (epoch %d). Checkpoints in %s",
                name, best_f1, best_epoch, args.output_dir)

    if not args.no_test:
        logger.info("Loading test data...")
        test_dataset = load_dataset(args.data_dir, "test", tokenizer, args.max_seq_length,
                                    max_examples=args.max_eval_examples)
        test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=args.eval_batch_size)
        best = torch.load(os.path.join(args.output_dir, "best_checkpoint.pt"), map_location=device)
        model.load_state_dict(best["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device)
        write_metrics(os.path.join(args.output_dir, "test_metrics.json"), name, "test", best_epoch, test_metrics)
        logger.info("[%s] TEST macro F1 = %.4f (best-valid epoch %d)\n%s",
                    name, test_metrics["macro_f1"], best_epoch, test_metrics["report"])


if __name__ == "__main__":
    main()
