# -*- coding: utf-8 -*-
"""Restore punctuation in raw (unpunctuated, accented) Vietnamese text.

    python predict.py --checkpoint outputs/phopunct_news/best_checkpoint.pt \
                      --text "bạn có khỏe không tôi rất nhớ bạn"
    python predict.py --checkpoint ... --input_file in.txt > out.txt   # one passage per line

Words must be whitespace-separated exactly as in the training data. Inputs
longer than --max_seq_length sub-words are processed in consecutive windows.
"""
from __future__ import annotations

import argparse
import sys

import torch

from phopunct.data import ID2LABEL, encode_words
from phopunct.models import load_checkpoint

PUNCT_MAP = {"O": "", "PERIOD": ".", "COMMA": ",", "COLON": ":", "QMARK": "?", "EXCLAM": "!", "SEMICOLON": ";"}
SENTENCE_END = {"PERIOD", "QMARK", "EXCLAM"}


@torch.no_grad()
def predict_labels(model, tokenizer, words, max_seq_length, device):
    labels = ["O"] * len(words)
    start = 0
    while start < len(words):
        window = words[start:]
        input_ids, attention_mask, word_starts, kept, _ = encode_words(window, tokenizer, max_seq_length)
        if not kept:  # remaining words produce no sub-words
            break
        word_mask = [1] * len(kept) + [0] * (max_seq_length - len(kept))
        to_t = lambda x: torch.tensor([x], dtype=torch.long, device=device)  # noqa: E731
        pred = model(to_t(input_ids), to_t(attention_mask), to_t(word_starts),
                     label_ids=None, word_mask=to_t(word_mask))[0]
        for i, lab_id in zip(kept, pred):
            labels[start + i] = ID2LABEL[lab_id]
        start += kept[-1] + 1
    return labels


def render(words, labels, capitalize=True):
    out, cap_next = [], capitalize
    for w, lab in zip(words, labels):
        tok = w[0].upper() + w[1:] if (cap_next and w) else w
        out.append(tok + PUNCT_MAP[lab])
        cap_next = capitalize and lab in SENTENCE_END
    return " ".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--input_file", default=None)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--no_capitalize", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.text is None and args.input_file is None:
        p.error("provide --text or --input_file")
    device = torch.device(args.device)
    model, tokenizer, _, _ = load_checkpoint(args.checkpoint, device, args.model_name_or_path)

    if args.text is not None:
        lines = [args.text]
    else:
        with open(args.input_file, encoding="utf-8") as f:
            lines = f.read().splitlines()
    sys.stdout.reconfigure(encoding="utf-8")
    for line in lines:
        words = line.split()
        labels = predict_labels(model, tokenizer, words, args.max_seq_length, device)
        print(render(words, labels, capitalize=not args.no_capitalize))


if __name__ == "__main__":
    main()
