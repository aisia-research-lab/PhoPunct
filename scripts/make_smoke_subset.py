# -*- coding: utf-8 -*-
"""Cut a small subset of a domain (News/ or Novels/) for quick CPU smoke tests.

Each split is truncated to at most N lines, backing off to the nearest
sentence-final label (PERIOD / QMARK / EXCLAM) so no sentence is cut in half.

    python scripts/make_smoke_subset.py --src_dir data/punctuation/News --dst_dir smoke_data/News
"""
import argparse
import os

EOS_LABELS = {"PERIOD", "QMARK", "EXCLAM"}


def cut_at_boundary(lines, target_n_lines):
    if target_n_lines >= len(lines):
        return lines
    cut = target_n_lines
    while cut > 0:
        parts = lines[cut - 1].strip().split()
        if len(parts) == 2 and parts[1] in EOS_LABELS:
            break
        cut -= 1
    if cut == 0:
        cut = target_n_lines  # no sentence boundary found: hard cut (rare)
    return lines[:cut]


def process_file(src_path, dst_path, n_lines):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    subset = cut_at_boundary(lines, n_lines)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(subset)
    print(f"{src_path} -> {dst_path}: {len(subset)}/{len(lines)} lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--n_train_lines", type=int, default=4000)
    ap.add_argument("--n_valid_lines", type=int, default=800)
    ap.add_argument("--n_test_lines", type=int, default=800)
    args = ap.parse_args()

    for split, n in (("train", args.n_train_lines), ("valid", args.n_valid_lines), ("test", args.n_test_lines)):
        process_file(os.path.join(args.src_dir, f"{split}.txt"), os.path.join(args.dst_dir, f"{split}.txt"), n)


if __name__ == "__main__":
    main()
