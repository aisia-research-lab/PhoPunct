# -*- coding: utf-8 -*-
"""Summarise the 17 x 2 runs as a Markdown table (macro F1 over the 6 punctuation classes).

    python scripts/collect_results.py --logs logs              # from training logs (valid only)
    python scripts/collect_results.py --outputs outputs        # from train.py outputs (valid + test)
"""
import argparse
import json
import os
import re

RUNS = [
    (1, "mbert", "mBERT"), (2, "velectra", "vELECTRA"), (3, "vibert", "ViBERT"), (4, "xlmr", "XLM-R"),
    (5, "mbert_bilstm", "mBERT + Bi-LSTM"), (6, "velectra_bilstm", "vELECTRA + Bi-LSTM"),
    (7, "vibert_bilstm", "ViBERT + Bi-LSTM"), (8, "xlmr_bilstm", "XLM-R + Bi-LSTM"),
    (9, "mbert_crf", "mBERT + CRF"), (10, "velectra_crf", "vELECTRA + CRF"),
    (11, "vibert_crf", "ViBERT + CRF"), (12, "xlmr_crf", "XLM-R + CRF"),
    (13, "mbert_bilstm_crf", "mBERT + Bi-LSTM + CRF"), (14, "velectra_bilstm_crf", "vELECTRA + Bi-LSTM + CRF"),
    (15, "vibert_bilstm_crf", "ViBERT + Bi-LSTM + CRF"), (16, "xlmr_bilstm_crf", "XLM-R + Bi-LSTM + CRF"),
    (17, "phopunct", "**PhoPunct** (PhoBERT-large + Bi-LSTM + CRF)"),
]
DOMAINS = ["news", "novels"]
VALID_RE = re.compile(r"Epoch (\d+) - Valid macro F1 = ([0-9.]+)")


def from_log(path):
    """Best validation macro F1 and its epoch, parsed from a training log."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        scores = [(float(f1), int(ep)) for ep, f1 in VALID_RE.findall(f.read())]
    if not scores:
        return None
    f1, ep = max(scores, key=lambda s: (s[0], -s[1]))
    return {"valid": f1, "epoch": ep, "n_epochs": len(scores)}


def from_outputs(run_dir):
    res = {}
    for split in ("valid", "test"):
        p = os.path.join(run_dir, f"{split}_metrics.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            res[split] = m["macro_f1"]
            res["epoch"] = m.get("epoch")
    return res or None


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "–"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--logs")
    g.add_argument("--outputs")
    args = ap.parse_args()

    splits = ["valid"] if args.logs else ["valid", "test"]
    header = ["#", "Model"] + [f"{d.capitalize()} {s}" for d in DOMAINS for s in splits]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for num, run, name in RUNS:
        cells = [str(num), name]
        for d in DOMAINS:
            if args.logs:
                r = from_log(os.path.join(args.logs, f"{run}_{d}.log"))
            else:
                r = from_outputs(os.path.join(args.outputs, f"{run}_{d}"))
            for s in splits:
                v = r.get(s) if r else None
                cell = fmt(v)
                if v is not None and r.get("epoch"):
                    cell += f" (ep {r['epoch']})"
                cells.append(cell)
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
