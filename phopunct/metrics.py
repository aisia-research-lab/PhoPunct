# -*- coding: utf-8 -*-
"""Evaluation: macro / micro / weighted P-R-F1 over the 6 punctuation classes ("O" excluded)."""
from __future__ import annotations

import torch
from sklearn.metrics import classification_report

from .data import ID2LABEL, PUNCT_LABELS


@torch.no_grad()
def predict_dataset(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        input_ids, attention_mask, word_starts, label_ids, word_mask = (t.to(device) for t in batch)
        pred_seqs = model(input_ids, attention_mask, word_starts, label_ids=None, word_mask=word_mask)
        gold = label_ids.cpu().numpy()
        mask = word_mask.cpu().numpy()
        for i, seq in enumerate(pred_seqs):
            n = int(mask[i].sum())
            y_true.extend(ID2LABEL[t] for t in gold[i][:n])
            y_pred.extend(ID2LABEL[t] for t in seq[:n])
    return y_true, y_pred


def score(y_true, y_pred) -> dict:
    report_dict = classification_report(
        y_true, y_pred, labels=PUNCT_LABELS, digits=4, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, labels=PUNCT_LABELS, digits=4, zero_division=0
    )
    return {"macro_f1": report_dict["macro avg"]["f1-score"], "report": report_str, "report_dict": report_dict}


def evaluate(model, dataloader, device) -> dict:
    return score(*predict_dataset(model, dataloader, device))
