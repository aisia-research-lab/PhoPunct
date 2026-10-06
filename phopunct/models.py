# -*- coding: utf-8 -*-
"""Model definition shared by all 17 configurations.

    tokens -> sub-word tokenizer -> transformer backbone
           -> gather first sub-word of each word (word_starts)   [before any Bi-LSTM]
           -> (optional) Bi-LSTM(hidden=128, 1 layer, bidirectional)
           -> dropout -> Linear -> float32 logits
           -> CRF (Viterbi decode)          if use_crf
              softmax / argmax (masked CE)  otherwise

| #     | backbone                          | Bi-LSTM | CRF |
|-------|-----------------------------------|---------|-----|
| 1-4   | mbert / velectra / vibert / xlmr  |   no    | no  |
| 5-8   | mbert / velectra / vibert / xlmr  |   yes   | no  |
| 9-12  | mbert / velectra / vibert / xlmr  |   no    | yes |
| 13-16 | mbert / velectra / vibert / xlmr  |   yes   | yes |
| 17    | phobert (PhoPunct)                |   yes   | yes |
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

BACKBONES: Dict[str, str] = {
    "mbert": "bert-base-multilingual-cased",
    "velectra": "FPTAI/velectra-base-discriminator-cased",
    "vibert": "NlpHUST/vibert4news-base-cased",
    "xlmr": "xlm-roberta-base",
    "phobert": "vinai/phobert-large",
}
BACKBONE_DISPLAY: Dict[str, str] = {
    "mbert": "mBERT",
    "velectra": "vELECTRA",
    "vibert": "ViBERT",
    "xlmr": "XLM-R",
    "phobert": "PhoBERT-large",
}
# Older checkpoints stored "bert" for ViBERT.
BACKBONE_ALIASES: Dict[str, str] = {"bert": "vibert"}

# Checkpoints such as NlpHUST/vibert4news-base-cased lack "model_type" /
# "tokenizer_class" metadata, so AutoModel/AutoTokenizer cannot detect them.
# Standard WordPiece BERT backbones are therefore loaded with the Bert* classes.
_BERT_CLASS_BACKBONES = {"mbert", "vibert"}


def canonical_backbone(key: str) -> str:
    return BACKBONE_ALIASES.get(key, key)


def display_name(backbone: str, use_bilstm: bool, use_crf: bool) -> str:
    backbone = canonical_backbone(backbone)
    if backbone == "phobert" and use_bilstm and use_crf:
        return "PhoPunct (PhoBERT-large + Bi-LSTM + CRF)"
    parts = [BACKBONE_DISPLAY[backbone]]
    if use_bilstm:
        parts.append("Bi-LSTM")
    if use_crf:
        parts.append("CRF")
    return " + ".join(parts)


def load_tokenizer(backbone: str, model_name_or_path: Optional[str] = None):
    backbone = canonical_backbone(backbone)
    name = model_name_or_path or BACKBONES[backbone]
    if backbone in _BERT_CLASS_BACKBONES:
        return BertTokenizer.from_pretrained(name)
    return AutoTokenizer.from_pretrained(name, use_fast=False)


class WordTagger(nn.Module):
    """Transformer backbone [+ Bi-LSTM] + linear classifier [+ CRF], word level."""

    def __init__(self, backbone: str, model_name_or_path: Optional[str] = None, num_labels: int = 7,
                 use_bilstm: bool = False, use_crf: bool = False,
                 lstm_hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        backbone = canonical_backbone(backbone)
        name = model_name_or_path or BACKBONES[backbone]
        # NOTE: sub-module creation order (bert, lstm, dropout, classifier, crf) is kept
        # identical to the scripts used for the paper so that seeded runs reproduce.
        if backbone in _BERT_CLASS_BACKBONES:
            self.bert = BertModel.from_pretrained(name)
        else:
            self.bert = AutoModel.from_pretrained(name)
        hidden = self.bert.config.hidden_size
        self.use_bilstm = use_bilstm
        self.use_crf = use_crf
        if use_bilstm:
            self.lstm = nn.LSTM(
                input_size=hidden, hidden_size=lstm_hidden_size,
                num_layers=1, batch_first=True, bidirectional=True,
            )
            classifier_in = lstm_hidden_size * 2
        else:
            self.lstm = None
            classifier_in = hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, num_labels)
        self.crf = CRF(num_labels, batch_first=True) if use_crf else None
        self.num_labels = num_labels

    @staticmethod
    def _gather_word_level(sequence_output: torch.Tensor, word_starts: torch.Tensor) -> torch.Tensor:
        """Gather the first sub-word representation of each word -> padded word sequence."""
        batch_size, _, feat_dim = sequence_output.shape
        max_words = word_starts.shape[1]
        word_hidden = torch.zeros(batch_size, max_words, feat_dim,
                                  dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            idx = torch.nonzero(word_starts[i], as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n == 0:
                continue
            word_hidden[i, :n] = sequence_output[i, idx]
        return word_hidden

    def forward(self, input_ids, attention_mask, word_starts, label_ids=None, word_mask=None):
        """With ``label_ids``: returns the training loss.
        Without: returns a list (batch) of predicted label-id lists."""
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        word_hidden = self._gather_word_level(bert_out, word_starts)
        if self.use_bilstm:
            word_hidden, _ = self.lstm(word_hidden)
        word_hidden = self.dropout(word_hidden)
        # Cast to float32 before the CRF: its logsumexp yields NaN gradients under fp16.
        logits = self.classifier(word_hidden).float()

        if self.use_crf:
            if label_ids is not None:
                log_likelihood = self.crf(logits, label_ids, mask=word_mask.bool(), reduction="mean")
                return -1.0 * log_likelihood
            mask = word_mask.bool() if word_mask is not None else None
            return self.crf.decode(logits, mask=mask)

        if label_ids is not None:
            # Per-token CE masked by word_mask (padded positions carry a dummy "O" label,
            # so ignore_index cannot be used).
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            mask_flat = word_mask.view(-1).float()
            return (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        pred_ids = logits.argmax(dim=-1)
        mask = word_mask.bool()
        result: List[List[int]] = []
        for i in range(pred_ids.shape[0]):
            n = int(mask[i].sum().item())
            result.append(pred_ids[i, :n].tolist())
        return result


def load_checkpoint(path: str, device="cpu", model_name_or_path: Optional[str] = None):
    """Rebuild a WordTagger from a checkpoint written by train.py.

    Also accepts checkpoints from the original per-family scripts, which did not
    store the full config (PhoPunct checkpoints had no metadata at all).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config")
    if cfg is None:
        state = ckpt["model_state_dict"]
        backbone = canonical_backbone(ckpt.get("model_key", "phobert"))
        cfg = {
            "backbone": backbone,
            "use_bilstm": ckpt.get("use_bilstm", any(k.startswith("lstm.") for k in state)),
            "use_crf": any(k.startswith("crf.") for k in state),
            "lstm_hidden_size": state["lstm.weight_hh_l0"].shape[1] if "lstm.weight_hh_l0" in state else 128,
            "model_name_or_path": None,
        }
    name = model_name_or_path or cfg.get("model_name_or_path")
    model = WordTagger(cfg["backbone"], name, use_bilstm=cfg["use_bilstm"], use_crf=cfg["use_crf"],
                       lstm_hidden_size=cfg["lstm_hidden_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tokenizer = load_tokenizer(cfg["backbone"], name)
    return model, tokenizer, cfg, ckpt
