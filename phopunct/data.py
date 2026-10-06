# -*- coding: utf-8 -*-
"""Word-level data pipeline shared by all 17 model configurations.

Input files contain one ``<token> <label>`` pair per line. Consecutive lines are
grouped into chunks of at most 128 words, cut only after a sentence-final label
(PERIOD / QMARK / EXCLAM). Each chunk is then tokenized word-by-word; the model
gathers the representation of the first sub-word of each word, so labels live
purely in word space (7 classes, no [CLS]/[SEP] sentinel labels).

Two correctness details matter here:
  (1) Truncation happens at word boundaries: the sub-word budget is checked
      *before* a word's pieces are appended, so sub-words and labels never
      become misaligned.
  (2) Sub-word-space padding (input_ids / attention_mask / word_starts) and
      word-space padding (label_ids / word_mask) are done in separate loops, so
      word_mask never contains 1s at padded positions.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

LABELS: List[str] = ["O", "PERIOD", "COMMA", "COLON", "QMARK", "EXCLAM", "SEMICOLON"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
# Classes reported in evaluation ("O" is excluded).
PUNCT_LABELS: List[str] = ["PERIOD", "COMMA", "COLON", "QMARK", "EXCLAM", "SEMICOLON"]


# --------------------------------------------------------------------------- #
# 1. Read "token label" file -> chunks of <= 128 words cut at sentence ends
# --------------------------------------------------------------------------- #
def read_examples(filepath: str, max_words_per_chunk: int = 128) -> List[Tuple[List[str], List[str]]]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    tokens, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                logger.warning("Skipping malformed line %s:%d -> %r", filepath, line_no, line)
                continue
            tok, lab = parts
            if lab not in LABEL2ID:
                logger.warning("Unknown label '%s' at %s:%d, treating as 'O'", lab, filepath, line_no)
                lab = "O"
            tokens.append(tok)
            labels.append(lab)

    df = pd.DataFrame({"token": tokens, "label": labels})
    n = len(df)
    examples: List[Tuple[List[str], List[str]]] = []
    idx = 0
    while 0 <= idx < n:
        sub_df = df.iloc[idx: min(idx + max_words_per_chunk, n)]
        end_idx = sub_df[sub_df.label.isin(EOS_MARKS)].tail(1).index
        if end_idx.empty:
            chunk = df.iloc[idx:]
            next_idx = -1
        else:
            cut = end_idx.item() + 1
            chunk = df.iloc[idx:cut]
            next_idx = cut
        if len(chunk) > 0:
            examples.append((chunk.token.tolist(), chunk.label.tolist()))
        idx = next_idx
    return examples


# --------------------------------------------------------------------------- #
# 2. Convert to word-level feature tensors
# --------------------------------------------------------------------------- #
@dataclass
class WordFeature:
    input_ids: List[int]        # sub-word level, len = max_seq_length
    attention_mask: List[int]   # sub-word level
    word_starts: List[int]      # sub-word level, 1 at the first sub-word of each word
    label_ids: List[int]        # word level, len = max_seq_length (>= number of real words)
    word_mask: List[int]        # word level, 1 for real word positions


def encode_words(words: List[str], tokenizer, max_seq_length: int, encode_word=None):
    """Tokenize a list of words into padded sub-word inputs.

    Returns (input_ids, attention_mask, word_starts, kept_indices, truncated) where
    ``kept_indices`` are the indices of the words that fit into the budget.
    Words that tokenize to nothing are skipped.
    """
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if encode_word is None:
        def encode_word(w):
            return tokenizer.encode(w, add_special_tokens=False)

    budget = max_seq_length - 2  # reserve room for <s> and </s>
    subword_ids: List[int] = []
    word_starts: List[int] = []
    kept: List[int] = []
    truncated = False

    for i, w in enumerate(words):
        piece_ids = encode_word(w)
        if not piece_ids:
            continue
        # Fix (1): cut at a WORD boundary, checking the budget before adding the word.
        if len(subword_ids) + len(piece_ids) > budget:
            truncated = True
            break
        subword_ids.extend(piece_ids)
        word_starts.extend([1] + [0] * (len(piece_ids) - 1))
        kept.append(i)

    input_ids = [bos_id] + subword_ids + [eos_id]
    word_starts_full = [0] + word_starts + [0]
    attention_mask = [1] * len(input_ids)

    # Fix (2): sub-word-space padding only; word-space padding is done by the caller.
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_id)
        attention_mask.append(0)
        word_starts_full.append(0)

    return input_ids, attention_mask, word_starts_full, kept, truncated


def convert_examples_to_features(
    examples: List[Tuple[List[str], List[str]]],
    tokenizer,
    max_seq_length: int = 256,
) -> List[WordFeature]:
    # Cache tokenizer.encode per unique word type to speed up large corpora.
    encode_cache: Dict[str, List[int]] = {}

    def encode_word(word: str) -> List[int]:
        if word not in encode_cache:
            encode_cache[word] = tokenizer.encode(word, add_special_tokens=False)
        return encode_cache[word]

    features: List[WordFeature] = []
    n_dropped_empty = 0
    n_truncated = 0

    for words, labels in examples:
        input_ids, attention_mask, word_starts_full, kept, truncated = encode_words(
            words, tokenizer, max_seq_length, encode_word=encode_word)

        if not kept:
            n_dropped_empty += 1
            continue
        if truncated:
            n_truncated += 1

        label_ids = [LABEL2ID[labels[i]] for i in kept]
        word_mask = [1] * len(kept)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)   # 0 = "O", masked out so it never affects the loss
            word_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(word_starts_full) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(word_mask) == max_seq_length
        assert sum(word_starts_full) == sum(word_mask), "word_starts must match the number of real words"

        features.append(WordFeature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_starts=word_starts_full,
            label_ids=label_ids,
            word_mask=word_mask,
        ))

    logger.info(
        "convert_examples_to_features: %d examples -> %d features (%d empty dropped, %d truncated)",
        len(examples), len(features), n_dropped_empty, n_truncated,
    )
    return features


def features_to_dataset(features: List[WordFeature]) -> TensorDataset:
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    word_starts = torch.tensor([f.word_starts for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    word_mask = torch.tensor([f.word_mask for f in features], dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, word_starts, label_ids, word_mask)


def load_dataset(data_dir: str, split: str, tokenizer, max_seq_length: int,
                 max_examples: Optional[int] = None) -> TensorDataset:
    """split in {'train', 'valid', 'test'}"""
    fname = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}[split]
    examples = read_examples(os.path.join(data_dir, fname))
    if max_examples is not None:
        examples = examples[:max_examples]
    features = convert_examples_to_features(examples, tokenizer, max_seq_length)
    return features_to_dataset(features)
