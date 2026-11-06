"""MLM collator that preferentially masks codes so the model must predict them from context."""
import random

import torch

from .tokenize import CODE_PART_ID, CODE_PART_TITLE


class CodeAwareMLMCollator:
    """Standard BERT masking (80/10/10) on ordinary tokens, plus whole-code masking.

    Each code in the code section is selected with prob `code_prob`; a selected code is masked as
      both identifier and title (predict from the note)   p = 0.5
      identifier only (title visible: description -> code) p = 0.25
      title only      (code visible: code -> description) p = 0.25
    Masked code tokens follow the same 80/10/10 corruption as ordinary tokens.
    """

    def __init__(self, tokenizer, mlm_probability=0.15, code_prob=0.5, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.mlm_probability = mlm_probability
        self.code_prob = code_prob
        self.pad_to_multiple_of = pad_to_multiple_of
        self.special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

    def _select(self, ids, groups, parts):
        selected = [False] * len(ids)
        modes = {}
        has_title = {g for g, p in zip(groups, parts) if p == CODE_PART_TITLE}
        for g in set(groups) - {0}:
            if random.random() < self.code_prob:
                modes[g] = (random.choices(["both", "id", "title"], weights=[0.5, 0.25, 0.25])[0]
                            if g in has_title else "id")
        for i, (t, g, p) in enumerate(zip(ids, groups, parts)):
            if g == 0:
                selected[i] = t not in self.special and random.random() < self.mlm_probability
            elif g in modes:
                m = modes[g]
                selected[i] = m == "both" or (m == "id" and p == CODE_PART_ID) or (m == "title" and p == CODE_PART_TITLE)
        return selected

    def __call__(self, features):
        batch = self.tok.pad({"input_ids": [f["input_ids"] for f in features]},
                             pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        L, mask_id, V = batch["input_ids"].shape[1], self.tok.mask_token_id, len(self.tok)
        labels = torch.full((len(features), L), -100)
        for b, f in enumerate(features):
            ids = batch["input_ids"][b]
            sel = torch.tensor(self._select(f["input_ids"], f["code_group"], f["code_part"]))
            n = len(f["input_ids"])
            labels[b, :n][sel] = ids[:n][sel]
            r = torch.rand(n)
            ids[:n][sel & (r < 0.8)] = mask_id
            rand = sel & (r >= 0.8) & (r < 0.9)
            ids[:n][rand] = torch.randint(V, (int(rand.sum()),))
        batch["labels"] = labels
        return batch
