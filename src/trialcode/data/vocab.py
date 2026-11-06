"""Code tokens with hierarchical embeddings.

Every code in the vocabulary (see mimic.code_vocabulary) is one token. Its *input* embedding is
the sum of its own row and the rows of its ICD prefix ancestors (E1165 = E1165 + E116 + E11), so
rarely seen codes inherit from their ancestors. The MLM output head is untied from the input
embeddings and holds one flat row per code.

Initialisation makes the composed input embedding of each code equal the mean embedding of its
description's tokens (own row = description mean - parent's description mean), so a code is
text-aware at step 0 and then drifts toward how it is used.
"""
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

VOCAB_FILE = "code_vocab.json"


class AncestorEmbedding(nn.Embedding):
    """nn.Embedding whose output adds the rows of each id's ancestors (via `parent`, -1 = none)."""
    depth = 8

    def forward(self, ids):
        out = super().forward(ids)
        p = self.parent[ids]
        for _ in range(self.depth):
            valid = p >= 0
            if not valid.any():
                break
            safe = p.clamp(min=0)
            out = out + F.embedding(safe, self.weight) * valid.unsqueeze(-1)
            p = torch.where(valid, self.parent[safe], p)
        return out


def attach_hierarchy(model, tokenizer, vocab):
    """Make the model's input embedding hierarchical (class swap: same weights and state_dict keys)."""
    emb = model.get_input_embeddings()
    parent = torch.full((emb.num_embeddings,), -1, dtype=torch.long)
    for v in vocab:
        if v["parent"]:
            parent[tokenizer.convert_tokens_to_ids(v["token"])] = tokenizer.convert_tokens_to_ids(v["parent"])
    emb.__class__ = AncestorEmbedding
    emb.register_buffer("parent", parent.to(emb.weight.device), persistent=False)


def add_code_tokens(tokenizer, model, vocab):
    """Register `vocab` as tokens, resize and initialise the model, and make it hierarchical."""
    title_ids = {v["token"]: tokenizer(v["title"] or v["code"], add_special_tokens=False)["input_ids"] for v in vocab}
    tokenizer.add_tokens([v["token"] for v in vocab])
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    W = model.get_input_embeddings().weight
    children = defaultdict(list)
    for v in vocab:
        if v["parent"]:
            children[v["parent"]].append(v["token"])
    mean = {}  # description-mean embedding per code; longest codes first so children precede parents
    with torch.no_grad():
        for v in sorted(vocab, key=lambda v: -len(v["code"])):
            t = v["token"]
            kids = [mean[c] for c in children[t] if c in mean]
            mean[t] = W[title_ids[t]].mean(0) if v["title"] or not kids else torch.stack(kids).mean(0)

        model.config.tie_word_embeddings = False  # flat output rows, composed input rows
        model.decoder.weight = nn.Parameter(W.detach().clone())
        for v in vocab:
            i = tokenizer.convert_tokens_to_ids(v["token"])
            model.decoder.weight[i] = mean[v["token"]]
            W[i] = mean[v["token"]] - (mean[v["parent"]] if v["parent"] else 0)
    attach_hierarchy(model, tokenizer, vocab)


def save_vocab(vocab, directory):
    with open(os.path.join(directory, VOCAB_FILE), "w") as f:
        json.dump(vocab, f)


def load_vocab(directory):
    with open(os.path.join(directory, VOCAB_FILE)) as f:
        return json.load(f)


def description_examples(vocab, with_text):
    """One row per code that has a title. with_text=True puts the title in the note slot
    (title -> code); False leaves it empty so the code section's own title is the only text
    (code -> title)."""
    return [{"note_id": "", "subject_id": "0", "hadm_id": "0",
             "text": v["title"] if with_text else "",
             "codes": [{k: v[k] for k in ("kind", "icd_version", "code", "title")}]}
            for v in vocab if v["title"]]
