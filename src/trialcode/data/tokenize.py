"""Turn raw MIMIC examples into fixed-length token sequences with code-span annotations.

Each sequence is  [CLS] note chunk [SEP] <code section> [SEP]  where the code section lists
the admission's codes as "<code> <long title>". Every chunk of a long note carries the same
code section, so codes can be predicted from any part of the note.

With code_tokens=True each code is a single vocabulary token (see vocab.py) instead of its text
spelling; include_titles=False drops the descriptions (text-free sequences).

Per-token side channels (used by CodeAwareMLMCollator):
    code_group  0 = ordinary text, k>0 = the k-th code in the section
    code_part   1 = code identifier tokens, 2 = title tokens (0 for ordinary text)
"""
from .mimic import code_ancestors, code_token

CODE_PART_ID, CODE_PART_TITLE = 1, 2


def _encode_codes(tokenizer, codes, max_tokens, code_tokens, include_titles, vocab):
    ids, group, part = [], [], []
    for k, c in enumerate(codes, start=1):
        cid = None
        for code in (c["code"], *code_ancestors(c["kind"], c["icd_version"], c["code"])):
            tok = code_token(c["kind"], c["icd_version"], code)
            if tok in vocab:  # own token, else back off to the nearest ancestor that has one
                cid = [vocab[tok]]
                break
        if cid is None:
            if code_tokens and not include_titles:
                continue  # code-token-only sequences skip codes with no token or ancestor
            cid = tokenizer(f" {c['code']}", add_special_tokens=False)["input_ids"]
        title = (tokenizer(f" {c['title']};", add_special_tokens=False)["input_ids"]
                 if include_titles else [])
        if len(ids) + len(cid) + len(title) > max_tokens:
            break
        ids += cid + title
        group += [k] * (len(cid) + len(title))
        part += [CODE_PART_ID] * len(cid) + [CODE_PART_TITLE] * len(title)
    return ids, group, part


def make_tokenize_fn(tokenizer, max_length=2048, max_code_tokens=512, min_note_tokens=64,
                     code_tokens=False, include_titles=True):
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    vocab = tokenizer.get_vocab() if code_tokens else {}  # once, not per example

    def fn(batch):
        out = {"input_ids": [], "code_group": [], "code_part": []}
        for text, codes in zip(batch["text"], batch["codes"]):
            c_ids, c_grp, c_part = _encode_codes(tokenizer, codes, max_code_tokens, code_tokens, include_titles, vocab)
            tail_ids = [sep] + c_ids + ([sep] if c_ids else [])
            n_tail = len(tail_ids)
            note_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
            size = max_length - 1 - n_tail
            chunks = [note_ids[i:i + size] for i in range(0, len(note_ids), size)] or [[]]
            for j, chunk in enumerate(chunks):
                if j > 0 and len(chunk) < min_note_tokens:
                    continue  # drop tiny trailing remainder
                ids = [cls] + chunk + tail_ids
                pad_n = 1 + len(chunk) + 1  # CLS + note + first SEP
                out["input_ids"].append(ids)
                out["code_group"].append([0] * pad_n + c_grp + ([0] if c_ids else []))
                out["code_part"].append([0] * pad_n + c_part + ([0] if c_ids else []))
        return out

    return fn
