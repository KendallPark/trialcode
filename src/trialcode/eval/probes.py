"""Probes for whether code embeddings capture clinical usage rather than just text.

    python -m trialcode.eval.probes --model outputs/<run>/final --mimic-root data/mimic \\
        --variant structured --baseline-model answerdotai/ModernBERT-base

dx_from_context  mask every diagnosis in held-out admissions and rank diagnosis tokens from the
                 remaining codes (procedures, drugs, abnormal labs, HCPCS). Drugs/labs share no
                 words with diagnoses, so text search cannot solve this. Compared with a
                 popularity prior and, via --baseline-model, the untrained (description-initialised) model.
neighbours       fraction of each ICD-10 diagnosis's nearest neighbours (cosine over token
                 embeddings) in the same 3-character category / chapter.
"""
import argparse
import json

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from trialcode.data.mimic import load_mimic, split_by_subject
from trialcode.data.tokenize import make_tokenize_fn
from trialcode.data.vocab import add_code_tokens, attach_hierarchy, load_vocab


def dx_from_context(model, tok, test_rows, vocab, drop_prefixes=("[DRG",), k=10, device="cuda"):
    dx = [v for v in vocab if v["kind"] == "diagnosis" and not v["ancestor"]]
    dx_ids = torch.tensor([tok.convert_tokens_to_ids(v["token"]) for v in dx], device=device)
    dx_index = {i: j for j, i in enumerate(dx_ids.tolist())}
    pop_top = set(tok.convert_tokens_to_ids(v["token"]) for v in sorted(dx, key=lambda v: -v["count"])[:k])
    model.eval()
    rec = pop_rec = mrr = n = n_gold = n_backoff = 0
    for row in test_rows:
        toks = tok.convert_ids_to_tokens(row["input_ids"])
        masked = [i for i, t in enumerate(toks) if t.startswith("[DX")]  # all diagnoses, incl. ancestor backoffs
        keep = [i for i, t in enumerate(toks) if not t.startswith(drop_prefixes)]
        if not masked or len(keep) <= len(masked) + 2:
            continue
        ids = torch.tensor(row["input_ids"])
        orig = ids.clone()
        ids[masked] = tok.mask_token_id
        ids = ids[keep].to(device)
        scored = [i for i in masked if orig[i].item() in dx_index]  # gold codes that have their own token
        n_gold += len(masked)
        n_backoff += len(masked) - len(scored)
        if not scored:
            continue
        with torch.no_grad():
            logits = model(input_ids=ids[None]).logits[0, [keep.index(i) for i in scored]][:, dx_ids]
        gold_idx = torch.tensor([dx_index[orig[i].item()] for i in scored], device=device)
        ranks = (logits > logits.gather(1, gold_idx[:, None])).sum(-1)  # 0 = best
        rec += (ranks < k).float().mean().item()
        mrr += (1.0 / (ranks + 1).float()).mean().item()
        pop_rec += sum(orig[i].item() in pop_top for i in scored) / len(scored)
        n += 1
    return {f"recall@{k}": rec / n, "mrr": mrr / n, f"popularity_recall@{k}": pop_rec / n,
            "n_admissions": n, "frac_gold_backoff": n_backoff / max(n_gold, 1)}


def neighbours(model, tok, vocab, k=10, min_count=20):
    dx = [v for v in vocab if v["kind"] == "diagnosis" and v["icd_version"] == "10" and not v["ancestor"]
          and v["count"] >= min_count]
    ids = torch.tensor([tok.convert_tokens_to_ids(v["token"]) for v in dx])
    with torch.no_grad():  # composed input embeddings (own row + ancestors)
        e = torch.nn.functional.normalize(model.get_input_embeddings()(ids.to(model.device)).float().cpu(), dim=-1)
    sim = e @ e.T
    sim.fill_diagonal_(-2)
    k = min(k, len(dx) - 1)
    top = sim.topk(k, dim=-1).indices
    cat = torch.tensor([hash(v["code"][:3]) for v in dx])
    chap = torch.tensor([hash(v["code"][0]) for v in dx])
    return {"n_codes": len(dx),
            f"same_category@{k}": (cat[top] == cat[:, None]).float().mean().item(),
            f"same_chapter@{k}": (chap[top] == chap[:, None]).float().mean().item()}


def run(model, tok, vocab, test_rows, k, device):
    return {"dx_from_context": dx_from_context(model.to(device), tok, test_rows, vocab, k=k, device=device),
            "neighbours": neighbours(model, tok, vocab, k=k)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mimic-root", required=True)
    p.add_argument("--variant", default="structured", choices=["codes", "structured", "all"])
    p.add_argument("--baseline-model", default=None, help="base model; evaluated with description-initialised code tokens")
    p.add_argument("--max-admissions", type=int, default=5000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = load_vocab(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    test = split_by_subject(load_mimic(a.mimic_root, "structured" if a.variant == "all" else a.variant))["test"]
    test = test.select(range(min(a.max_admissions, len(test))))
    rows = test.map(make_tokenize_fn(tok, 1024, code_tokens=True, include_titles=False),
                    batched=True, remove_columns=test.column_names)
    model = AutoModelForMaskedLM.from_pretrained(a.model)
    attach_hierarchy(model, tok, vocab)
    results = {"trained": run(model, tok, vocab, rows, a.k, device)}
    if a.baseline_model:
        btok = AutoTokenizer.from_pretrained(a.baseline_model)
        bmodel = AutoModelForMaskedLM.from_pretrained(a.baseline_model)
        add_code_tokens(btok, bmodel, vocab)
        results["untrained_init"] = run(bmodel, btok, vocab, rows, a.k, device)
    print(json.dumps(results, indent=2))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
