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

from trialcode.data.mimic import VARIANT_KINDS, code_vocabulary, load_mimic, split_by_subject
from trialcode.data.tokenize import make_tokenize_fn
from trialcode.data.vocab import add_code_tokens


def dx_from_context(model, tok, test_rows, vocab, drop_prefixes=("[DRG",), k=10, device="cuda"):
    dx = [v for v in vocab if v["kind"] == "diagnosis"]
    dx_ids = torch.tensor([tok.convert_tokens_to_ids(v["token"]) for v in dx], device=device)
    pop_top = set(tok.convert_tokens_to_ids(v["token"]) for v in sorted(dx, key=lambda v: -v["count"])[:k])
    model.eval()
    rec = pop_rec = mrr = n = 0
    for row in test_rows:
        toks = tok.convert_ids_to_tokens(row["input_ids"])
        gold = [i for i, t in enumerate(toks) if t.startswith("[DX")]
        keep = [i for i, t in enumerate(toks) if not t.startswith(drop_prefixes)]
        if not gold or len(keep) <= len(gold) + 2:
            continue
        ids = torch.tensor(row["input_ids"])
        gold_ids = ids[gold].tolist()
        ids[gold] = tok.mask_token_id
        ids = ids[keep].to(device)
        pos = [keep.index(i) for i in gold]
        with torch.no_grad():
            logits = model(input_ids=ids[None]).logits[0, pos][:, dx_ids]
        ranks = (logits.argsort(-1, descending=True) == torch.stack(
            [(dx_ids == g).nonzero()[0, 0] for g in map(torch.tensor, gold_ids)]).to(device)[:, None]).float().argmax(-1)
        rec += (ranks < k).float().mean().item()
        mrr += (1.0 / (ranks + 1).float()).mean().item()
        pop_rec += sum(g in pop_top for g in gold_ids) / len(gold_ids)
        n += 1
    return {f"recall@{k}": rec / n, "mrr": mrr / n, f"popularity_recall@{k}": pop_rec / n, "n_admissions": n}


def neighbours(model, tok, vocab, k=10, min_count=20):
    dx = [v for v in vocab if v["kind"] == "diagnosis" and v["icd_version"] == "10" and v["count"] >= min_count]
    ids = torch.tensor([tok.convert_tokens_to_ids(v["token"]) for v in dx])
    e = torch.nn.functional.normalize(model.get_input_embeddings().weight[ids].float().detach().cpu(), dim=-1)
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
    p.add_argument("--min-code-count", type=int, default=5)
    p.add_argument("--max-admissions", type=int, default=5000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = code_vocabulary(a.mimic_root, a.min_code_count, VARIANT_KINDS[a.variant])
    tok = AutoTokenizer.from_pretrained(a.model)
    test = split_by_subject(load_mimic(a.mimic_root, "structured" if a.variant == "all" else a.variant))["test"]
    test = test.select(range(min(a.max_admissions, len(test))))
    rows = test.map(make_tokenize_fn(tok, 1024, code_tokens=True, include_titles=False),
                    batched=True, remove_columns=test.column_names)
    results = {"trained": run(AutoModelForMaskedLM.from_pretrained(a.model), tok, vocab, rows, a.k, device)}
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
