"""Proof-of-concept agent eval on the trial-criteria set (ICD-9/ICD-10 diagnoses only).

    python -m trialcode.eval.poc_agent data/eval/trial_criteria outputs/<run>/final data/mimic [--k 10 25]

The "agent" (Claude) proposed seed codes from each criterion's text alone, before seeing any reference
codes. Seeds ending in '*' mean "this code and everything under it". Three conditions, scored as
exact-code micro precision/recall/F1 against the expert reference sets:

  agent      the seeds exactly as proposed ('*' expanded through the ICD dictionary, the verifier step)
  +model     agent codes plus, per criterion, the k diagnosis tokens nearest (cosine over the model's
             composed input embeddings) to the mean embedding of the seeds, same ICD version only
  model_only the same k neighbours without the seeds (how much the model finds by itself)

With --dump the candidate neighbours (with ICD titles) are written for the agent to read; with --keep
(json {criterion id: "code code ..."}, dotless) the agent's picks are scored as "+filtered@k".
"""
import argparse
import gzip
import json
from collections import defaultdict

import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForMaskedLM, AutoTokenizer

from trialcode.data.vocab import attach_hierarchy, load_vocab

SEEDS = {  # criterion id -> {icd version: seeds}; written from the criterion text only
    "C01": {"10": "N18.2", "9": "585.2"},
    "C02": {"10": "N18.2 N18.30 N18.4", "9": "585.2 585.3 585.4"},
    "C03": {"10": "E10*", "9": "250.01 250.03"},
    "C04": {"10": "Q61*", "9": "753.1*"},
    "C05": {"10": "N17*", "9": "584*"},
    "C07": {"10": "I10", "9": "401*"},
    "C08": {"10": "Z95.820 Z98.62 Z95.828"},
    "C09": {"10": "E10.2* E10.3* E11.2* E11.3* E11.4* I51.7", "9": "250.4* 250.5* 250.6* 429.3"},
    "C10": {"10": "Z89.4* Z89.5* Z89.6* S78* S88*"},
    "C11": {"10": "I60*", "9": "430"},
    "C14": {"10": "J44*", "9": "491* 492* 496"},
    "C15": {"10": "E88.01", "9": "273.4"},
    "C16": {"10": "I21* I20.0", "9": "410* 411.1"},
    "C17": {"10": "J00 J01* J02* J03* J04* J06* J20* J21* J22 J18*", "9": "460 461* 465* 466* 486"},
    "C19": {"10": "K74* K70.3*", "9": "571.5 571.2"},
    "C20": {"10": "N20* N21*", "9": "592* 594*"},
    "C21": {"10": "N25.0", "9": "588.0"},
    "C23": {"10": "I21* I25.2 I50*", "9": "410* 414.8 428*"},
    "C24": {"10": "E66.01 Z68.4*", "9": "278.01"},
    "C25": {"10": "I10 I16*", "9": "401*"},
    "C27": {"10": "E10.1* E11.1* E13.1*", "9": "250.1*"},
    "C28": {"10": "I82.4* I80.2*", "9": "453.4* 451.1*"},
    "C29": {"10": "I60* I61* I62*", "9": "430 431 432*"},
    "C30": {"10": "K92.2 K25* K26* K27* K28*", "9": "578* 531* 532* 533* 534*"},
    "C31": {"10": "Z95.2"},
    "C33": {"10": "F10* F11* F12* F13* F14* F15* F16* F18* F19*", "9": "303* 304* 305*"},
    "C34": {"10": "L20* J30*", "9": "691.8 477*"},
    "C35": {"10": "M32* K50* K51*", "9": "710.0 555* 556*"},
    "C36": {"10": "E21* E05*", "9": "252.0* 242*"},
    "C37": {"10": "E24* E27*", "9": "255*"},
    "C38": {"10": "I50*", "9": "428*"},
    "C39": {"10": "I21* I63*", "9": "410* 434*"},
    "C40": {"10": "E87.6 E83.42", "9": "276.8 275.2"},
}


def expand(seeds, dictionary):
    out = set()
    for s in seeds.split():
        s = s.replace(".", "")
        out |= {c for c in dictionary if c.startswith(s[:-1])} if s.endswith("*") else ({s} & dictionary)
    return out


def prf(tp, n_pred, n_ref):
    p, r = tp / max(n_pred, 1), tp / max(n_ref, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("criteria")
    ap.add_argument("model")
    ap.add_argument("mimic_root")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--dump", help="write candidate neighbours (max k) with titles to this json")
    ap.add_argument("--keep", help="json of the agent's picks among the candidates")
    a = ap.parse_args()

    d = pd.read_csv(f"{a.mimic_root}/hosp/d_icd_diagnoses.csv.gz", dtype=str)
    titles = dict(zip(zip(d.icd_version, d.icd_code), d.long_title))
    keep = json.load(open(a.keep)) if a.keep else {}
    dump = defaultdict(dict)
    dicts = {v: set(d[d.icd_version == v].icd_code) for v in ("9", "10")}
    vocab = load_vocab(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model)
    attach_hierarchy(model, tok, vocab)
    dx = {v: [e for e in vocab if e["kind"] == "diagnosis" and e["icd_version"] == v and not e["ancestor"]]
          for v in ("9", "10")}
    with torch.no_grad():
        emb = {v: torch.nn.functional.normalize(model.get_input_embeddings()(torch.tensor(
            [tok.convert_tokens_to_ids(e["token"]) for e in dx[v]])).float(), dim=-1) for v in dx}
    index = {v: {e["code"]: i for i, e in enumerate(dx[v])} for v in dx}

    # counts[(condition, system)] = [tp, n_pred, n_ref]
    counts = defaultdict(lambda: [0, 0, 0])
    for r in load_from_disk(a.criteria):
        for sys_name, ver in (("icd10", "10"), ("icd9", "9")):
            ref = {c.replace(".", "") for c in r["codes"][sys_name]}
            if not ref or r["id"] not in SEEDS:
                continue
            agent = expand(SEEDS[r["id"]].get(ver, ""), dicts[ver])
            rows = [index[ver][c] for c in agent if c in index[ver]]
            neigh = {}
            if rows:
                q = torch.nn.functional.normalize(emb[ver][rows].mean(0), dim=0)
                order = (emb[ver] @ q).argsort(descending=True).tolist()
                for k in a.k:
                    neigh[k] = [dx[ver][i]["code"] for i in order if dx[ver][i]["code"] not in agent][:k]
            conds = {"agent": agent}
            if a.dump and neigh:
                dump[r["id"]][ver] = [[c, titles.get((ver, c), "")] for c in neigh[max(a.k)]]
            for k in a.k:
                if a.keep:
                    conds[f"+filtered@{k}"] = agent | (set(neigh.get(k, [])) & set(keep.get(r["id"], "").split()))
                conds[f"+model@{k}"] = agent | set(neigh.get(k, []))
                conds[f"model_only@{k}"] = set(neigh.get(k, []))
            for cond, pred in conds.items():
                for key in ((cond, sys_name), (cond, "all")):
                    c = counts[key]
                    c[0] += len(pred & ref)
                    c[1] += len(pred)
                    c[2] += len(ref)
    if a.dump:
        json.dump(dump, open(a.dump, "w"))
    print(f"{'condition':16s} {'system':6s}   P      R      F1     (pred / ref)")
    for (cond, system), (tp, npred, nref) in sorted(counts.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        m = prf(tp, npred, nref)
        print(f"{cond:16s} {system:6s} {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}   ({npred} / {nref})")


if __name__ == "__main__":
    main()
