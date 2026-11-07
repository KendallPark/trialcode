"""Proof-of-concept agent eval on the full trial-criteria set (all six code systems).

    python -m trialcode.eval.poc_agent data/eval/trial_criteria outputs/<run>/final data/mimic [--k 10 25]

The "agent" (Claude) proposed seed codes from each criterion's text alone, before seeing any reference
codes. Seeds ending in '*' mean "this code and everything under it". Three conditions, scored as
exact-code micro precision/recall/F1 against the expert reference sets:

  agent      the seeds exactly as proposed ('*' expanded through the ICD dictionary, the verifier step)
  +model     agent codes plus, per criterion, the k diagnosis tokens nearest (cosine over the model's
             composed input embeddings) to the mean embedding of the seeds, same ICD version only

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
    "C01": {"icd10": "N18.2", "icd9": "585.2"},
    "C02": {"icd10": "N18.2 N18.30 N18.4", "icd9": "585.2 585.3 585.4"},
    "C03": {"icd10": "E10*", "icd9": "250.01 250.03"},
    "C04": {"icd10": "Q61*", "icd9": "753.1*"},
    "C05": {"icd10": "N17*", "icd9": "584*"},
    "C07": {"icd10": "I10", "icd9": "401*"},
    "C08": {"icd10": "Z95.820 Z98.62 Z95.828"},
    "C09": {"icd10": "E10.2* E10.3* E11.2* E11.3* E11.4* I51.7", "icd9": "250.4* 250.5* 250.6* 429.3"},
    "C10": {"icd10": "Z89.4* Z89.5* Z89.6* S78* S88*"},
    "C11": {"icd10": "I60*", "icd9": "430"},
    "C14": {"icd10": "J44*", "icd9": "491* 492* 496"},
    "C15": {"icd10": "E88.01", "icd9": "273.4"},
    "C16": {"icd10": "I21* I20.0", "icd9": "410* 411.1"},
    "C17": {"icd10": "J00 J01* J02* J03* J04* J06* J20* J21* J22 J18*", "icd9": "460 461* 465* 466* 486"},
    "C19": {"icd10": "K74* K70.3*", "icd9": "571.5 571.2"},
    "C20": {"icd10": "N20* N21*", "icd9": "592* 594*"},
    "C21": {"icd10": "N25.0", "icd9": "588.0"},
    "C23": {"icd10": "I21* I25.2 I50*", "icd9": "410* 414.8 428*"},
    "C24": {"icd10": "E66.01 Z68.4*", "icd9": "278.01"},
    "C25": {"icd10": "I10 I16*", "icd9": "401*"},
    "C27": {"icd10": "E10.1* E11.1* E13.1*", "icd9": "250.1*"},
    "C28": {"icd10": "I82.4* I80.2*", "icd9": "453.4* 451.1*"},
    "C29": {"icd10": "I60* I61* I62*", "icd9": "430 431 432*"},
    "C30": {"icd10": "K92.2 K25* K26* K27* K28*", "icd9": "578* 531* 532* 533* 534*"},
    "C31": {"icd10": "Z95.2"},
    "C33": {"icd10": "F10* F11* F12* F13* F14* F15* F16* F18* F19*", "icd9": "303* 304* 305*"},
    "C34": {"icd10": "L20* J30*", "icd9": "691.8 477*"},
    "C35": {"icd10": "M32* K50* K51*", "icd9": "710.0 555* 556*"},
    "C36": {"icd10": "E21* E05*", "icd9": "252.0* 242*"},
    "C37": {"icd10": "E24* E27*", "icd9": "255*"},
    "C38": {"icd10": "I50*", "icd9": "428*"},
    "C39": {"icd10": "I21* I63*", "icd9": "410* 434*"},
    "C40": {"icd10": "E87.6 E83.42", "icd9": "276.8 275.2"},
}

SEEDS["C06"] = {"rxnorm": "1488564 1488574"}  # RxNorm / LOINC / PCS seeds: from memory, no dictionary to verify against
SEEDS["C08"]["icd9pcs"] = "39.2* 39.5*"
SEEDS["C08"]["icd10pcs"] = "041* 047*"
SEEDS["C10"]["icd9pcs"] = "84.1*"
SEEDS["C10"]["icd10pcs"] = "0Y6*"
SEEDS["C12"] = {"rxnorm": "73494 205304"}
SEEDS["C13"] = {"rxnorm": "35296 198188"}
SEEDS["C18"] = {"rxnorm": "1114874 1535600 1424884"}
SEEDS["C20"]["icd9pcs"] = "55.04 56.0* 98.5*"
SEEDS["C20"]["icd10pcs"] = "0TC* 0TF*"
SEEDS["C22"] = {"loinc": "4548-4 17856-6 4549-2 41995-2 59261-8 55454-3 71875-9 62388-4 17855-8"}
SEEDS["C26"] = {"rxnorm": "475968 6809"}
SEEDS["C30"]["icd9pcs"] = "44.43"
SEEDS["C30"]["icd10pcs"] = "0W3P8ZZ"
SEEDS["C31"]["icd9pcs"] = "35.2*"
SEEDS["C32"] = {"rxnorm": "1364430"}

SYSTEMS = {"icd10": ("diagnosis", "10"), "icd9": ("diagnosis", "9"),
           "icd10pcs": ("procedure", "10"), "icd9pcs": ("procedure", "9"), "rxnorm": None, "loinc": None}


def expand(seeds, dictionary):
    if dictionary is None:  # no dictionary (RxNorm, LOINC): the agent's codes stand as proposed
        return set(seeds.split())
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

    kinds = {"diagnosis": "d_icd_diagnoses", "procedure": "d_icd_procedures"}
    titles, dicts = {}, {}
    for kind, f in kinds.items():
        d = pd.read_csv(f"{a.mimic_root}/hosp/{f}.csv.gz", dtype=str)
        titles.update(zip(zip([kind] * len(d), d.icd_version, d.icd_code), d.long_title))
        for v in ("9", "10"):
            dicts[(kind, v)] = set(d[d.icd_version == v].icd_code)
    keep = json.load(open(a.keep)) if a.keep else {}
    dump = defaultdict(dict)
    vocab = load_vocab(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model)
    attach_hierarchy(model, tok, vocab)
    toks = {kv: [e for e in vocab if (e["kind"], e["icd_version"]) == kv and not e["ancestor"]] for kv in dicts}
    with torch.no_grad():
        emb = {kv: torch.nn.functional.normalize(model.get_input_embeddings()(torch.tensor(
            [tok.convert_tokens_to_ids(e["token"]) for e in toks[kv]])).float(), dim=-1) for kv in toks}
    index = {kv: {e["code"]: i for i, e in enumerate(toks[kv])} for kv in toks}

    counts = defaultdict(lambda: [0, 0, 0])  # (condition, system) -> [tp, n_pred, n_ref]
    for r in load_from_disk(a.criteria):
        for system, kv in SYSTEMS.items():
            norm = (lambda c: c.replace(".", "")) if kv else (lambda c: c)
            ref = {norm(c) for c in r["codes"][system]}
            if not ref:
                continue
            agent = expand(SEEDS.get(r["id"], {}).get(system, ""), dicts[kv] if kv else None)
            neigh = {}
            rows = [index[kv][c] for c in agent if kv and c in index[kv]]
            if rows:  # systems/seeds without model tokens fall back to the agent alone
                q = torch.nn.functional.normalize(emb[kv][rows].mean(0), dim=0)
                order = (emb[kv] @ q).argsort(descending=True).tolist()
                for k in a.k:
                    neigh[k] = [toks[kv][i]["code"] for i in order if toks[kv][i]["code"] not in agent][:k]
                if a.dump:
                    dump[r["id"]][system] = [[c, titles.get((kv[0], kv[1], c), "")] for c in neigh[max(a.k)]]
            conds = {"agent": agent}
            for k in a.k:
                n = set(neigh.get(k, []))
                conds[f"+model@{k}"] = agent | n
                if a.keep:
                    conds[f"+filtered@{k}"] = agent | (n & set(keep.get(r["id"], "").split()))
            for cond, pred in conds.items():
                for key in ((cond, system), (cond, "all")):
                    c = counts[key]
                    c[0] += len(pred & ref)
                    c[1] += len(pred)
                    c[2] += len(ref)
    if a.dump:
        json.dump(dump, open(a.dump, "w"))
    print(f"{'condition':16s} {'system':8s}   P      R      F1     (pred / ref)")
    for (cond, system), (tp, npred, nref) in sorted(counts.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        m = prf(tp, npred, nref)
        print(f"{cond:16s} {system:8s} {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}   ({npred} / {nref})")


if __name__ == "__main__":
    main()
