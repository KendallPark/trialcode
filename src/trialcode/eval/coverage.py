"""How much of each criterion's reference code set has a token in a trained model's vocabulary.

    python -m trialcode.eval.coverage data/eval/trial_criteria.jsonl outputs/<run>/final/code_vocab.json

Recall over the model's vocabulary is capped at this fraction, so report it next to retrieval recall.
Codes are matched dotless; ancestor (prefix) nodes count as covered only when the exact code has its own token.
"""
import json
import sys
from collections import defaultdict

SYSTEMS = {"icd10": ("diagnosis", "10"), "icd9": ("diagnosis", "9"),
           "icd10pcs": ("procedure", "10"), "icd9pcs": ("procedure", "9")}


def main(criteria, vocab_path):
    vocab = {(v["kind"], v["icd_version"], v["code"]) for v in json.load(open(vocab_path)) if not v["ancestor"]}
    tot = defaultdict(lambda: [0, 0])
    for line in open(criteria):
        r = json.loads(line)
        for system, codes in r["codes"].items():
            if system not in SYSTEMS:
                continue  # rxnorm / loinc have no MIMIC-token equivalent
            kind, ver = SYSTEMS[system]
            hit = sum((kind, ver, c.replace(".", "")) in vocab for c in codes)
            tot[system][0] += hit
            tot[system][1] += len(codes)
    for system, (hit, n) in sorted(tot.items()):
        print(f"{system:9s} {hit:5d}/{n:5d}  {hit / max(n, 1):.1%}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
