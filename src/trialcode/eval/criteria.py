"""Build the trial-criteria eval set from the paper's criteria table (docs/paper/40.docx).

    python -m trialcode.eval.criteria docs/paper/40.docx data/eval/trial_criteria [--push-to-hub org/name]

Saves a HuggingFace dataset (load_from_disk) plus a jsonl copy alongside it.

One row per criterion: trial, tier, text, and the reference code set split by system. A model is
evaluated on whether it retrieves/embeds a criterion's codes together (and, with MIMIC, whether
admissions containing any code in the set are recoverable from the remaining context).
"""
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

from datasets import Dataset, Features, Sequence, Value

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ICD10 = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
ICD9 = re.compile(r"^(V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?|\d{3}(\.\d{1,2})?)$")
ICD10PCS = re.compile(r"^[0-9][0-9A-Z]{6}$")
ICD9PCS = re.compile(r"^\d{2}\.\d{1,2}$")
LOINC = re.compile(r"^\d{1,5}-\d$")


def read_rows(path):
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    return [["".join(t.text or "" for t in tc.iter(W + "t")) for tc in tr.findall(W + "tc")]
            for tr in root.iter(W + "tr")]


def split_codes(text):
    out = {"icd10": [], "icd9": [], "icd9pcs": [], "icd10pcs": [], "rxnorm": [], "loinc": []}
    if "RXNORM" in text.upper():
        out["rxnorm"] = re.findall(r"\d{4,}", text.split(":", 1)[-1])
        return out
    if "LOINC" in text.upper():
        out["loinc"] = sorted(set(re.findall(r"\d{1,5}-\d", text)))
        return out
    for tok in re.split(r"[,;\s]+", re.sub(r"[A-Za-z ]*(diagnosis|procedure|ICD ?-?\d+)[^:]*:", " ", text)):
        tok = tok.strip().rstrip(".")
        if not tok:
            continue
        if ICD10PCS.match(tok) and not tok.isdigit():
            out["icd10pcs"].append(tok)
        elif ICD9PCS.match(tok):
            out["icd9pcs"].append(tok)
        elif ICD10.match(tok):
            out["icd10"].append(tok)
        elif ICD9.match(tok):
            out["icd9"].append(tok)
    return {k: sorted(set(v)) for k, v in out.items()}


SYSTEMS = ["icd10", "icd9", "icd9pcs", "icd10pcs", "rxnorm", "loinc"]
FEATURES = Features({"id": Value("string"), "trial": Value("string"), "tier": Value("int32"),
                     "criterion": Value("string"), "raw_codes": Value("string"),
                     "codes": {k: Sequence(Value("string")) for k in SYSTEMS}})


def build(src):
    recs = []
    for r in read_rows(src)[1:]:
        cid, trial, tier, text, codes = [c.strip() for c in r[:5]]
        recs.append({"id": cid, "trial": re.sub(r"\(.*", "", trial).strip(" _"), "tier": int(tier),
                     "criterion": text, "raw_codes": codes, "codes": split_codes(codes)})
    return Dataset.from_list(recs, features=FEATURES)


def main(src, dst, hub=None):
    ds = build(src)
    ds.save_to_disk(dst)
    ds.to_json(dst + ".jsonl")
    print(f"wrote {len(ds)} criteria to {dst}")
    if hub:
        ds.push_to_hub(hub, private=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], args[3] if len(args) > 3 and args[2] == "--push-to-hub" else None)
