"""Build the trial-criteria eval set from the paper's criteria table (docs/paper/40.docx).

    python -m trialcode.eval.criteria docs/paper/40.docx data/eval/trial_criteria.jsonl

One row per criterion: trial, tier, text, and the reference code set split by system. A model is
evaluated on whether it retrieves/embeds a criterion's codes together (and, with MIMIC, whether
admissions containing any code in the set are recoverable from the remaining context).
"""
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ICD10 = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
ICD9 = re.compile(r"^(V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?|\d{3}(\.\d{1,2})?)$")
ICD10PCS = re.compile(r"^[0-9][0-9A-Z]{6}$")


def read_rows(path):
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    return [["".join(t.text or "" for t in tc.iter(W + "t")) for tc in tr.findall(W + "tc")]
            for tr in root.iter(W + "tr")]


def split_codes(text):
    out = {"icd10": [], "icd9": [], "icd10pcs": [], "rxnorm": []}
    if "RXNORM" in text.upper():
        out["rxnorm"] = re.findall(r"\d{4,}", text.split(":", 1)[-1])
        return out
    for tok in re.split(r"[,;\s]+", re.sub(r"[A-Za-z ]*(diagnosis|procedure|ICD ?-?\d+)[^:]*:", " ", text)):
        tok = tok.strip().rstrip(".")
        if not tok:
            continue
        if ICD10PCS.match(tok) and not tok.isdigit():
            out["icd10pcs"].append(tok)
        elif ICD10.match(tok):
            out["icd10"].append(tok)
        elif ICD9.match(tok):
            out["icd9"].append(tok)
    return {k: sorted(set(v)) for k, v in out.items()}


def main(src, dst):
    rows = read_rows(src)[1:]
    with open(dst, "w") as f:
        for r in rows:
            cid, trial, tier, text, codes = [c.strip() for c in r[:5]]
            f.write(json.dumps({"id": cid, "trial": re.sub(r"\(.*", "", trial).strip(" _"), "tier": int(tier),
                                "criterion": text, "codes": split_codes(codes), "raw_codes": codes}) + "\n")
    print(f"wrote {len(rows)} criteria to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
