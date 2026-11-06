"""MIMIC-IV (+ MIMIC-IV-Note) as a Hugging Face dataset for continued pretraining.

Expected layout under `root` (as downloaded by scripts/download_mimic.sh):
    hosp/{admissions,diagnoses_icd,procedures_icd,d_icd_diagnoses,d_icd_procedures}.csv.gz
    hosp/{drgcodes,hcpcsevents,d_hcpcs,prescriptions,labevents,d_labitems}.csv.gz   (structured/all)
    note/discharge.csv.gz                                                           (all)

Variants control how much of the record the model sees:
    codes       diagnosis + procedure codes
    structured  codes + DRG, HCPCS/CPT, prescribed drugs, abnormal labs (no free text)
    all         discharge note + everything in `structured`

A "code" is a dict {kind, icd_version, code, title}; `icd_version` is the code-system version
("9"/"10" for ICD, the DRG type for DRGs, "0" where there is none).
"""
import os

import pandas as pd
from datasets import Dataset

VARIANTS = ("all", "structured", "codes")
KINDS = ("diagnosis", "procedure", "drg", "hcpcs", "drug", "lab")  # order within an admission
_ICD_KINDS = KINDS[:2]
VARIANT_KINDS = {"codes": _ICD_KINDS, "structured": KINDS, "all": KINDS}
_PREFIX = {"diagnosis": "DX", "procedure": "PX", "drg": "DRG", "hcpcs": "HCPCS", "drug": "RX", "lab": "LAB"}


def _read(root, rel, **kw):
    return pd.read_csv(os.path.join(root, rel), dtype=str, **kw)


def _frame(df, kind, hadm_id, code, title, version="0", seq=None):
    out = pd.DataFrame({
        "hadm_id": df[hadm_id], "kind": kind, "icd_code": df[code], "icd_version": df[version] if version in df else version,
        "long_title": df[title] if title in df else title, "seq_num": df[seq].astype(int) if seq else 0,
    })
    return out.dropna(subset=["hadm_id", "icd_code"])


def _load_kind(root, kind):
    if kind in _ICD_KINDS:
        t, d = (("diagnoses_icd", "d_icd_diagnoses") if kind == "diagnosis" else ("procedures_icd", "d_icd_procedures"))
        ev = _read(root, f"hosp/{t}.csv.gz", usecols=["hadm_id", "seq_num", "icd_code", "icd_version"])
        titles = _read(root, f"hosp/{d}.csv.gz", usecols=["icd_code", "icd_version", "long_title"])
        df = ev.merge(titles, on=["icd_code", "icd_version"], how="left")
        return _frame(df, kind, "hadm_id", "icd_code", "long_title", "icd_version", "seq_num")
    if kind == "drg":
        df = _read(root, "hosp/drgcodes.csv.gz", usecols=["hadm_id", "drg_type", "drg_code", "description"])
        return _frame(df, kind, "hadm_id", "drg_code", "description", "drg_type")
    if kind == "hcpcs":
        ev = _read(root, "hosp/hcpcsevents.csv.gz", usecols=["hadm_id", "hcpcs_cd", "short_description"])
        d = _read(root, "hosp/d_hcpcs.csv.gz", usecols=["code", "long_description"]).rename(columns={"code": "hcpcs_cd"})
        df = ev.merge(d, on="hcpcs_cd", how="left")
        df["title"] = df["long_description"].fillna(df["short_description"])
        return _frame(df, kind, "hadm_id", "hcpcs_cd", "title").drop_duplicates(["hadm_id", "icd_code"])
    if kind == "drug":
        df = _read(root, "hosp/prescriptions.csv.gz", usecols=["hadm_id", "drug"]).dropna()
        df["drug"] = df["drug"].str.lower().str.strip()
        return _frame(df, kind, "hadm_id", "drug", "drug").drop_duplicates(["hadm_id", "icd_code"])
    if kind == "lab":  # abnormal results only; chunked because labevents is ~150M rows
        parts = []
        for ch in pd.read_csv(os.path.join(root, "hosp/labevents.csv.gz"), dtype=str,
                              usecols=["hadm_id", "itemid", "flag"], chunksize=5_000_000):
            ch = ch[ch["flag"].eq("abnormal") & ch["hadm_id"].notna()]
            parts.append(ch[["hadm_id", "itemid"]].drop_duplicates())
        ev = pd.concat(parts).drop_duplicates()
        d = _read(root, "hosp/d_labitems.csv.gz", usecols=["itemid", "label", "fluid"])
        df = ev.merge(d, on="itemid", how="left")
        df["title"] = "abnormal " + df["label"].fillna("") + " (" + df["fluid"].fillna("") + ")"
        return _frame(df, kind, "hadm_id", "itemid", "title")
    raise ValueError(kind)


def load_admission_codes(root, kinds=KINDS):
    """One row per (hadm_id, kind, code) with version/title, ordered by kind then seq_num."""
    df = pd.concat([_load_kind(root, k) for k in kinds])
    df["kind"] = pd.Categorical(df["kind"], categories=KINDS, ordered=True)
    return df.sort_values(["hadm_id", "kind", "seq_num"], kind="stable")


def code_token(kind, icd_version, code):
    """Single-token spelling of a code: [DX10:E119], [PX9:3893], [DRGAPR:690], [RX:metformin], [LAB:50912]."""
    ver = str(icd_version) if kind in (*_ICD_KINDS, "drg") else ""
    return f"[{_PREFIX[kind]}{ver}:{str(code).replace(' ', '_')}]"


def code_vocabulary(root, min_count=5, kinds=KINDS):
    """Codes seen in at least `min_count` admissions, with titles, as the code-token vocabulary."""
    codes = load_admission_codes(root, kinds)
    codes["title"] = codes["long_title"].fillna("")
    codes["kind"] = codes["kind"].astype(str)
    g = codes.groupby(["kind", "icd_version", "icd_code"], sort=True).agg(
        count=("hadm_id", "nunique"), title=("title", "first")).reset_index()
    g = g[g["count"] >= min_count]
    vocab = {}
    for r in g.itertuples():
        tok = code_token(r.kind, r.icd_version, r.icd_code)
        vocab.setdefault(tok, {"token": tok, "kind": r.kind, "icd_version": r.icd_version,
                               "code": r.icd_code, "title": r.title, "count": int(r.count)})
    return list(vocab.values())


def _generate(root, variant):
    codes = load_admission_codes(root, VARIANT_KINDS[variant])
    codes["kind"] = codes["kind"].astype(str)
    codes["long_title"] = codes["long_title"].fillna("")
    by_hadm = {h: g for h, g in codes.groupby("hadm_id", sort=False)}

    def code_list(hadm_id):
        g = by_hadm.get(hadm_id)
        if g is None:
            return []
        return [{"kind": r.kind, "icd_version": str(r.icd_version), "code": r.icd_code, "title": r.long_title}
                for r in g.itertuples()]

    if variant == "all":
        notes = _read(root, "note/discharge.csv.gz", usecols=["note_id", "subject_id", "hadm_id", "text"])
        for r in notes.itertuples():
            yield {"note_id": r.note_id, "subject_id": r.subject_id, "hadm_id": r.hadm_id,
                   "text": r.text, "codes": code_list(r.hadm_id)}
    else:
        adm = _read(root, "hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id"])
        for r in adm.itertuples():
            codes_ = code_list(r.hadm_id)
            if codes_:
                yield {"note_id": "", "subject_id": r.subject_id, "hadm_id": r.hadm_id,
                       "text": "", "codes": codes_}


def load_mimic(root, variant="all", **kwargs):
    """Raw per-admission dataset (text + structured code list). Cached by HF datasets."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return Dataset.from_generator(_generate, gen_kwargs={"root": root, "variant": variant}, **kwargs)


def split_by_subject(ds, val_pct=1, test_pct=1):
    """Deterministic patient-level split so no subject spans train/val/test."""
    def bucket(ex):
        return int(ex["subject_id"]) % 100
    return {
        "train": ds.filter(lambda ex: bucket(ex) >= val_pct + test_pct),
        "validation": ds.filter(lambda ex: bucket(ex) < val_pct),
        "test": ds.filter(lambda ex: val_pct <= bucket(ex) < val_pct + test_pct),
    }
