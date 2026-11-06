import pandas as pd
import pytest


@pytest.fixture
def mimic_root(tmp_path):
    """Tiny MIMIC-shaped directory: 20 patients, ICD-10 dx/px, DRG, HCPCS, drugs, abnormal labs, notes."""
    hosp, note = tmp_path / "hosp", tmp_path / "note"
    hosp.mkdir(), note.mkdir()
    ids = range(1, 21)
    w = lambda df, p: df.to_csv(p, index=False)
    w(pd.DataFrame({"icd_code": ["E119", "E1165", "I10", "0BH17EZ"], "icd_version": ["10"] * 4,
                    "long_title": ["Type 2 diabetes without complications", "Type 2 diabetes with hyperglycemia",
                                   "Essential hypertension", "Insertion of endotracheal airway"]}).iloc[:3],
      hosp / "d_icd_diagnoses.csv.gz")
    w(pd.DataFrame({"icd_code": ["0BH17EZ"], "icd_version": ["10"], "long_title": ["Insertion of endotracheal airway"]}),
      hosp / "d_icd_procedures.csv.gz")
    dx = [(str(s), str(100 + s), "1", c, "10") for s in ids for c in (["E119", "I10"] if s % 2 else ["E1165", "I10"])]
    w(pd.DataFrame(dx, columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]), hosp / "diagnoses_icd.csv.gz")
    h = [str(100 + s) for s in ids]
    w(pd.DataFrame({"subject_id": [str(s) for s in ids], "hadm_id": h, "seq_num": "1", "icd_code": "0BH17EZ",
                    "icd_version": "10"}), hosp / "procedures_icd.csv.gz")
    w(pd.DataFrame({"subject_id": [str(s) for s in ids], "hadm_id": h}), hosp / "admissions.csv.gz")
    w(pd.DataFrame({"hadm_id": h, "drg_type": "APR", "drg_code": "690", "description": "Kidney infection"}), hosp / "drgcodes.csv.gz")
    w(pd.DataFrame({"hadm_id": h, "hcpcs_cd": "99213", "short_description": "Office visit"}), hosp / "hcpcsevents.csv.gz")
    w(pd.DataFrame({"code": ["99213"], "long_description": ["Office visit, established patient"]}), hosp / "d_hcpcs.csv.gz")
    w(pd.DataFrame({"hadm_id": h + h, "drug": ["Metformin"] * 20 + ["Insulin"] * 20}), hosp / "prescriptions.csv.gz")
    w(pd.DataFrame({"hadm_id": h, "itemid": "50852", "flag": "abnormal"}), hosp / "labevents.csv.gz")
    w(pd.DataFrame({"itemid": ["50852"], "label": ["Hemoglobin A1c"], "fluid": ["Blood"]}), hosp / "d_labitems.csv.gz")
    w(pd.DataFrame({"note_id": [f"n{s}" for s in ids], "subject_id": [str(s) for s in ids], "hadm_id": h,
                    "text": "Patient with diabetes. " * 30}), note / "discharge.csv.gz")
    return str(tmp_path)
