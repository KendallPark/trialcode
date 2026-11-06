#!/usr/bin/env bash
# Download only the MIMIC files we use. Requires credentialed PhysioNet access (MIMIC-IV and
# MIMIC-IV-Note DUAs signed). Usage: PHYSIONET_USER=you scripts/download_mimic.sh [dest]
# wget prompts for your PhysioNet password.
set -euo pipefail
DEST="${1:-data/mimic}"
IV=https://physionet.org/files/mimiciv/3.1
NOTE=https://physionet.org/files/mimic-iv-note/2.2
mkdir -p "$DEST/hosp" "$DEST/note"
get() { wget -c -N --user "$PHYSIONET_USER" --ask-password -P "$2" "$1"; }
for f in diagnoses_icd procedures_icd d_icd_diagnoses d_icd_procedures admissions; do
  get "$IV/hosp/$f.csv.gz" "$DEST/hosp"
done
get "$NOTE/note/discharge.csv.gz" "$DEST/note"
