#!/usr/bin/env bash
# Download only the MIMIC files we use. Requires credentialed PhysioNet access (MIMIC-IV and
# MIMIC-IV-Note DUAs signed). Usage: PHYSIONET_USER=you scripts/download_mimic.sh [dest]
# wget prompts for your PhysioNet password.
#   --extra   also fetch the tables for the structured/all variants (DRG, HCPCS, prescriptions, labevents)
#   --no-notes   skip MIMIC-IV-Note (only needed for the "all" variant)
set -euo pipefail
EXTRA=0; NOTES=1; ARGS=()
for a in "$@"; do case "$a" in --extra) EXTRA=1;; --no-notes) NOTES=0;; *) ARGS+=("$a");; esac; done
DEST="${ARGS[0]:-data/mimic}"
IV=https://physionet.org/files/mimiciv/3.1
NOTE=https://physionet.org/files/mimic-iv-note/2.2
mkdir -p "$DEST/hosp" "$DEST/note"
get() { wget -c -N --user "$PHYSIONET_USER" --ask-password -P "$2" "$1"; }
for f in diagnoses_icd procedures_icd d_icd_diagnoses d_icd_procedures admissions; do
  get "$IV/hosp/$f.csv.gz" "$DEST/hosp"
done
if [ "$EXTRA" = 1 ]; then
  for f in drgcodes hcpcsevents d_hcpcs prescriptions labevents d_labitems; do
    get "$IV/hosp/$f.csv.gz" "$DEST/hosp"
  done
fi
if [ "$NOTES" = 1 ]; then get "$NOTE/note/discharge.csv.gz" "$DEST/note"; fi
