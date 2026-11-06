#!/usr/bin/env bash
# Waits for (a) scripts/overnight2.sh to exit and (b) the --extra MIMIC download to finish and verify,
# then runs structured (1 epoch) + probes and all (1 epoch) + probes. A failure in one does not stop the next.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs results
stamp() { echo "[$(date '+%F %T')] $*"; }
NEED="drgcodes hcpcsevents d_hcpcs prescriptions labevents d_labitems"

while pgrep -f "scripts/overnight2.sh" > /dev/null; do sleep 30; done
stamp "overnight2 done; waiting for downloads"
while true; do
  ok=1
  pgrep -f "wget.*physionet" > /dev/null && ok=0
  for f in $NEED; do gzip -t "data/mimic/hosp/$f.csv.gz" 2>/dev/null || ok=0; done
  [ "$ok" = 1 ] && break
  sleep 60
done
stamp "downloads verified"

run_variant() {  # $1 = variant
  local name="cpt-$1-modernbert-e1"
  stamp "train $name"
  $PY -m trialcode.pretrain --mimic-root data/mimic --variant "$1" --epochs 1 --batch-size 64 --grad-accum 1 \
      --run-name "$name" --output-dir "outputs/$name" > "logs/$name.log" 2>&1 || { stamp "TRAIN FAILED $name"; return 1; }
  stamp "probes $name"
  $PY -m trialcode.eval.probes --model "outputs/$name/final" --mimic-root data/mimic --variant "$1" \
      --baseline-model answerdotai/ModernBERT-base --max-admissions 5000 --out "results/probes-$name.json" \
      > "logs/probes-$name.log" 2>&1 || stamp "PROBES FAILED $name"
}
run_variant structured
run_variant all

git add results && git commit -qm "Add structured/all probe results

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin main
stamp "done"
