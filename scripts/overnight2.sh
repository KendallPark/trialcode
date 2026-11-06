#!/usr/bin/env bash
# Runs after scripts/overnight.sh finishes (waits on its PID): a longer 4-epoch ModernBERT codes run + probes.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
stamp() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "scripts/overnight.sh" > /dev/null; do sleep 30; done
name=cpt-codes-modernbert-e4
stamp "train $name"
$PY -m trialcode.pretrain --mimic-root data/mimic --variant codes --epochs 4 --batch-size 64 --grad-accum 1 \
    --run-name "$name" --output-dir "outputs/$name" > "logs/$name.log" 2>&1 || { stamp "TRAIN FAILED $name"; exit 1; }
stamp "probes $name"
$PY -m trialcode.eval.probes --model "outputs/$name/final" --mimic-root data/mimic --variant codes \
    --baseline-model answerdotai/ModernBERT-base --max-admissions 5000 --out "results/probes-$name.json" \
    > "logs/probes-$name.log" 2>&1 || stamp "PROBES FAILED $name"
git add results && git commit -qm "Add 4-epoch probe results

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin main
stamp "done"
