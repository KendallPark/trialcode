#!/usr/bin/env bash
# Sequential overnight chain on the single GPU:
#   1. codes variant, ModernBERT-base, 1 epoch (W&B on) -> probes
#   2. preflight Ettin-150m (3 steps); if it works, same run on Ettin -> probes
# Each step logs to logs/ and a failure does not stop the chain. Results land in results/.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs results
COMMON="--mimic-root data/mimic --variant codes --epochs 1 --batch-size 64 --grad-accum 1"
stamp() { echo "[$(date '+%F %T')] $*"; }

run_model() {  # $1 = HF model id, $2 = short name
  local name="cpt-codes-$2-e1"
  stamp "train $name"
  $PY -m trialcode.pretrain $COMMON --model "$1" --run-name "$name" --output-dir "outputs/$name" \
      > "logs/$name.log" 2>&1 || { stamp "TRAIN FAILED $name"; return 1; }
  stamp "probes $name"
  $PY -m trialcode.eval.probes --model "outputs/$name/final" --mimic-root data/mimic --variant codes \
      --baseline-model "$1" --max-admissions 5000 --out "results/probes-$name.json" \
      > "logs/probes-$name.log" 2>&1 || stamp "PROBES FAILED $name"
}

run_model answerdotai/ModernBERT-base modernbert

stamp "preflight ettin"
if $PY -m trialcode.pretrain $COMMON --model jhu-clsp/ettin-encoder-150m --max-train-examples 500 --max-steps 3 \
     --batch-size 8 --no-wandb --output-dir outputs/preflight-ettin > logs/preflight-ettin.log 2>&1; then
  run_model jhu-clsp/ettin-encoder-150m ettin
else
  stamp "ETTIN PREFLIGHT FAILED (see logs/preflight-ettin.log)"
fi

stamp "committing results"
git add results && git commit -qm "Add overnight probe results

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin main
stamp "done"
