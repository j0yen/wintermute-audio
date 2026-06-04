#!/usr/bin/env bash
# Full wake-word training driver — runs in a STABLE dir (~/wintermute/wake-train),
# NOT a /build worktree, so a /build tick can't git-reset it out from under us.
# Sequence: venv install -> torch stack -> prefetch assets -> full train pipeline.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/wintermute-train"
log(){ echo "[$(date +%T)] DRIVER: $*"; }

log "== STEP 1/4: install.sh (venv + tensorflow-cpu + microwakeword) =="
bash "$TRAIN/install.sh"        || { echo "DRIVER: install.sh FAILED"; exit 1; }

log "== STEP 2/4: install_torch.sh (torch/piper/onnx/tensorboard/torchcodec) =="
bash "$TRAIN/install_torch.sh"  || { echo "DRIVER: install_torch.sh FAILED"; exit 1; }

log "== STEP 3/4: prefetch.sh (piper-sample-generator + LibriTTS voice) =="
bash "$TRAIN/prefetch.sh"       || log "prefetch had non-fatal failures (falls back to lessac voice)"

log "== STEP 4/4: train-wintermute.sh (deps -> positives -> augment -> features -> config -> train -> export -> verify) =="
bash "$HERE/train-wintermute.sh"
rc=$?
log "== train pipeline exited rc=$rc =="
if [ -f "$TRAIN/out/wintermute.onnx" ]; then
  log "ARTIFACT: $TRAIN/out/wintermute.onnx ($(du -h "$TRAIN/out/wintermute.onnx" | cut -f1)) sha256=$(sha256sum "$TRAIN/out/wintermute.onnx" | cut -d' ' -f1)"
fi
echo "[DRIVER EXIT $rc]"
exit $rc
