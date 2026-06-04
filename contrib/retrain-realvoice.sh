#!/usr/bin/env bash
# retrain-realvoice.sh — rebuild the positive set as real-voice-heavy, then
# retrain the wake model reusing the (expensive) 30GB negative features.
#
# Strategy: the model that ignores real voice was trained on 100% synthetic
# positives. Here we oversample the captured real clips and keep only a
# subset of synthetic for robustness, so real voice dominates the positives
# (the augmentation pipeline then expands each into many variants).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/wintermute-train"
GEN="$ROOT/generated_samples"
OUT="$ROOT/out"
REAL="$HERE/realvoice"
OVERSAMPLE="${OVERSAMPLE_REAL:-4}"   # copies of each real clip before augmentation
SYNTH_KEEP="${SYNTH_KEEP:-600}"      # synthetic clips retained for robustness
log(){ echo "[$(date +%T)] retrain: $*"; }

rc=$(find "$REAL" -name 'jsy_*.wav' 2>/dev/null | wc -l)
[ "$rc" -gt 0 ] || { echo "no real clips in $REAL — run capture-wintermute.sh first"; exit 1; }
log "real clips: $rc | oversample x$OVERSAMPLE -> $((rc*OVERSAMPLE)) real | synth kept: $SYNTH_KEEP"

# Preserve the original all-synthetic positives once.
if [ ! -d "$GEN.synthetic" ]; then
  cp -a "$GEN" "$GEN.synthetic"
  log "backed up synthetic positives -> $GEN.synthetic"
fi

# Rebuild positives: oversampled real + synthetic subset.
rm -rf "$GEN"; mkdir -p "$GEN"
for f in "$REAL"/jsy_*.wav; do
  b=$(basename "$f" .wav)
  for k in $(seq 1 "$OVERSAMPLE"); do cp "$f" "$GEN/real_${b}_r${k}.wav"; done
done
# Deterministic synthetic subset (sorted, take first N) to keep some voice variety.
find "$GEN.synthetic" -name '*.wav' | sort | head -n "$SYNTH_KEEP" | while read -r s; do
  cp "$s" "$GEN/syn_$(basename "$s")"
done
log "new positive set: $(find "$GEN" -name '*.wav' | wc -l) clips ($((rc*OVERSAMPLE)) real + $SYNTH_KEEP synth)"

# Force positive-feature + RIR regen; REUSE the 30GB negative_datasets.
rm -rf "$OUT/generated_augmented_features" "$OUT/mit_rirs"
log "cleared positive features + mit_rirs; keeping $(du -sh "$OUT/negative_datasets" 2>/dev/null | cut -f1) negatives"

log "=== running pipeline from features stage ==="
bash "$HERE/train-wintermute.sh" --stage features
prc=$?
log "=== pipeline exited rc=$prc ==="
if [ -f "$OUT/wintermute.onnx" ]; then
  log "MODEL: $OUT/wintermute.onnx sha=$(sha256sum "$OUT/wintermute.onnx" | cut -d' ' -f1)"
fi
echo "[RETRAIN EXIT $prc]"
exit $prc
