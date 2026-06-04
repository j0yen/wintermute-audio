#!/usr/bin/env bash
# train-wintermute.sh — reproducible micro-wake-word training for the
# "wintermute" wake word, exporting an ONNX model for wm-audio's `ort`
# inference path. PRD-wintermute-wake-word §3.
#
# Stages: deps → positives → augment-data → features → config → train →
#         export → verify. `--smoke` scales sample counts + steps way down
#         to validate the whole pipeline end-to-end fast (PRD AC4).
#
# Usage:
#   ./train-wintermute.sh --smoke            # tiny fast pipeline check
#   ./train-wintermute.sh                    # full run (hours on CPU)
#   ./train-wintermute.sh --stage features   # run from one stage onward
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/wintermute-train"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
WORD="wintermute"
OUT="$ROOT/out"            # spectrogram feature sets + trained model land here
GEN="$ROOT/generated_samples"
VOICE="$HOME/.local/share/wintermute/tts/models/en_US-lessac-medium.onnx"
LIBRITTS_PT="$ROOT/piper-sample-generator/models/en_US-libritts_r-medium.pt"

SMOKE=0; START_STAGE="deps"
while [ $# -gt 0 ]; do case "$1" in
  --smoke) SMOKE=1;;
  --stage) START_STAGE="$2"; shift;;
  -h|--help) sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; shift; done

if [ "$SMOKE" = 1 ]; then
  N_POS=24; TRAIN_STEPS="200"; BATCH=24; OUT="$ROOT/out-smoke"
else
  N_POS=1500; TRAIN_STEPS="5000,5000"; BATCH=128
fi

log(){ echo "[$(date +%T)] $*"; }
stage_idx(){ case "$1" in deps)echo 0;;positives)echo 1;;augment-data)echo 2;;features)echo 3;;config)echo 4;;train)echo 5;;export)echo 6;;verify)echo 7;;*)echo 99;;esac; }
SKIP_TO=$(stage_idx "$START_STAGE")
run_stage(){ [ "$(stage_idx "$1")" -ge "$SKIP_TO" ]; }

mkdir -p "$OUT" "$GEN"

# ── deps ────────────────────────────────────────────────────────────────
if run_stage deps; then
  log "STAGE deps: verifying venv + imports"
  [ -x "$PY" ] || { echo "venv missing at $VENV — run the installer first" >&2; exit 1; }
  # Apply TF-2.21 / NumPy-2.x compatibility fixes to the vendored microwakeword
  # checkout (gitignored, so the patch is the source of truth). Idempotent:
  # skip if already applied. Fixes: result[...].numpy() on plain ndarrays, and
  # np.trapz -> np.trapezoid (removed in NumPy 2.0).
  PATCH="$ROOT/microwakeword-tf221-numpy2-compat.patch"
  if [ -f "$PATCH" ] && [ -d "$ROOT/microWakeWord-src/.git" ]; then
    if git -C "$ROOT/microWakeWord-src" apply --check "$PATCH" 2>/dev/null; then
      git -C "$ROOT/microWakeWord-src" apply "$PATCH" && log "deps: applied microwakeword compat patch"
    else
      log "deps: compat patch already applied (or N/A) — skipping"
    fi
  fi
  "$PY" - <<'PYEOF' || { echo "import check FAILED" >&2; exit 1; }
import tensorflow as tf
import microwakeword
from microwakeword import model_train_eval
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from microwakeword.audio.augmentation import Augmentation
print("deps OK — TF", tf.__version__)
PYEOF
fi

# ── positives: generate "wintermute" utterances ─────────────────────────
if run_stage positives; then
  log "STAGE positives: generating $N_POS '$WORD' samples"
  cd "$ROOT/piper-sample-generator"
  # Newer rhasspy generator: `python -m piper_sample_generator <text> --model <onnx voice>`.
  # Prefer the LibriTTS-R .pt (more voice variety) if present, else the lessac onnx we already ship.
  if [ -f "$LIBRITTS_PT" ]; then
    "$PY" -m piper_sample_generator "$WORD" --model "$LIBRITTS_PT" \
      --max-samples "$N_POS" --batch-size "$BATCH" --output-dir "$GEN"
  else
    "$PY" -m piper_sample_generator "$WORD" --model "$VOICE" \
      --max-samples "$N_POS" --batch-size "$BATCH" --output-dir "$GEN"
  fi
  cd "$ROOT"
  log "positives: $(ls "$GEN"/*.wav 2>/dev/null | wc -l) wav files"
fi

# ── augment-data: RIRs + a small background-noise set ───────────────────
if run_stage augment-data; then
  log "STAGE augment-data: fetching MIT RIRs (+ ambient for FA estimate)"
  "$PY" - <<'PYEOF'
import os, datasets, scipy, numpy as np
os.makedirs("mit_rirs", exist_ok=True)
ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses",
                           split="train", streaming=True)
n=0
for row in ds:
    a=row["audio"]; scipy.io.wavfile.write(f"mit_rirs/{n}.wav", a["sampling_rate"],
        (a["array"]*32767).astype(np.int16)); n+=1
    if n>=(20 if os.environ.get("MWW_SMOKE")=="1" else 200): break
print("RIRs:", n)
PYEOF
fi

# ── negatives: pre-generated spectrogram features from HuggingFace ──────
if run_stage features; then
  log "STAGE features: pulling pre-generated negative features + generating positive features"
  # Negative + ambient feature mmaps live on HF (kahrendt/microwakeword).
  MWW_SMOKE=$SMOKE "$PY" "$HERE/wintermute-train/gen_features.py" \
    --word "$WORD" --positives "$GEN" --out "$OUT" --smoke "$SMOKE" \
    || { echo "feature generation FAILED" >&2; exit 1; }
fi

# ── config: write training_parameters.yaml ──────────────────────────────
if run_stage config; then
  log "STAGE config: writing training_parameters.yaml (steps=$TRAIN_STEPS)"
  MWW_SMOKE=$SMOKE TRAIN_STEPS="$TRAIN_STEPS" OUT="$OUT" \
    "$PY" "$HERE/wintermute-train/write_config.py"
fi

# ── train ───────────────────────────────────────────────────────────────
if run_stage train; then
  log "STAGE train: model_train_eval (this is the long one on CPU)"
  cd "$OUT"
  # Emit BOTH streaming variants (production target) AND the non-streaming
  # variants — export_onnx.py falls back through them if the stateful int8
  # streaming ops refuse to convert via tf2onnx (PRD-flagged top risk).
  "$PY" -m microwakeword.model_train_eval \
    --training_config="$OUT/training_parameters.yaml" \
    --train 1 --restore_checkpoint 1 \
    --test_tf_nonstreaming 0 \
    --test_tflite_nonstreaming 1 \
    --test_tflite_streaming 1 --test_tflite_streaming_quantized 1 \
    mixednet \
      --pointwise_filters "64,64,64,64" \
      --repeat_in_block "1,1,1,1" \
      --mixconv_kernel_sizes "[5],[9],[13],[21]" \
      --residual_connection "0,0,0,0" \
      --first_conv_filters 32 --first_conv_kernel_size 3 \
      --stride 3 \
    || { echo "training FAILED" >&2; exit 1; }
  cd "$ROOT"
fi

# ── export: tflite → onnx ───────────────────────────────────────────────
if run_stage export; then
  log "STAGE export: tflite → onnx"
  "$PY" "$HERE/wintermute-train/export_onnx.py" --out "$OUT" \
    || { echo "onnx export FAILED (PRD-flagged risk: streaming tflite ops)" >&2; exit 1; }
fi

# ── verify: load under onnxruntime, report contract ─────────────────────
if run_stage verify; then
  log "STAGE verify: loading wintermute.onnx under onnxruntime"
  "$PY" - "$OUT/wintermute.onnx" <<'PYEOF'
import sys, onnxruntime as ort, hashlib
p=sys.argv[1]
s=ort.InferenceSession(p, providers=["CPUExecutionProvider"])
print("INPUTS:", [(i.name,i.shape,i.type) for i in s.get_inputs()])
print("OUTPUTS:",[(o.name,o.shape,o.type) for o in s.get_outputs()])
print("sha256:", hashlib.sha256(open(p,'rb').read()).hexdigest())
PYEOF
  log "DONE — model + sha256 above. Pin the sha256 into wintermute-audio/src/models.rs."
fi
