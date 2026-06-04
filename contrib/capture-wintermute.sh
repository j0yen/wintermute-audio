#!/usr/bin/env bash
# capture-wintermute.sh — record real-voice "wintermute" utterances for
# wake-word retraining. Each prompt records one ~1.8 s clip; say the word
# ONCE per prompt. Vary distance, tone, and speed a little across the set —
# that variety is what teaches the model to generalise to your real voice.
#
# Usage:  ./capture-wintermute.sh [count] [out_dir] [seconds_per_clip]
#   count   number of utterances to capture (default 120)
# Resumable: re-run to add more; it continues numbering from existing files.
set -uo pipefail

N="${1:-120}"
OUT="${2:-$HOME/wintermute/wake-train/realvoice}"
DUR="${3:-1.8}"
MIC="${WM_MIC_NODE:-alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source}"
PY="$HOME/wintermute/wake-train/wintermute-train/.venv/bin/python"

mkdir -p "$OUT"
rms_of() { "$PY" - "$1" <<'P' 2>/dev/null || echo 0
import wave,sys,numpy as np
w=wave.open(sys.argv[1]); d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)
print(int((d**2).mean()**0.5) if len(d) else 0)
P
}

existing=$(find "$OUT" -name 'jsy_*.wav' 2>/dev/null | wc -l)
echo "=== wintermute real-voice capture ==="
echo "mic:    $MIC"
echo "out:    $OUT  (already have $existing clip(s))"
echo "target: $N new utterances, ${DUR}s each"
echo
echo "Tips: say 'wintermute' ONCE per prompt, clearly. Move around / change"
echo "tone & pace a bit across the set. Quiet clips are auto-rejected & redone."
printf "Press Enter to begin... "; read -r

i=0
while [ "$i" -lt "$N" ]; do
  idx=$(printf "%04d" "$(( existing + i ))")
  f="$OUT/jsy_$idx.wav"
  printf "\n[%d/%d] " "$((i+1))" "$N"
  printf "ready"; for d in 3 2 1; do sleep 0.45; printf " %s" "$d"; done; sleep 0.3
  printf "  >>> SAY \"wintermute\" <<<\n"
  timeout "$DUR" pw-record --target "$MIC" --rate 16000 --channels 1 --format s16 "$f" 2>/dev/null
  rms=$(rms_of "$f")
  if [ "${rms:-0}" -lt 120 ]; then
    printf "    too quiet (rms=%s) — let's redo this one\n" "${rms:-0}"
    rm -f "$f"
    continue
  fi
  printf "    captured (rms=%s)\n" "$rms"
  i=$((i+1))
done

total=$(find "$OUT" -name 'jsy_*.wav' | wc -l)
echo
echo "=== done: $total total clips in $OUT ==="
echo "Next: tell Claude 'captured' and it will mix + retrain."
