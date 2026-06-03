#!/usr/bin/env python3
"""Regenerate (or verify) the AC2 golden mel vector from the reference frontend.

The golden at ``tests/golden/mel_440hz_8000amp.json`` is the ground-truth
log-mel feature matrix the wake model was trained against. It is produced by
the **reference** TFLM micro-frontend shipped as the ``pymicro_features``
package (OHF-Voice/micro-wake-word), NOT by our Rust ``features::mel_window``.
That makes it an independent oracle: AC2 asserts the Rust port reproduces this
vector to <=1e-3, so the golden must come from the reference impl, never from
the code under test.

Provenance / scaling
--------------------
``MicroFrontend.process_samples`` consumes 160-sample (10 ms) hops and emits a
40-bin feature frame once warmed up. In this package the emitted ``features``
are already **floats** (the C frontend's uint16 spectrogram divided down). The
model is fed ``feature * 0.0390625`` (= 1/25.6); see microwakeword
``data.py``/``inference.py`` (``spectrogram.astype(np.float32) * 0.0390625``).
So the golden value is ``reference_float_feature * 0.0390625``.

IMPORTANT — do not "validate" the golden by checking that every value is an
integer multiple of 0.0390625. It is NOT: the reference features are floats,
so float*0.0390625 carries the float's fractional part. A 2026-06 build tick
made exactly that wrong assumption and spent six ticks chasing a non-bug. This
script is the authoritative provenance: run ``--verify`` and trust a bit-exact
match over any divisibility heuristic.

Usage
-----
    python contrib/gen_golden_mel.py --verify   # assert committed golden matches
    python contrib/gen_golden_mel.py --write     # (re)write the golden file

Requires ``pymicro_features`` (``pip install pymicro-features``); stdlib only
otherwise (the sine uses truncate-toward-zero to match ``np.astype(int16)``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "golden", "mel_440hz_8000amp.json",
)

SAMPLE_RATE = 16000
N_SAMPLES = 30240          # (186 frames + 3 warmup) * 160-sample hop
FREQ_HZ = 440.0
AMPLITUDE = 8000
NUM_MEL_BINS = 40
NUM_FRAMES = 186           # MEL_WINDOW geometry: 30240/160 - 3 = 186
FEATURE_SCALE = 0.0390625  # 1 / 25.6
HOP_BYTES = 160 * 2        # 160 i16 samples per process_samples call


def _sine_pcm() -> bytes:
    """440 Hz sine, amplitude 8000, N_SAMPLES i16 @ 16 kHz, little-endian."""
    out = bytearray()
    two_pi_f = 2.0 * math.pi * FREQ_HZ
    for n in range(N_SAMPLES):
        v = int(AMPLITUDE * math.sin(two_pi_f * n / SAMPLE_RATE))  # trunc toward 0 == np.astype(int16)
        v = max(-32768, min(32767, v))
        out += int(v & 0xFFFF).to_bytes(2, "little")
    return bytes(out)


def _features() -> list[list[float]]:
    from pymicro_features import MicroFrontend  # type: ignore

    mf = MicroFrontend()
    mf.reset()
    audio = _sine_pcm()
    frames: list[list[float]] = []
    i = 0
    while i + HOP_BYTES <= len(audio):
        out = mf.process_samples(audio[i:i + HOP_BYTES])
        i += HOP_BYTES
        if out.features:
            frames.append([f * FEATURE_SCALE for f in out.features])
    # The reference emits one extra trailing frame; the trained window is the
    # first NUM_FRAMES. Cap to the documented geometry.
    return frames[:NUM_FRAMES]


def _payload(feats: list[list[float]]) -> dict:
    return {
        "description": (
            "Golden TFLM-microfrontend log-mel features for AC2 parity. Input "
            "is a 440 Hz sine at amplitude 8000, 30240 i16 samples @16kHz. "
            "Features are the pymicro_features MicroFrontend float output "
            "multiplied by 0.0390625 (= 1/25.6), matching "
            "microwakeword.data/inference scaling. dtype f32, shape [186,40]."
        ),
        "sample_rate": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
        "freq_hz": FREQ_HZ,
        "amplitude": AMPLITUDE,
        "window_step_ms": 10,
        "window_size_ms": 30,
        "num_mel_bins": NUM_MEL_BINS,
        "num_frames": NUM_FRAMES,
        "feature_scale": FEATURE_SCALE,
        "source": (
            "OHF-Voice/micro-wake-word pymicro_features.MicroFrontend "
            "(process_samples streaming, float features * 0.0390625); "
            "regenerate with contrib/gen_golden_mel.py"
        ),
        "features": feats,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="(re)write the golden")
    g.add_argument("--verify", action="store_true",
                   help="assert committed golden matches the reference export")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)

    feats = _features()
    if len(feats) != NUM_FRAMES or any(len(r) != NUM_MEL_BINS for r in feats):
        print(f"ERROR: shape is {len(feats)}x"
              f"{len(feats[0]) if feats else 0}, want "
              f"{NUM_FRAMES}x{NUM_MEL_BINS}", file=sys.stderr)
        return 2

    if args.write:
        with open(GOLDEN_PATH, "w") as fh:
            json.dump(_payload(feats), fh)
            fh.write("\n")
        print(f"wrote {GOLDEN_PATH} ({NUM_FRAMES}x{NUM_MEL_BINS})")
        return 0

    # verify
    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)["features"]
    maxabs = 0.0
    for r in range(NUM_FRAMES):
        for c in range(NUM_MEL_BINS):
            maxabs = max(maxabs, abs(feats[r][c] - golden[r][c]))
    if maxabs <= args.tol:
        print(f"OK golden matches reference export: maxabs={maxabs:.6g} "
              f"<= tol={args.tol}")
        return 0
    print(f"MISMATCH maxabs={maxabs:.6g} > tol={args.tol}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
