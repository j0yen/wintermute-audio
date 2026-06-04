#!/usr/bin/env python3
"""Convert a trained micro-wake-word TFLite model to ONNX.

PRD-flagged risk: the streaming model carries internal ring-buffer state and
consumes a small feature slice (NOT raw PCM). This script tries every TFLite
variant microwakeword produced — quantized streaming, float streaming, then the
non-streaming fallbacks — and reports which one actually converts to a loadable
ONNX plus its TRUE input/output contract, so the wm-audio inference path (MFCC
front-end + state handling) can be matched to it. It does NOT assume the model
fits the legacy [1,1280]->scalar contract.
"""
import argparse, glob, hashlib, os, subprocess, sys
from pathlib import Path


def candidate_tflites(out):
    """Ordered list of (label, path) variants to attempt, best-first.

    Streaming-quantized is the production target; if its stateful int8 ops do
    not convert, we fall back through float-streaming and the non-streaming
    variants (which have no internal state and convert most reliably).
    """
    base = out / "trained_models" / "wakeword"
    ordered = [
        ("stream_quant",
         base / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"),
        ("stream_float",
         base / "tflite_stream_state_internal" / "stream_state_internal.tflite"),
        ("nonstream_quant",
         base / "tflite_non_stream_quant" / "non_stream_quant.tflite"),
        ("nonstream_float",
         base / "tflite_non_stream" / "non_stream.tflite"),
    ]
    found = [(lbl, p) for lbl, p in ordered if p.exists()]
    if not found:
        # last-ditch: any tflite under the train dir
        for p in sorted(glob.glob(str(base / "**" / "*.tflite"), recursive=True)):
            found.append(("found:" + Path(p).stem, Path(p)))
    return found


def try_convert(tfl, onnx_path):
    cmd = [sys.executable, "-m", "tf2onnx.convert",
           "--tflite", str(tfl), "--output", str(onnx_path), "--opset", "17"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def report_contract(onnx_path):
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape, i.type) for i in s.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in s.get_outputs()]
    return ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    cands = candidate_tflites(out)
    if not cands:
        print("[export_onnx] ERROR: no .tflite found under", out / "trained_models")
        sys.exit(1)

    print("[export_onnx] candidate tflite variants (best-first):")
    for lbl, p in cands:
        print("   ", lbl, "->", p, f"({p.stat().st_size} bytes)")

    onnx_path = out / "wintermute.onnx"
    last_err = ""
    for lbl, tfl in cands:
        print(f"\n[export_onnx] === attempting variant '{lbl}': {tfl}")
        rc, so, se = try_convert(tfl, onnx_path)
        tail = (so[-1500:] + se[-1500:]).strip()
        if rc != 0 or not onnx_path.exists():
            print(f"[export_onnx] variant '{lbl}' tf2onnx FAILED (rc={rc})")
            print(tail)
            last_err = f"{lbl}: {se[-600:]}"
            if onnx_path.exists():
                onnx_path.unlink()
            continue
        # Loadable under onnxruntime?
        try:
            ins, outs = report_contract(onnx_path)
        except Exception as e:  # noqa: BLE001
            print(f"[export_onnx] variant '{lbl}' converted but onnxruntime load FAILED: {e}")
            last_err = f"{lbl}: ort load: {e}"
            onnx_path.unlink()
            continue

        sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        print(f"\n[export_onnx] SUCCESS with variant '{lbl}'")
        print("[export_onnx] INPUTS:", ins)
        print("[export_onnx] OUTPUTS:", outs)
        print("[export_onnx] OK ->", onnx_path)
        print("[export_onnx] sha256:", sha)
        print("[export_onnx] size:", onnx_path.stat().st_size, "bytes")
        (out / "wintermute.onnx.sha256").write_text(sha + "\n")
        (out / "wintermute.onnx.variant").write_text(lbl + "\n")
        return

    print("\n[export_onnx] tflite->onnx FAILED for ALL variants.")
    print("[export_onnx] last error:", last_err)
    sys.exit(1)


if __name__ == "__main__":
    main()
