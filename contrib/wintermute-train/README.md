# wintermute-train — custom "wintermute" wake-word training harness

Reproducible [micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)
training for the **"wintermute"** wake word, exporting an ONNX model for
`wm-audio`'s `ort` inference path. Implements PRD-wintermute-wake-word §3.

Training is **offline** and not part of `cargo test`. The `--smoke` path
(AC4) proves the whole pipeline end-to-end with a tiny sample count in
minutes; a full run takes hours on CPU.

## Layout

| File | Role |
|---|---|
| `../train-wintermute.sh` | the driver: `deps → positives → augment-data → features → config → train → export → verify` |
| `install.sh` | create `.venv`, install `tensorflow-cpu` + `pymicro-features` + `microwakeword` |
| `install_torch.sh` | install `torch` (CPU), `piper-tts`, `datasets`, `tf2onnx`, `onnxruntime`, … |
| `prefetch.sh` | clone `piper-sample-generator`, fetch the LibriTTS-R voice + smoke negative features |
| `gen_features.py` | build positive spectrogram features + pull pre-generated negatives (HF) |
| `write_config.py` | emit `training_parameters.yaml` (scales down under `--smoke`) |
| `export_onnx.py` | tflite → onnx, trying each streaming/non-streaming variant best-first |

Everything those scripts download or generate (`.venv/`, cloned sources,
datasets, `out*/`, `*.onnx`, …) is git-ignored — see `.gitignore`. Only the
harness source is tracked.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) and `git` on `$PATH`.

```sh
cd contrib/wintermute-train
./install.sh          # venv + TF + microwakeword
./install_torch.sh    # torch-cpu + piper + tf2onnx + onnxruntime
./prefetch.sh         # clone piper-sample-generator, fetch voice + smoke negatives
```

## Run

```sh
cd contrib
./train-wintermute.sh --help     # documents stages + flags
./train-wintermute.sh --smoke    # tiny end-to-end pipeline check (AC4)
./train-wintermute.sh            # full run (hours on CPU)
./train-wintermute.sh --stage features   # resume from a stage
```

On success the `verify` stage prints the model's true input/output contract
and sha256. **Pin that sha256** (and publish the `.onnx`) into
`wintermute-audio/src/models.rs`'s `wintermute` MANIFEST entry (PRD §1),
which currently carries a documented placeholder.

## Notes

- Positives are synthesized with piper (LibriTTS-R for voice variety, falling
  back to the shipped `en_US-lessac-medium`), then augmented with MIT RIRs and
  background noise — standard micro-wake-word augmentation.
- `export_onnx.py` tries the streaming-quantized model first (production
  target) and falls back through float-streaming and non-streaming variants:
  stateful int8 streaming ops are the PRD-flagged top conversion risk.
- Synthetic-only positives can overfit one voice; real-recording fine-tuning
  is a possible follow-on.
