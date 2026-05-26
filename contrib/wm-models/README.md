# wm-models — pretrained ONNX bundle

AUR-style package that installs the three pretrained ONNX/ggml assets
wintermute-audio + wintermute-stt need at runtime, into
`/usr/share/wintermute/models/`.

Contents:

| File | Source | Used by |
|---|---|---|
| `hey_jarvis.onnx` | esphome/micro-wake-word-models v2 | `wm-audio` wake |
| `okay_nabu.onnx` | esphome/micro-wake-word-models v2 | `wm-audio` wake |
| `hey_mycroft.onnx` | esphome/micro-wake-word-models v2 | `wm-audio` wake |
| `silero_vad.onnx` | snakers4/silero-vad v5.1.2 | `wm-audio` VAD |
| `ggml-base.en.bin` | ggerganov/whisper.cpp v1.7.5 | `wm-stt` |

## Build & install

```sh
cd contrib/wm-models
./update-hashes.sh         # one-time: fill real sha256sums
makepkg -si --noconfirm
```

The `update-hashes.sh` helper fetches each source URL and runs
`updpkgsums` (from `pacman-contrib`) to replace the `SKIP` placeholders
in `PKGBUILD` with real hashes. The shipped PKGBUILD intentionally uses
`SKIP` so the file parses anywhere and so this scaffold can land without
network access at build-time.

## Why this exists

PRD-wintermute-audio §2.5 requires hash-pinned model distribution and
forbids first-boot downloads inside the daemon. `wm-models` solves both:
the pacman package is the single source of truth and a fresh install
always lands at the same pinned versions.

## Licensing

The PKGBUILD and scripts here are MIT/Apache-2.0 (matching the parent
repo). The shipped model files retain their upstream licenses:

- microWakeWord pretrained ONNX: Apache-2.0
- Silero VAD: MIT
- whisper.cpp ggml model: MIT (Georgi Gerganov)
