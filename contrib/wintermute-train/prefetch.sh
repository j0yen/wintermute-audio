#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
# Clone the piper sample generator if it isn't present yet.
[ -d piper-sample-generator/.git ] || \
  git clone --depth 1 https://github.com/rhasspy/piper-sample-generator.git piper-sample-generator
mkdir -p piper-sample-generator/models negative_datasets
echo "[$(date +%T)] prefetch: LibriTTS-R generator model (voice variety for positives)"
curl -fsSL -o piper-sample-generator/models/en_US-libritts_r-medium.pt \
  'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt' \
  && echo "  libritts .pt: $(du -h piper-sample-generator/models/en_US-libritts_r-medium.pt | cut -f1)" || echo "  libritts FAILED"
echo "[$(date +%T)] prefetch: smoke negative features (no_speech.zip)"
curl -fsSL -o negative_datasets/no_speech.zip \
  'https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/no_speech.zip' \
  && echo "  no_speech.zip: $(du -h negative_datasets/no_speech.zip | cut -f1)" || echo "  no_speech FAILED"
echo "[$(date +%T)] prefetch DONE"
