#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
echo "[$(date +%T)] installing torch-cpu + piper-sample-generator deps..."
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchaudio 2>&1
# Pin datasets<4: datasets>=4 routes Audio decode through torchcodec, whose
# native libs need ffmpeg 4-7 + (CUDA) and fail on this CPU-only/ffmpeg-8 box
# (libnvrtc.so.13 / libtorchcodec_core*.so load errors). datasets 3.x decodes
# via soundfile, which is already installed — so NO torchcodec needed.
uv pip install --python .venv/bin/python piper-tts==1.3.0 'numpy>=2,<3' 'datasets>=2,<4' soundfile scipy audio-metadata mmap-ninja tqdm pyyaml 2>&1
uv pip install --python .venv/bin/python tf2onnx onnx onnxruntime 2>&1
# tensorboard: required by tf.summary.scalar during training (TF 2.21 hard-fails without it).
uv pip install --python .venv/bin/python tensorboard 2>&1
echo "[$(date +%T)] torch-stack DONE rc=$?"
