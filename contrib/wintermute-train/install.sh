#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export VIRTUAL_ENV="$PWD/.venv"
[ -x .venv/bin/python ] || { echo "[$(date +%T)] creating .venv (python3.11)..."; uv venv --python 3.11 .venv; }
echo "[$(date +%T)] installing tensorflow-cpu + pymicro-features..."
uv pip install --python .venv/bin/python tensorflow-cpu pymicro-features 2>&1
echo "[$(date +%T)] vendoring + editable-installing kahrendt/microWakeWord..."
# NOTE: the OHF-Voice/micro-wake-word fork dropped the `microwakeword.audio`
# subpackage (clips/spectrograms/augmentation) that this pipeline imports, so
# the deps stage fails with ModuleNotFoundError. Use kahrendt's original repo,
# vendored as microWakeWord-src so the deps stage can apply the compat patch.
[ -d microWakeWord-src/.git ] || \
  git clone --depth 1 https://github.com/kahrendt/microWakeWord.git microWakeWord-src
uv pip install --python .venv/bin/python -e microWakeWord-src --no-deps 2>&1
echo "[$(date +%T)] DONE rc=$?"
