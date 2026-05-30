#!/usr/bin/env bash
# install.sh — install wintermute-audio (the `wm-audio` binary).
#
# Modes:
#   1. Repo-local: invoked as `./install.sh` from a checkout.
#   2. Curl-piped: invoked as `curl ... | bash`. Self-clones into
#      ~/.local/share/wintermute-audio/ then continues.
#
# The build links against pipewire + onnxruntime system libraries;
# install both before running this script (Arch:
# `pacman -S pipewire onnxruntime`, or build onnxruntime from the AUR).

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=""
if [ -f "$SCRIPT_PATH" ]; then
  SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
fi

if [ -z "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/Cargo.toml" ] \
   || ! grep -q '^name = "wintermute-audio"' "$SCRIPT_DIR/Cargo.toml" 2>/dev/null; then
  echo "→ self-cloning j0yen/wintermute-audio..."
  command -v git >/dev/null 2>&1 || { echo "fatal: git not found"; exit 1; }

  CLONE_ROOT="${WINTERMUTE_AUDIO_CLONE_ROOT:-$HOME/.local/share/wintermute-audio}"
  mkdir -p "$(dirname "$CLONE_ROOT")"

  if [ -d "$CLONE_ROOT/.git" ]; then
    echo "→ existing clone at $CLONE_ROOT — refreshing"
    git -C "$CLONE_ROOT" fetch --depth 1 origin main
    git -C "$CLONE_ROOT" reset --hard origin/main
  else
    git clone --depth 1 https://github.com/j0yen/wintermute-audio.git "$CLONE_ROOT"
  fi

  SCRIPT_DIR="$CLONE_ROOT"
fi

cd "$SCRIPT_DIR"

command -v cargo >/dev/null 2>&1 || {
  echo "fatal: cargo not found. Install Rust: https://rustup.rs/"
  exit 1
}

cargo install --path . --locked

if ! command -v wm-audio >/dev/null 2>&1; then
  echo
  echo "! wm-audio installed but not on PATH. Add ~/.cargo/bin to PATH:"
  echo "    export PATH=\"\$HOME/.cargo/bin:\$PATH\""
fi

# AEC config drop (PRD-wintermute-audio-aec §2.1).
#
# Default-on; opt out by setting WM_AUDIO_INSTALL_AEC=0 or by building
# with `cargo install --path . --no-default-features --features pipewire-only`
# (then setting WM_AUDIO_INSTALL_AEC=0 here too, since this script
# can't introspect the cargo features that were applied above).
#
# We try the system-wide path first because PipeWire's system service
# reads /etc/pipewire/pipewire.conf.d/ on session start; if /etc is
# not writable (no sudo) we fall back to the user dir.
if [ "${WM_AUDIO_INSTALL_AEC:-1}" = "1" ]; then
  AEC_CONF_SRC="$SCRIPT_DIR/pkg/pipewire-config/99-wintermute-aec.conf"
  if [ ! -f "$AEC_CONF_SRC" ]; then
    echo "! AEC config $AEC_CONF_SRC missing in checkout; skipping AEC install."
  else
    SYS_DIR="/etc/pipewire/pipewire.conf.d"
    USER_DIR="$HOME/.config/pipewire/pipewire.conf.d"
    INSTALLED=""
    if [ -w "$SYS_DIR" ] 2>/dev/null; then
      install -m 644 "$AEC_CONF_SRC" "$SYS_DIR/99-wintermute-aec.conf"
      INSTALLED="$SYS_DIR/99-wintermute-aec.conf"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
      sudo mkdir -p "$SYS_DIR"
      sudo install -m 644 "$AEC_CONF_SRC" "$SYS_DIR/99-wintermute-aec.conf"
      INSTALLED="$SYS_DIR/99-wintermute-aec.conf"
    fi
    if [ -z "$INSTALLED" ]; then
      mkdir -p "$USER_DIR"
      install -m 644 "$AEC_CONF_SRC" "$USER_DIR/99-wintermute-aec.conf"
      INSTALLED="$USER_DIR/99-wintermute-aec.conf"
    fi
    echo "✓ AEC config: $INSTALLED"

    # Reload the user PipeWire service so the new module loads. Best-
    # effort: skip silently if systemctl --user is unavailable.
    if command -v systemctl >/dev/null 2>&1 && systemctl --user --version >/dev/null 2>&1; then
      systemctl --user restart pipewire pipewire-pulse 2>/dev/null || true
      echo "  pipewire user service restarted"
    fi
  fi
else
  echo "↷ WM_AUDIO_INSTALL_AEC=0; skipping AEC config drop."
fi

# Mirror ~/.cargo/bin/wm-audio into ~/.local/bin/ so the binary is on
# the bootstrap-provided PATH even when ~/.cargo/bin is not.
if [ -x "$HOME/.cargo/bin/wm-audio" ]; then
  mkdir -p "$HOME/.local/bin"
  install -m 755 "$HOME/.cargo/bin/wm-audio" "$HOME/.local/bin/wm-audio"
  echo "✓ wm-audio mirrored to $HOME/.local/bin/wm-audio"
fi

echo "✓ wm-audio installed."
echo
echo "Next:"
echo "  # verify AEC source loaded:"
echo "  pactl list short sources | grep wm-mic-aec"
echo "  # then start the daemon (will probe wm-mic-aec at startup):"
echo "  wm-audio start                       # uses bootstrap env"
echo "  WM_WAKE_WORD=hey_jarvis wm-audio start"
