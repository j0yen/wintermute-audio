# wintermute-audio: companion config and helpers

The Rust daemon (`wm-audio`) is one piece of the audio stack. To make it
useful you also need a PipeWire echo-cancel config, NoiseTorch-ng for
keyboard/fan rejection, and (later) the pretrained `wm-models` package
for wake-word + VAD + STT ONNX bundles.

This directory ships the hand-rolled pieces. Everything here is
GPL-3-compatible at runtime (NoiseTorch is GPL-3), but the scripts and
configs themselves are MIT/Apache-2.0 to match the parent repo.

## What lives here

| Path | Purpose | Install location |
|---|---|---|
| `pipewire/99-wintermute.conf` | echo-cancel drop-in (PRD §2.1) | `~/.config/pipewire/pipewire.conf.d/` |
| `bin/wm-noise` | NoiseTorch-ng `on / off / status` helper (PRD §2.2) | `~/.local/bin/` |

## Install

```sh
# PipeWire AEC
mkdir -p ~/.config/pipewire/pipewire.conf.d
install -m 644 contrib/pipewire/99-wintermute.conf \
  ~/.config/pipewire/pipewire.conf.d/99-wintermute.conf
systemctl --user restart pipewire

# NoiseTorch-ng (Arch / AUR)
yay -S noisetorch-ng-bin    # or paru -S, makepkg, …

# wm-noise helper
install -m 755 contrib/bin/wm-noise ~/.local/bin/wm-noise
```

After install, `wm-noise on` loads the NoiseTorch virtual source;
`wm-audio` automatically prefers it when present and falls back to the
PipeWire AEC-cancelled `wm-mic-cancelled` source otherwise.

## AEC3 detection

The drop-in requests `webrtc.aec3 = true`. If the local `pipewire`
package was built without AEC3, `wm-audio` logs a warning at startup
and the classic WebRTC AEC engages. To verify:

```sh
pw-cli list-modules | grep -i echo
journalctl --user -u wm-audio -b | grep -i aec3
```

## Deferred

- `wm-models` PKGBUILD — bundles microWakeWord ONNX, Silero VAD,
  and the default whisper.cpp model into `/usr/share/wintermute/models/`.
  Lands once upstream model URLs and hashes are pinned. Until then,
  point `WM_WAKE_MODEL` / `WM_VAD_MODEL` env vars at locally downloaded
  ONNX files.
