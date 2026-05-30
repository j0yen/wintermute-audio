# Changelog

## v0.3.0 — 2026-05-30

Adds `wm-audio fetch-models` subcommand and `src/models.rs` module that
downloads, sha256-verifies, and installs the four pretrained ONNX models
(3 microWakeWord wake-word + 1 Silero VAD) into
`/usr/share/wintermute/models/{wake,vad}/`. Idempotent; writes provenance
sidecar `MODELS.json`; exits 2 with a clear sudo hint when the default
prefix is root-owned. Unblocks the `audio-inference` PRD by guaranteeing
model files are present on disk.

All notable changes to wintermute-audio are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(loose interpretation while pre-1.0).

## [0.2.1] - 2026-05-28

Acoustic echo cancellation lands as a PipeWire drop-in plus a tiny Rust
probe (PRD-wintermute-audio-aec). Once installed, the laptop's own TTS
playback is subtracted from the mic signal so the wake-word detector
stops re-triggering on the daemon's own voice — the precondition for
the inference + barge-in PRDs that follow.

### Added

- `pkg/pipewire-config/99-wintermute-aec.conf` — a `module-echo-cancel`
  drop-in that creates two virtual nodes:
  - `wm-mic-aec`: AEC-cancelled microphone source (set
    `WM_MIC_NODE=wm-mic-aec` to consume it).
  - `wm-spk-aec`: AEC playback reference sink.
  WebRTC AEC under the hood, with gain-control + noise-suppression +
  extended-filter enabled.
- `install.sh` now drops the AEC config into
  `/etc/pipewire/pipewire.conf.d/` (or
  `~/.config/pipewire/pipewire.conf.d/` when /etc isn't writable) and
  restarts the user `pipewire` service. Opt out with
  `WM_AUDIO_INSTALL_AEC=0` or build with
  `--no-default-features --features pipewire-only`.
- `install.sh` mirrors `~/.cargo/bin/wm-audio` into `~/.local/bin/` so
  the binary lives on the bootstrap PATH.
- `source::AecProbe` + `source::run_aec_probe` — startup probe that
  shells out to `pactl list short sources` (overridable via
  `WM_PACTL_BIN`) and checks for the `wm-mic-aec` node. When present,
  the daemon substitutes `wm-mic-aec` for the configured
  `WM_MIC_NODE`; when missing, logs `aec_module_missing` and falls
  back to the existing AC9 mic-node-fallback chain so wm-audio stays
  up half-duplex.
- `aec` Cargo feature (default-on). When off, the probe is compiled
  out and the AEC fallback path is unreachable — the existing
  pipewire-only behaviour is preserved verbatim for AEC vs no-AEC
  A/B comparisons. `pipewire-only` is the matching opt-out marker
  feature.

### Changed

- `main.rs` runs the AEC probe before resolving the mic node, so the
  AC9 fallback chain operates on the effective (AEC-substituted)
  value rather than the raw `WM_MIC_NODE` env var.
- `lib.rs` re-exports `AecProbe`, `AEC_SOURCE_NODE`, `DEFAULT_PACTL`,
  `aec_feature_on`, `parse_pactl_short_sources`,
  `probe_pactl_sources`, and `run_aec_probe`.

## [0.2.0] - 2026-05-28

Mirror of `wintermute-tts` v0.2.0 (PipeWire output): the daemon now
actually streams microphone PCM through the existing UDS fanout.
Until this release the fanout module was wired but no source published
into it, so `mic.sock` stayed empty.

### Added

- `SupervisedPwRecord` — default mic source. Spawns `pw-record` (or
  `$WM_PW_RECORD_BIN`), reads 16 kHz mono i16 frames off stdout, and
  publishes them on the existing broadcast channel + UDS fanout.
- Capture lifecycle events on the bus:
  - `wm.audio.capture.start` once per spawn (with `mic`, `rate`,
    `channels`)
  - `wm.audio.capture.end` once per spawn-exit (with `outcome`,
    `dur_ms`, `reason`)
  - `wm.audio.error` for `pw_record_missing`, `mic_node_fallback`,
    `spawn_failed`
- Persistent-service retry: if `pw-record` exits while we're still
  running, sleep with exponential backoff (1 s → 2 s → 4 s → … capped
  at 30 s) and respawn.
- Soft-fallback when `WM_MIC_NODE` is set but isn't in the live
  `pactl list short sources` output — daemon stays up, captures from
  the PipeWire default, and emits `wm.audio.error{kind:"mic_node_fallback"}`
  rather than refusing to start.
- Soft-fail when `pw-record` is not on `$PATH` — `wm.audio.error{kind:"pw_record_missing"}`,
  backoff, retry; no crash.
- `CapturedBytes` counter (PRD §2.4) on the `Daemon` handle; bumped
  per frame published into the fanout.
- Self-emitted topic filter (`events::is_self_emitted_topic`) for the
  control loop — mirror of the `wm-tts` defense from
  PRD-wintermute-tts-error-loop-suppress. Drops echoes of our own
  `wm.audio.{capture.start, capture.end, error, …}` publishes so any
  future broadening of the subscribe prefix won't recurse.

### Changed

- MSRV bumped 1.85 → 1.88. Required because `agorabus` v0.3 uses
  let-chains, stable in 1.88+.
- `deny.toml` adds `allow-wildcard-paths = true` so in-tree path
  dependencies (e.g. `agorabus`) don't trip the wildcard ban.
- `lib.rs` no longer claims "the PipeWire capture implementation is
  deferred" — it ships here.

### Notes

- `wm-tts` v0.2.0 (commit `9c440ee`, today) wired the symmetric output
  path with `pw-cat`. This release closes the input mirror; together
  the two daemons now own the canonical PCM stream in both directions.
- Bootstrap drift: `/etc/wintermute/conf.d/00-bootstrap.env` hardcodes
  `WM_MIC_NODE=…HiFi__Mic1__source`. On this laptop Mic1 IS available,
  but the AC9 fallback covers laptops where the bootstrap install
  picked the wrong node. The follow-on PRD
  `PRD-wintermute-bootstrap-mic-autodetect` should teach the bootstrap
  to probe `pactl list short sources` itself.

## [0.1.0] - 2026-05-27

Initial scaffold: agorabus event vocabulary, wake/VAD windowing,
NullSource baseline, UDS fanout module. No live mic capture.
