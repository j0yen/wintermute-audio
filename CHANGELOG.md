# Changelog

## v0.13.0 — 2026-06-13

pulse-hearing-probe: add --emit to wm-audio selftest publishing wm.health.hearing envelope

## v0.12.0 — 2026-06-13

changeover-daemon-claims: wire wm-audio daemon to hold agorabus://daemon/wm-audio claim via ClaimGuard for lifetime of process

## v0.11.0 — 2026-06-05

Turn-id spine (PRD lucid-turn-id, AC1/AC2/AC5 — wm-audio leg).

- **`TurnId` type** in `events.rs`: collision-resistant `<unix_ms_hex>-<seq_hex>`
  token backed by a process-global `AtomicU32` counter. `TurnId::mint()` mints
  a fresh id; `TurnId::parse(s)` validates and round-trips; `Display` for
  logging. No new dependencies.
- **`WakeDetected`, `SpeechStart`, `SpeechChunk`, `SpeechEnd`** all gain
  `turn_id: Option<TurnId>`. Field is `#[serde(skip_serializing_if = "Option::is_none")]`
  so legacy consumers never see a null field.
- **`daemon.rs`** mints a `TurnId` at each wake and propagates the same id onto
  every `speech.start`, `speech.chunk`, and `speech.end` for that utterance.
- **8 new tests**: `turn_id_mint_is_parseable`, `turn_id_two_mints_differ`,
  `turn_id_parse_rejects_garbage`, `turn_id_display`,
  `wake_with_turn_id_serializes_field`, `legacy_wake_payload_deserializes_without_turn_id`,
  `legacy_speech_end_payload_deserializes_without_turn_id`, plus
  `wake_payload_round_trip` extended to assert absent `turn_id` stays absent.
  135 tests green.

## v0.10.0 — 2026-06-04

End-to-end custom `wintermute` wake word. `WM_WAKE_WORD=wintermute` now
parses to the new `WakeWord::Wintermute` enum variant (config.rs), with
`as_label` → `"wintermute"` and the unknown-word error listing it; all
pre-existing wake words still parse. The `fetch-models` manifest carries a
`wintermute` wake entry, and `contrib/train-wintermute.sh` documents the
offline microWakeWord training pipeline (`--help`/`--smoke`) that produces
the installable `wintermute.onnx`. Detection runs through the bit-exact
`[1, 186, 40]` mel front-end shipped in v0.9.0. 140 tests green; `cargo
deny check bans licenses sources` clean. ACs 2–6 (real model asset /
recorded-utterance / training-run) remain asset- and human-gated per the
PRD's no-self-fixture rule.

## v0.9.0 — 2026-06-04

Bit-exact mel front-end: the wake detector now feeds the model `[1, 186, 40]`
log-mel features (AC1 shape contract, verified against the ONNX graph's declared
input dims), produced by a TFLM `micro_features` microfrontend port that matches
the training preprocessor byte-for-byte (AC2 mel parity, maxabs=0 vs the golden
vector exported from `contrib/wintermute-train`). Replaces the original `[1, 1280]`
raw-PCM path that could never have fired against a real microWakeWord model.
AC3 (held-out clip) and AC6 (live mic) remain PENDING-USER per the no-self-fixture
rule — this version makes the plumbing bit-exact, not the model deploy-quality.

## v0.8.1 — 2026-06-03

Regression fix (test-only): `wake_bus_smoke` integration tests (`wake_publishes_through_real_bus`,
`wake_publish_within_two_hundred_ms_ac3`) drove `NullSource` with 50×320 = 16 000 samples — under
one full mel window. The v0.7.0 mel front-end raised the daemon wake window to `MEL_WINDOW_SAMPLES`
(30 240 samples) with a `MEL_STRIDE_SAMPLES` (2 560) advance, so `MelWindowBuffer::next_window()`
never yielded, the scripted detector never reached its 3rd `process` call, and no `wm.audio.wake`
event published. Sized the test frame count from the mel constants (`MEL_WINDOW_SAMPLES +
2*MEL_STRIDE_SAMPLES`, ceil-div + margin) so ≥3 windows drain and the detector fires. No production
code change; full suite green.

## v0.8.0 — 2026-06-03

Implements PRD `earshot-vad-patience`: the VAD silence-hangover before
`wm.audio.speech.end` is now configurable, with an elder-friendly 1500 ms
default that tolerates mid-sentence pauses without cutting off the speaker.
Set `WM_VAD_SILENCE_MS` to override; values are clamped to [300, 3000] ms.

## v0.7.0 — 2026-06-03

Fix wake-word detection contract mismatch: add `src/features.rs` mel-spectrogram
front-end producing `[186, 40]` log-mel features (30 ms Hann window, 10 ms hop,
40 triangular mel bins, 125–7500 Hz) matching the `OHF-Voice/micro-wake-word`
training preprocessor. Rewire `OnnxWakeDetector::process` to feed `[1, 186, 40]`
f32 tensors instead of raw PCM `[1, 1280]`; add load-time shape-contract
verification (AC1). Repair `fetch-models` manifest: remove 404 upstream wake
URLs, pin real silero_vad v5.1 sha256 `2623a29…` (AC5). Add `MelWindowBuffer`
ring buffer for mel-stride accumulation. 124 lib tests green.

## v0.6.0 — 2026-06-02

Add `wm-audio selftest` subcommand: runtime voice-path diagnostic with fixture
mode (scripted wake+VAD via in-process bus) and live mode (subscribes to running
daemon). Verdicts: healthy | deaf: no-wake | deaf: no-speech-segment |
unreachable: <reason>. Exit codes 0/1/2 mirror `agorabus doctor`. --format
json|text. 10 new unit tests. Encodes the 2026-05-29 diagnostic session.

## v0.5.0 — 2026-05-30

`wm-audio` v0.4.0 adds ONNX inference backends for wake-word detection
(microWakeWord via `OnnxWakeDetector`) and VAD (Silero VAD v4 via
`OnnxVadDetector`) behind the existing `WakeDetector` / `VadDetector`
traits. Both backends fall back to null engines when models are absent,
satisfying PRD AC7. 16 new lib tests; `cargo deny` clean (rustls, no
openssl). See PRD-wintermute-audio-inference.

## v0.4.0 — 2026-05-30

PRD-wintermute-audio-aec: acoustic echo cancellation — already shipped in v0.2.1.
AEC probe, PipeWire config drop, install.sh integration, and lib.rs re-exports
were integrated in prior tick; this tick confirms gates pass (87 tests green).

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
