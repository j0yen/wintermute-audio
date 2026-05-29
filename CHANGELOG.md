# Changelog

All notable changes to wintermute-audio are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(loose interpretation while pre-1.0).

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
