# wintermute-audio

Microphone pipeline for the wintermute voice assistant.

`wm-audio` owns everything between the raw mic PCM and "ready-for-STT
speech chunks". PipeWire's `module-echo-cancel` removes the laptop's
own TTS from the mic signal; NoiseTorch-ng (optional) suppresses
keyboard / fan / room noise; **microWakeWord** ONNX runs on the
cleaned stream for low-CPU wake detection; **Silero VAD** detects
utterance boundaries; events fan out on agorabus and PCM frames fan
out on a Unix socket so STT, wake, VAD, and any future
speaker-diarization service all read from one canonical capture
stream.

This is the **audio** component of Fleet 1 of the wintermute vision.

## Recent

- **v0.10.0 (2026-06-04)** — Custom **`wintermute`** wake word, end-to-end.
  `WM_WAKE_WORD=wintermute` parses to the new `WakeWord::Wintermute` variant
  (the house answers to its own name); all stock wake words still parse. The
  `fetch-models` manifest gains a `wintermute` entry, and
  `contrib/train-wintermute.sh` documents the offline microWakeWord training
  pipeline (`--help` / `--smoke`) that produces the installable
  `wintermute.onnx`. Detection runs through the bit-exact `[1, 186, 40]` mel
  front-end. Deploying a trained model and live-mic acceptance remain
  asset-/human-gated.

- **AC2 mel parity (2026-06-03)** — `src/features.rs` is now a **bit-exact**
  Rust port of the TFLM `audio_microfrontend` (`pymicro_features.MicroFrontend`):
  fixed-point 512-pt real FFT, mel filterbank, noise reduction, PCAN auto-gain
  and log-scale, all reproducing the reference C frontend's uint16 output
  bit-for-bit. The AC2 golden parity test passes at maxabs 0 (≤1e-3). Also
  fixed a double-scale bug in `contrib/gen_golden_mel.py` and regenerated the
  golden single-scaled. (Implementation-only; not yet versioned/landed.)

- **v0.7.0 (2026-06-03)** — Mel-spectrogram front-end fixes wake-word detection.
  `src/features.rs` produces `[1, 186, 40]` log-mel features (30 ms Hann window,
  10 ms hop, 40 mel bins) matching the microWakeWord training preprocessor.
  `OnnxWakeDetector` rewired to feed the correct shape with load-time contract
  verification (AC1). Honest `fetch-models` manifest: dead upstream wake URLs
  removed, real silero_vad v5.1 sha256 pinned (AC5). 124 lib tests green.

- **v0.4.0 (2026-05-29)** — ONNX inference backends land. `src/inference.rs`
  adds `OnnxWakeDetector` (microWakeWord) and `OnnxVadDetector` (Silero VAD v4)
  via `ort` 2.0. Both fall back to null engines when models are absent
  (`wake_model_missing` / `vad_model_missing` log lines) so the daemon
  always starts cleanly. 16 new lib tests; `cargo deny` clean (rustls, no
  openssl). See PRD-wintermute-audio-inference.

- **v0.3.0 (2026-05-30)** — `wm-audio fetch-models` subcommand ships.
  Downloads, sha256-verifies, and installs the four pretrained ONNX
  models (3 microWakeWord wake-word + 1 Silero VAD) into
  `/usr/share/wintermute/models/{wake,vad}/`. Idempotent; writes provenance
  sidecar `MODELS.json`; exits 2 with a clear `sudo` hint when the default
  prefix is root-owned. `--prefix <dir>` allows unprivileged test installs.
  Unblocks `audio-inference`. See PRD-rouse-wake-vad-models.

- **v0.2.0 (2026-05-28)** — live PipeWire capture ships. The daemon
  now spawns `pw-record` as the default mic source, streams 16 kHz
  mono i16 frames into the UDS fanout, and emits a
  `wm.audio.capture.{start,end}` envelope pair (plus `wm.audio.error`
  for `pw_record_missing` / `mic_node_fallback`). Capture is a
  persistent service — `pw-record` exit triggers retry with 1 s /
  2 s / 4 s / … capped-at-30 s exponential backoff. Mirror of the
  same-day `wintermute-tts` PipeWire-output ship; see
  PRD-wintermute-audio-pipewire-input.

## What it does

On startup, `wm-audio`:

- Opens the configured input node (`WM_MIC_NODE` from
  `/etc/wintermute/conf.d/00-bootstrap.env`), routed through
  PipeWire's AEC and (optionally) NoiseTorch.
- Resamples to 16 kHz mono PCM if needed.
- Spawns three async consumers of the shared ring buffer:
  - **socket fanout** — accepts UDS connections at
    `$XDG_RUNTIME_DIR/wintermute/mic.sock` and pushes PCM frames to
    each subscriber.
  - **wake** — runs microWakeWord ONNX inference every 80 ms on a
    1280-sample window.
  - **VAD** — runs Silero VAD ONNX every 32 ms; emits `speech.start`
    on rising edge with hangover, `speech.end` on 500-ms-confirmed
    silence.

Send `wm.audio.reload` on agorabus to hot-swap the wake word; the
daemon re-reads env and swaps the ONNX model without restarting.

## Events published

| Topic | Payload |
|---|---|
| `wm.audio.wake` | `{wake_word, confidence, ts}` |
| `wm.audio.speech.start` | `{ts}` |
| `wm.audio.speech.chunk` | `{seq, pcm_b64, ts}` |
| `wm.audio.speech.end` | `{duration_ms, ts}` |
| `wm.audio.mute` / `wm.audio.unmute` | `{ts}` |

## Events subscribed

| Topic | Behavior |
|---|---|
| `wm.tts.start` | mute wake detection (AEC tail guard) |
| `wm.tts.end` | unmute wake detection |
| `wm.dialog.mute_request` | mute mic entirely |
| `wm.dialog.unmute_request` | unmute |

## Acceptance tests

1. With `wm-tts` playing a 5-second sentence over speakers, the wake
   word does not fire across 30 repetitions. (AEC working.)
2. Typing on the keyboard while the mic is open reduces input level
   by ≥10 dB vs. AEC-only mode (NoiseTorch).
3. Wake → `wm.audio.wake` publish latency: <200 ms.
4. End-of-speech → `wm.audio.speech.end` latency: <500 ms after
   confirmed silence.
5. False-accept rate on 60 min of ambient living-room speech: <0.5/hr.
6. `wm.audio.reload` completes in <2 s without dropping capture.
7. Two simultaneous mic.sock subscribers consume PCM for 60 min
   without dropouts.
8. Daemon recovers from `systemctl --user restart pipewire` within 5 s.

## Install

One-liner (curl-pipe):

```
curl -fsSL https://raw.githubusercontent.com/j0yen/wintermute-audio/main/install.sh | bash
```

Or from a checkout:

```
git clone https://github.com/j0yen/wintermute-audio
cd wintermute-audio
./install.sh
```

The default build links against `pipewire` + `onnxruntime` system
libraries; install both before running `install.sh`.

Drop the PipeWire AEC config (see `contrib/pipewire/`) into
`~/.config/pipewire/pipewire.conf.d/99-wintermute.conf` and restart
PipeWire (`systemctl --user restart pipewire`). Place the
microWakeWord + Silero VAD ONNX models under
`/usr/share/wintermute/models/`.

Start the daemon:

```
wm-audio start                         # uses bootstrap env
WM_WAKE_WORD=hey_jarvis wm-audio start # explicit stock wake word
WM_WAKE_WORD=wintermute wm-audio start # custom "wintermute" wake word
```

The `wintermute` wake word is trained offline via
`contrib/train-wintermute.sh` (microWakeWord; `--help` documents the
pipeline, `--smoke` runs a tiny end-to-end pass). Install the resulting
`wintermute.onnx` under `<prefix>/wake/wintermute.onnx`; the daemon loads it
through the same `[1, 186, 40]` mel front-end as the stock models.

## Hardware reality verification

ACs 1, 2, 5, 8 are acoustic/OS-bound (live AEC against real speaker
playback, ≥10 dB noise reduction measured on real mic input, real-recording
false-accept rate, recovery from a live PipeWire restart). They are declared
in the PRD's `deferred_acs:` + `mock_unjustified_for:` frontmatter with a
one-sentence justification each, because an in-process fake would assert the
math we wrote rather than the hardware's real acoustic behavior.

To validate them against real audio hardware, run:

```sh
cargo test --features=real-hardware
```

This feature is opt-in and off by default, so `cargo test` stays green on
hosts without a microphone or PipeWire graph. The drift-report sweep that
compares mock vs. real-hardware outcomes (`hardware-drift.json`) is
scaffolded as a follow-on PRD and is not invoked by default.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
