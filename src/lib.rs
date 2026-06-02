//! wintermute-audio — mic → AEC/NS → wake/VAD → agorabus events.
//!
//! v0.2.0 wires the daemon end-to-end:
//!
//! * Strongly-typed event vocabulary for the topics in PRD §2.3,
//!   including the v0.2.0 capture lifecycle events
//!   (`wm.audio.capture.start`, `wm.audio.capture.end`,
//!   `wm.audio.error`).
//! * A `MicSource` trait abstracting the capture device so `PipeWire`,
//!   file replay, and tests can all drive the pipeline uniformly.
//! * A shipping default [`SupervisedPwRecord`] source that spawns
//!   `pw-record` (PRD-pipewire-input §2.1), reads 320-sample i16
//!   frames off stdout, and respawns with 1 s / 2 s / 4 s / … 30 s
//!   backoff so capture is a persistent service.
//! * A UDS PCM fanout (broadcast channel + per-connection writer task)
//!   serving the canonical 16 kHz mono stream to N subscribers.
//! * An `agorabus` client connector that publishes the lifecycle events
//!   and subscribes to TTS / dialog control topics, driving a mute
//!   state machine. Self-emitted topic echoes (`wm.audio.error`, …)
//!   are silently dropped — see [`events::is_self_emitted_topic`].
//! * Graceful shutdown via tokio signal + a [`Shutdown`] handle that
//!   integration tests can flip without raising real signals.
//!
//! Wake-word (`microWakeWord`) and VAD (`Silero`) ONNX inference plug
//! in behind [`WakeDetector`] / [`VadDetector`] via the [`inference`]
//! module (v0.4.0, PRD-wintermute-audio-inference). Both backends fall
//! back to their null engines when models are absent so the daemon
//! always starts cleanly.

#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod config;
pub mod daemon;
pub mod errors;
pub mod events;
pub mod fanout;
pub mod inference;
pub mod models;
pub mod selftest;
pub mod source;
pub mod state;
pub mod vad;
pub mod wake;

pub use config::{Config, DEFAULT_PW_RECORD};
pub use daemon::{CapturedBytes, Daemon};
pub use errors::AudioError;
pub use events::{
    AudioErrorPayload, AudioEvent, CaptureEnd, CaptureStart, ControlEvent, MuteSource,
    SpeechChunk, SpeechEnd, SpeechStart, Timestamp, Topics, WakeDetected,
    is_self_emitted_topic,
};
pub use source::{
    AEC_SOURCE_NODE, AecProbe, DEFAULT_PACTL, MicNodeSelection, MicSource, NullSource,
    PW_RECORD_FRAME_SAMPLES, PcmFrame, PwRecordSource, SourceMeta, SupervisedPwRecord,
    aec_feature_on, backoff_ms_for_attempt, parse_pactl_short_sources, probe_pactl_sources,
    pw_record_args, resolve_mic_node, run_aec_probe,
};
pub use models::{
    InstallOutcome, ListEntry, ModelEntry, ModelError, ModelKind, OutputFormat, ProvenanceRecord,
    MANIFEST, build_list, dir_is_writable, is_current, provision_one, read_provenance,
    sha256_hex, upsert_provenance, write_provenance,
};
pub use inference::{OnnxVadDetector, OnnxWakeDetector, load_or_null_vad, load_or_null_wake};
pub use state::{MuteReason, MuteState, Shutdown};
pub use vad::{
    NullVadDetector, SPEECH_END_HANGOVER_MS, SpeechEdge, VAD_FRAME_MS, VAD_STRIDE_SAMPLES,
    VAD_WINDOW_SAMPLES, VadDetector, VadEdgeTracker, VadOutcome, VadWindow,
};
pub use wake::{
    NullWakeDetector, WAKE_STRIDE_SAMPLES, WAKE_WINDOW_SAMPLES, WakeDetector, WakeOutcome,
    WakeSlot, WakeWindow, read_slot, wake_slot, write_slot,
};
