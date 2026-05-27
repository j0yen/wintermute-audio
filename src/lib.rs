//! wintermute-audio — mic → AEC/NS → wake/VAD → agorabus events.
//!
//! Through iter-3 the daemon skeleton wires:
//!
//! * Strongly-typed event vocabulary for the topics in PRD §2.3.
//! * A `MicSource` trait abstracting the capture device so `PipeWire`,
//!   file replay, and tests can all drive the pipeline uniformly.
//! * A UDS PCM fanout (broadcast channel + per-connection writer task)
//!   serving the canonical 16 kHz mono stream to N subscribers.
//! * An `agorabus` client connector that publishes the lifecycle events
//!   and subscribes to TTS / dialog control topics, driving a mute
//!   state machine.
//! * Graceful shutdown via tokio signal + a [`Shutdown`] handle that
//!   integration tests can flip without raising real signals.
//!
//! The actual wake-word (`microWakeWord`) and VAD (`Silero`) ONNX
//! inference, plus the `PipeWire` capture implementation, are deferred
//! to subsequent iterations. They plug in behind [`MicSource`] /
//! [`fanout::channel`] without disturbing the topology built here.

#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod config;
pub mod daemon;
pub mod errors;
pub mod events;
pub mod fanout;
pub mod source;
pub mod state;
pub mod vad;
pub mod wake;

pub use config::Config;
pub use daemon::{Daemon, run};
pub use errors::AudioError;
pub use events::{
    AudioEvent, ControlEvent, MuteSource, SpeechChunk, SpeechEnd, SpeechStart, Timestamp,
    Topics, WakeDetected,
};
pub use source::{MicSource, NullSource, PcmFrame, SourceMeta};
pub use state::{MuteReason, MuteState, Shutdown};
pub use vad::{
    NullVadDetector, SPEECH_END_HANGOVER_MS, SpeechEdge, VAD_FRAME_MS, VAD_STRIDE_SAMPLES,
    VAD_WINDOW_SAMPLES, VadDetector, VadEdgeTracker, VadOutcome, VadWindow,
};
pub use wake::{
    NullWakeDetector, WAKE_STRIDE_SAMPLES, WAKE_WINDOW_SAMPLES, WakeDetector, WakeOutcome,
    WakeSlot, WakeWindow, read_slot, wake_slot, write_slot,
};
