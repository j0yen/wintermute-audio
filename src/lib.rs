//! wintermute-audio — mic → AEC/NS → wake/VAD → agorabus events.
//!
//! This iteration (iter-2) wires the daemon skeleton:
//!
//! * Strongly-typed event vocabulary for the topics in PRD §2.3.
//! * A `MicSource` trait abstracting the capture device so `PipeWire`,
//!   file replay, and tests can all drive the pipeline uniformly.
//! * A bounded ring buffer fanout (PCM frames in, fan-out to wake/VAD/
//!   socket consumers in later iterations).
//! * An `agorabus` client connector that publishes the lifecycle events
//!   and subscribes to TTS / dialog control topics, driving a mute
//!   state machine.
//! * Graceful shutdown via tokio signal + a [`Shutdown`] handle that
//!   integration tests can flip without raising real signals.
//!
//! The actual wake-word (`microWakeWord`) and VAD (`Silero`) ONNX
//! inference, plus the `PipeWire` capture implementation, are deferred
//! to subsequent iterations. They plug in behind [`MicSource`] /
//! [`PcmConsumer`] without disturbing the topology built here.

#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod config;
pub mod daemon;
pub mod errors;
pub mod events;
pub mod source;
pub mod state;

pub use config::Config;
pub use daemon::{Daemon, run};
pub use errors::AudioError;
pub use events::{
    AudioEvent, ControlEvent, MuteSource, SpeechChunk, SpeechEnd, SpeechStart, Timestamp,
    Topics, WakeDetected,
};
pub use source::{MicSource, NullSource, PcmFrame, SourceMeta};
pub use state::{MuteReason, MuteState, Shutdown};
