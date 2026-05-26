//! Daemon-wide error type.
//!
//! Pipeline subsystems (capture, ring fanout, agorabus, ONNX inference)
//! map their domain errors into [`AudioError`] so the top-level run loop
//! can branch cleanly on recoverable vs fatal classes.

use thiserror::Error;

/// Errors surfaced by the wintermute-audio daemon.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Failure to read the bootstrap configuration.
    #[error("config: {0}")]
    Config(String),

    /// Capture device error (`PipeWire` / `NoiseTorch` source).
    #[error("capture: {0}")]
    Capture(String),

    /// Agorabus client connect or transport error.
    #[error("agorabus: {0}")]
    Bus(String),

    /// JSON (de)serialization error on bus payloads.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    /// Underlying I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Catch-all wrapper for crates that only return `anyhow::Error`.
    #[error("other: {0}")]
    Other(String),
}

impl From<anyhow::Error> for AudioError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(format!("{e:#}"))
    }
}
