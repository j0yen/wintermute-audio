//! Daemon configuration: bootstrap env + defaults.
//!
//! The PRD calls for reading `/etc/wintermute/conf.d/00-bootstrap.env`.
//! For the iter-2 skeleton we read directly from environment variables
//! (the bootstrap file is `export`-shaped, so a systemd unit would
//! `EnvironmentFile=` it before exec'ing the daemon).

use crate::errors::AudioError;
use std::path::PathBuf;

/// Pretrained wake words shipped via the `wm-models` bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeWord {
    /// "hey jarvis" — microWakeWord default.
    HeyJarvis,
    /// "okay nabu" — Home Assistant default.
    OkayNabu,
    /// "hey mycroft" — legacy compatibility.
    HeyMycroft,
}

impl WakeWord {
    /// Parse from the bootstrap env value.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Config`] for unknown wake-word identifiers.
    pub fn parse(s: &str) -> Result<Self, AudioError> {
        match s.trim() {
            "hey_jarvis" | "hey-jarvis" | "hey jarvis" => Ok(Self::HeyJarvis),
            "okay_nabu" | "okay-nabu" | "okay nabu" => Ok(Self::OkayNabu),
            "hey_mycroft" | "hey-mycroft" | "hey mycroft" => Ok(Self::HeyMycroft),
            other => Err(AudioError::Config(format!(
                "unknown WM_WAKE_WORD={other:?}; expected hey_jarvis|okay_nabu|hey_mycroft"
            ))),
        }
    }

    /// Stable kebab-case label used in published events.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::HeyJarvis => "hey-jarvis",
            Self::OkayNabu => "okay-nabu",
            Self::HeyMycroft => "hey-mycroft",
        }
    }
}

/// Default `pw-record` binary name on `$PATH`.
///
/// PRD §5: overridable via `WM_PW_RECORD_BIN` so tests / packaging
/// can substitute. Mirror of `wm-tts`'s `WM_PW_CAT_BIN`.
pub const DEFAULT_PW_RECORD: &str = "pw-record";

/// Daemon configuration assembled from environment + sensible defaults.
#[derive(Debug, Clone)]
pub struct Config {
    /// Capture node name (`PipeWire`). Empty string means "default input".
    pub mic_node: String,
    /// Selected wake word.
    pub wake_word: WakeWord,
    /// Wake-word activation threshold (0.0–1.0).
    pub wake_threshold: f32,
    /// UDS path for the PCM fanout socket.
    pub mic_socket: PathBuf,
    /// Path to the agorabus daemon socket.
    pub bus_socket: PathBuf,
    /// Session id this daemon announces on the bus.
    pub session_id: String,
    /// Path to the `pw-record` binary (PRD §5). Overridable via
    /// `WM_PW_RECORD_BIN`; defaults to `"pw-record"` (resolved on
    /// `$PATH`).
    pub pw_record_bin: String,
}

impl Config {
    /// Load configuration from environment + sensible defaults.
    ///
    /// Honored variables:
    ///
    /// * `WM_MIC_NODE` — capture device name (defaults to empty / PW default).
    /// * `WM_WAKE_WORD` — `hey_jarvis|okay_nabu|hey_mycroft` (default `hey_jarvis`).
    /// * `WM_WAKE_THRESHOLD` — float `0.0..=1.0` (default `0.6`).
    /// * `WM_MIC_SOCK` — UDS path for PCM fanout (default
    ///   `$XDG_RUNTIME_DIR/wintermute/mic.sock`).
    /// * `AGORABUS_SOCK` — path to agorabus daemon (default `~/.cache/agorabus/sock`).
    /// * `WM_AUDIO_SESSION` — session id (default `wm-audio-<pid>`).
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Config`] for malformed wake-word names or
    /// out-of-range thresholds.
    pub fn from_env() -> Result<Self, AudioError> {
        let mic_node = std::env::var("WM_MIC_NODE").unwrap_or_default();

        let wake_word = std::env::var("WM_WAKE_WORD")
            .ok()
            .as_deref()
            .map_or(Ok(WakeWord::HeyJarvis), WakeWord::parse)?;

        let wake_threshold = match std::env::var("WM_WAKE_THRESHOLD") {
            Ok(s) => {
                let v: f32 = s.parse().map_err(|e| {
                    AudioError::Config(format!("WM_WAKE_THRESHOLD={s:?}: {e}"))
                })?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(AudioError::Config(format!(
                        "WM_WAKE_THRESHOLD={v} out of range [0.0, 1.0]"
                    )));
                }
                v
            }
            Err(_) => 0.6_f32,
        };

        let mic_socket = std::env::var("WM_MIC_SOCK").map_or_else(
            |_| {
                let xdg = std::env::var("XDG_RUNTIME_DIR")
                    .unwrap_or_else(|_| "/tmp".to_string());
                PathBuf::from(xdg).join("wintermute/mic.sock")
            },
            PathBuf::from,
        );

        let bus_socket = std::env::var("AGORABUS_SOCK")
            .map_or_else(|_| agorabus::default_socket_path(), PathBuf::from);

        let session_id = std::env::var("WM_AUDIO_SESSION")
            .unwrap_or_else(|_| format!("wm-audio-{}", std::process::id()));

        let pw_record_bin = std::env::var("WM_PW_RECORD_BIN")
            .unwrap_or_else(|_| DEFAULT_PW_RECORD.to_owned());

        Ok(Self {
            mic_node,
            wake_word,
            wake_threshold,
            mic_socket,
            bus_socket,
            session_id,
            pw_record_bin,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wake_word_parse_round_trip() {
        let cases = [
            ("hey_jarvis", WakeWord::HeyJarvis, "hey-jarvis"),
            ("okay_nabu", WakeWord::OkayNabu, "okay-nabu"),
            ("hey_mycroft", WakeWord::HeyMycroft, "hey-mycroft"),
        ];
        for (input, parsed, label) in cases {
            let got = WakeWord::parse(input).ok();
            assert_eq!(got, Some(parsed), "input={input}");
            assert_eq!(parsed.as_label(), label);
        }
    }

    #[test]
    fn wake_word_parse_unknown_errors() {
        let err = WakeWord::parse("nope").err();
        assert!(err.is_some(), "expected error for unknown wake word");
    }
}
