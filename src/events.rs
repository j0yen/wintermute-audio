//! Event vocabulary published on / subscribed from `agorabus`.
//!
//! Mirrors PRD §2.3. Each enum variant carries a typed payload that
//! serializes to the JSON shapes documented in the table. Topic strings
//! are interned in [`Topics`] so call sites can't drift.

use serde::{Deserialize, Serialize};

/// Topic strings published on / subscribed from the bus.
///
/// Held as a struct of `&'static str` to avoid stringly-typed sites and
/// to give one source of truth that an integration test can lock down.
pub struct Topics;

impl Topics {
    /// Topic prefix owned by this daemon (for subscriptions other
    /// components might want).
    pub const AUDIO_PREFIX: &'static str = "wm.audio.";

    /// Wake-word detection.
    pub const WAKE: &'static str = "wm.audio.wake";
    /// Speech rising-edge.
    pub const SPEECH_START: &'static str = "wm.audio.speech.start";
    /// Speech mid-stream chunk (only while active).
    pub const SPEECH_CHUNK: &'static str = "wm.audio.speech.chunk";
    /// Speech falling-edge (after confirmed silence).
    pub const SPEECH_END: &'static str = "wm.audio.speech.end";
    /// Mic-mute notification.
    pub const MUTE: &'static str = "wm.audio.mute";
    /// Mic-unmute notification.
    pub const UNMUTE: &'static str = "wm.audio.unmute";
    /// Hot-swap signal (re-read env, swap ONNX models).
    pub const RELOAD: &'static str = "wm.audio.reload";

    // Subscribed topics
    /// TTS playback began (mute wake to avoid double-fire).
    pub const TTS_START: &'static str = "wm.tts.start";
    /// TTS playback finished (unmute wake).
    pub const TTS_END: &'static str = "wm.tts.end";
    /// Dialog explicitly asked to mute the mic.
    pub const DIALOG_MUTE_REQ: &'static str = "wm.dialog.mute_request";
    /// Dialog explicitly asked to unmute the mic.
    pub const DIALOG_UNMUTE_REQ: &'static str = "wm.dialog.unmute_request";
}

/// Monotonic-ish timestamp attached to every event.
///
/// We emit Unix epoch milliseconds — recipients that need monotonic
/// time can read it from the bus daemon's own delivery timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Current wall time as epoch milliseconds.
    ///
    /// Returns `Timestamp(0)` if the system clock is set before the
    /// Unix epoch (a "shouldn't happen" we still don't want to panic on).
    #[must_use]
    pub fn now() -> Self {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| {
                let m = d.as_millis();
                u64::try_from(m).unwrap_or(u64::MAX)
            });
        Self(ms)
    }
}

/// `wm.audio.wake` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeDetected {
    /// Kebab-case wake-word label, e.g. `"hey-jarvis"`.
    pub wake_word: String,
    /// Model confidence, `0.0..=1.0`.
    pub confidence: f32,
    /// Emission timestamp.
    pub ts: Timestamp,
}

/// `wm.audio.speech.start` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechStart {
    /// Rising-edge timestamp.
    pub ts: Timestamp,
}

/// `wm.audio.speech.chunk` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechChunk {
    /// Monotonic sequence number within a single utterance.
    pub seq: u64,
    /// Base64-encoded little-endian i16 PCM, 16 kHz mono.
    pub pcm_b64: String,
    /// Emission timestamp.
    pub ts: Timestamp,
}

/// `wm.audio.speech.end` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechEnd {
    /// Duration of the just-completed utterance in milliseconds.
    pub duration_ms: u32,
    /// Falling-edge timestamp.
    pub ts: Timestamp,
}

/// Tagged source of a mute/unmute transition (informational; not part of
/// the bus payload, which carries only `{ts}` per PRD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MuteSource {
    /// Daemon muted itself in response to `wm.tts.start`.
    TtsAec,
    /// Dialog requested via `wm.dialog.mute_request`.
    DialogRequest,
    /// Operator / CLI.
    Manual,
}

/// All events this daemon publishes. Use [`AudioEvent::topic`] to get
/// the bus topic and [`AudioEvent::payload`] for the JSON payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioEvent {
    /// Wake word fired.
    Wake(WakeDetected),
    /// Speech started.
    SpeechStart(SpeechStart),
    /// Speech ongoing chunk.
    SpeechChunk(SpeechChunk),
    /// Speech ended.
    SpeechEnd(SpeechEnd),
    /// Mic muted (`{ts}`).
    Mute {
        /// Timestamp the mute took effect.
        ts: Timestamp,
    },
    /// Mic unmuted (`{ts}`).
    Unmute {
        /// Timestamp the unmute took effect.
        ts: Timestamp,
    },
}

impl AudioEvent {
    /// Bus topic this event publishes on.
    #[must_use]
    pub const fn topic(&self) -> &'static str {
        match self {
            Self::Wake(_) => Topics::WAKE,
            Self::SpeechStart(_) => Topics::SPEECH_START,
            Self::SpeechChunk(_) => Topics::SPEECH_CHUNK,
            Self::SpeechEnd(_) => Topics::SPEECH_END,
            Self::Mute { .. } => Topics::MUTE,
            Self::Unmute { .. } => Topics::UNMUTE,
        }
    }

    /// JSON payload as published on the bus. Matches the per-topic
    /// shapes documented in the PRD.
    ///
    /// # Errors
    ///
    /// Returns the underlying `serde_json` error if any payload field
    /// fails to serialize (in practice, none of these can).
    pub fn payload(&self) -> Result<serde_json::Value, serde_json::Error> {
        match self {
            Self::Wake(w) => serde_json::to_value(w),
            Self::SpeechStart(s) => serde_json::to_value(s),
            Self::SpeechChunk(c) => serde_json::to_value(c),
            Self::SpeechEnd(e) => serde_json::to_value(e),
            Self::Mute { ts } | Self::Unmute { ts } => Ok(serde_json::json!({ "ts": ts })),
        }
    }
}

/// Events this daemon subscribes to (control plane).
///
/// Wire format: the `agorabus` daemon delivers an opaque
/// `serde_json::Value`; [`ControlEvent::from_topic`] classifies a topic
/// + payload pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlEvent {
    /// `wm.tts.start` — mute wake while TTS plays.
    TtsStart,
    /// `wm.tts.end` — re-arm wake.
    TtsEnd,
    /// `wm.dialog.mute_request` — operator/policy mute of mic capture.
    MuteRequest,
    /// `wm.dialog.unmute_request` — release a `MuteRequest`.
    UnmuteRequest,
    /// `wm.audio.reload` — re-read env + swap models without dropping capture.
    Reload,
}

impl ControlEvent {
    /// Classify an incoming bus topic. Returns `None` for topics that
    /// fall outside this daemon's control surface.
    #[must_use]
    pub fn from_topic(topic: &str) -> Option<Self> {
        match topic {
            Topics::TTS_START => Some(Self::TtsStart),
            Topics::TTS_END => Some(Self::TtsEnd),
            Topics::DIALOG_MUTE_REQ => Some(Self::MuteRequest),
            Topics::DIALOG_UNMUTE_REQ => Some(Self::UnmuteRequest),
            Topics::RELOAD => Some(Self::Reload),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]

    use super::*;

    #[test]
    fn topic_table_matches_prd() {
        // Lock the PRD §2.3 topic table down. If anyone renames a
        // topic this test pins the regression.
        assert_eq!(Topics::WAKE, "wm.audio.wake");
        assert_eq!(Topics::SPEECH_START, "wm.audio.speech.start");
        assert_eq!(Topics::SPEECH_CHUNK, "wm.audio.speech.chunk");
        assert_eq!(Topics::SPEECH_END, "wm.audio.speech.end");
        assert_eq!(Topics::MUTE, "wm.audio.mute");
        assert_eq!(Topics::UNMUTE, "wm.audio.unmute");
        assert_eq!(Topics::TTS_START, "wm.tts.start");
        assert_eq!(Topics::TTS_END, "wm.tts.end");
        assert_eq!(Topics::DIALOG_MUTE_REQ, "wm.dialog.mute_request");
        assert_eq!(Topics::DIALOG_UNMUTE_REQ, "wm.dialog.unmute_request");
        assert_eq!(Topics::RELOAD, "wm.audio.reload");
    }

    #[test]
    fn wake_payload_round_trip() {
        let ev = AudioEvent::Wake(WakeDetected {
            wake_word: "hey-jarvis".into(),
            confidence: 0.87,
            ts: Timestamp(42),
        });
        assert_eq!(ev.topic(), "wm.audio.wake");
        let v = ev.payload().ok();
        let v = v.unwrap_or(serde_json::Value::Null);
        assert_eq!(v["wake_word"], "hey-jarvis");
        let conf = v["confidence"].as_f64().unwrap_or(-1.0);
        assert!((conf - 0.87).abs() < 1e-3, "round-trip lost confidence");
        assert_eq!(v["ts"], 42);
    }

    #[test]
    fn speech_chunk_payload_shape() {
        let ev = AudioEvent::SpeechChunk(SpeechChunk {
            seq: 7,
            pcm_b64: "AAAA".into(),
            ts: Timestamp(100),
        });
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        assert_eq!(v["seq"], 7);
        assert_eq!(v["pcm_b64"], "AAAA");
        assert_eq!(v["ts"], 100);
    }

    #[test]
    fn mute_payload_is_just_ts() {
        let ev = AudioEvent::Mute { ts: Timestamp(99) };
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        let obj_len = v.as_object().map_or(0, serde_json::Map::len);
        assert_eq!(obj_len, 1, "PRD says mute payload is just {{ts}}");
        assert_eq!(v["ts"], 99);
    }

    #[test]
    fn control_classifier_covers_subscriptions() {
        assert_eq!(
            ControlEvent::from_topic("wm.tts.start"),
            Some(ControlEvent::TtsStart)
        );
        assert_eq!(
            ControlEvent::from_topic("wm.tts.end"),
            Some(ControlEvent::TtsEnd)
        );
        assert_eq!(
            ControlEvent::from_topic("wm.dialog.mute_request"),
            Some(ControlEvent::MuteRequest)
        );
        assert_eq!(
            ControlEvent::from_topic("wm.dialog.unmute_request"),
            Some(ControlEvent::UnmuteRequest)
        );
        assert_eq!(
            ControlEvent::from_topic("wm.audio.reload"),
            Some(ControlEvent::Reload)
        );
        assert_eq!(ControlEvent::from_topic("wm.audio.wake"), None);
        assert_eq!(ControlEvent::from_topic("unrelated.topic"), None);
    }
}
