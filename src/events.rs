//! Event vocabulary published on / subscribed from `agorabus`.
//!
//! Mirrors PRD §2.3. Each enum variant carries a typed payload that
//! serializes to the JSON shapes documented in the table. Topic strings
//! are interned in [`Topics`] so call sites can't drift.

use std::sync::atomic::{AtomicU32, Ordering};

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
    /// Capture session began (after `pw-record` spawn).
    pub const CAPTURE_START: &'static str = "wm.audio.capture.start";
    /// Capture session ended (graceful stop or `pw-record` exit).
    pub const CAPTURE_END: &'static str = "wm.audio.capture.end";
    /// Capture / configuration failure (e.g. `pw-record` missing,
    /// `WM_MIC_NODE` unresolvable and no fallback available).
    pub const ERROR: &'static str = "wm.audio.error";
    /// Daemon-shutdown notice (best-effort; published during teardown).
    pub const SHUTDOWN: &'static str = "wm.audio.shutdown";

    /// Every outbound topic this daemon publishes.
    ///
    /// Used by the dispatch loop to silently skip events whose topic
    /// matches one of our own publishes. Without this filter, the
    /// broadcast bus echoes our `wm.audio.error` back to us, the
    /// control-event classifier rejects it as an unknown topic, and we
    /// would publish another `wm.audio.error` describing the rejection
    /// — the recursive storm template that bit `wm-tts` before
    /// PRD-wintermute-tts-error-loop-suppress shipped. See
    /// PRD-wintermute-audio-pipewire-input §2.2 for the analogous
    /// requirement on the input side.
    pub const ALL_OUTBOUND: &'static [&'static str] = &[
        Self::WAKE,
        Self::SPEECH_START,
        Self::SPEECH_CHUNK,
        Self::SPEECH_END,
        Self::MUTE,
        Self::UNMUTE,
        Self::CAPTURE_START,
        Self::CAPTURE_END,
        Self::ERROR,
        Self::SHUTDOWN,
    ];

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

/// A collision-resistant turn identifier minted at wake and propagated
/// downstream through every event in the same spoken turn.
///
/// Format: `<unix_ms_hex>-<seq_hex>` where `seq_hex` is a
/// process-global monotonic counter that distinguishes two wakes that
/// land in the same millisecond. Both parts are zero-padded hex to keep
/// the string lexicographically sortable and human-readable.
///
/// The field is **optional** in every envelope — events without a
/// `turn_id` (legacy or from a daemon that hasn't yet adopted this PRD)
/// remain valid; consumers MUST handle `None` gracefully.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub String);

/// Global monotone counter for same-millisecond collision avoidance.
static TURN_SEQ: AtomicU32 = AtomicU32::new(0);

impl TurnId {
    /// Mint a fresh, collision-resistant `TurnId`.
    ///
    /// Uses wall-clock milliseconds and a process-global counter so two
    /// calls in the same millisecond produce distinct ids.
    #[must_use]
    pub fn mint() -> Self {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| {
                let m = d.as_millis();
                u64::try_from(m).unwrap_or(u64::MAX)
            });
        let seq = TURN_SEQ.fetch_add(1, Ordering::Relaxed);
        Self(format!("{ms:013x}-{seq:04x}"))
    }

    /// Parse a `TurnId` from a string slice, returning `None` if the
    /// format does not match `<hex>-<hex>`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let mut parts = s.splitn(2, '-');
        let ms_part = parts.next()?;
        let seq_part = parts.next()?;
        // Validate both parts are non-empty hex strings.
        if ms_part.is_empty()
            || seq_part.is_empty()
            || !ms_part.chars().all(|c| c.is_ascii_hexdigit())
            || !seq_part.chars().all(|c| c.is_ascii_hexdigit())
        {
            return None;
        }
        Some(Self(s.to_owned()))
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TurnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
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
    /// Turn identifier minted at this wake. Shared by all downstream
    /// events in the same spoken turn. Optional for backward compat.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<TurnId>,
}

/// `wm.audio.speech.start` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechStart {
    /// Rising-edge timestamp.
    pub ts: Timestamp,
    /// Turn identifier propagated from the triggering wake event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<TurnId>,
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
    /// Turn identifier propagated from the triggering wake event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<TurnId>,
}

/// `wm.audio.speech.end` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechEnd {
    /// Duration of the just-completed utterance in milliseconds.
    pub duration_ms: u32,
    /// Falling-edge timestamp.
    pub ts: Timestamp,
    /// Turn identifier propagated from the triggering wake event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<TurnId>,
}

/// `wm.audio.capture.start` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureStart {
    /// Resolved mic node — either the configured one when available
    /// or the literal `"default"` if `WM_MIC_NODE` was empty or
    /// fell back to the `PipeWire` default.
    pub mic: String,
    /// Capture sample rate in Hz.
    pub rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Start timestamp.
    pub ts: Timestamp,
}

/// `wm.audio.capture.end` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureEnd {
    /// Outcome — `"ok"` for clean stop, `"error"` for crash/exit-nonzero.
    pub outcome: String,
    /// Wall-clock duration of the capture session, in milliseconds.
    pub dur_ms: u64,
    /// Free-form reason code (exit code, signal name, "shutdown").
    pub reason: String,
    /// End timestamp.
    pub ts: Timestamp,
}

/// `wm.audio.error` payload — emitted on configuration or spawn
/// failures (e.g. `pw-record` binary missing). Mirrors the
/// `wm.tts.error` shape from the sibling PRD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioErrorPayload {
    /// Short machine-readable error class (`pw_record_missing`,
    /// `mic_node_fallback`, `spawn_failed`, …).
    pub kind: String,
    /// Human-readable explanation.
    pub message: String,
    /// Emission timestamp.
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
    /// Capture session began.
    CaptureStart(CaptureStart),
    /// Capture session ended.
    CaptureEnd(CaptureEnd),
    /// Configuration / spawn failure.
    Error(AudioErrorPayload),
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
            Self::CaptureStart(_) => Topics::CAPTURE_START,
            Self::CaptureEnd(_) => Topics::CAPTURE_END,
            Self::Error(_) => Topics::ERROR,
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
            Self::CaptureStart(c) => serde_json::to_value(c),
            Self::CaptureEnd(e) => serde_json::to_value(e),
            Self::Error(e) => serde_json::to_value(e),
        }
    }
}

/// Returns `true` iff `topic` is one of the daemon's own outbound topics.
///
/// Used by the control loop to silently skip echoes of `wm.audio.*`
/// publishes that come back through our `wm.audio.reload` subscription.
/// Mirror of the `wm-tts` defense; see [`Topics::ALL_OUTBOUND`] for the
/// full list.
#[must_use]
pub fn is_self_emitted_topic(topic: &str) -> bool {
    Topics::ALL_OUTBOUND.contains(&topic)
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
            turn_id: None,
        });
        assert_eq!(ev.topic(), "wm.audio.wake");
        let v = ev.payload().ok();
        let v = v.unwrap_or(serde_json::Value::Null);
        assert_eq!(v["wake_word"], "hey-jarvis");
        let conf = v["confidence"].as_f64().unwrap_or(-1.0);
        assert!((conf - 0.87).abs() < 1e-3, "round-trip lost confidence");
        assert_eq!(v["ts"], 42);
        // No turn_id → field absent from serialized payload (backward compat).
        assert!(v.get("turn_id").is_none(), "absent turn_id must not appear in JSON");
    }

    #[test]
    fn speech_chunk_payload_shape() {
        let ev = AudioEvent::SpeechChunk(SpeechChunk {
            seq: 7,
            pcm_b64: "AAAA".into(),
            ts: Timestamp(100),
            turn_id: None,
        });
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        assert_eq!(v["seq"], 7);
        assert_eq!(v["pcm_b64"], "AAAA");
        assert_eq!(v["ts"], 100);
    }

    // ---- TurnId: AC1 (mint/parse, same-ms collision avoidance) ----

    #[test]
    fn turn_id_mint_is_parseable() {
        let id = TurnId::mint();
        let parsed = TurnId::parse(id.as_str());
        assert!(parsed.is_some(), "minted id must round-trip through parse");
        assert_eq!(parsed.as_ref().map(TurnId::as_str), Some(id.as_str()));
    }

    #[test]
    fn turn_id_two_mints_differ() {
        // Two ids minted back-to-back (same or adjacent ms) must be distinct
        // because the seq counter increments.
        let a = TurnId::mint();
        let b = TurnId::mint();
        assert_ne!(a, b, "two minted TurnIds must differ");
    }

    #[test]
    fn turn_id_parse_rejects_garbage() {
        assert!(TurnId::parse("").is_none());
        assert!(TurnId::parse("no-dash-hex").is_none(), "non-hex part must be rejected");
        assert!(TurnId::parse("-abc").is_none(), "empty ms part must be rejected");
        assert!(TurnId::parse("abc-").is_none(), "empty seq part must be rejected");
    }

    #[test]
    fn turn_id_display() {
        let id = TurnId("0000000000001-0000".to_owned());
        assert_eq!(format!("{id}"), "0000000000001-0000");
    }

    #[test]
    fn wake_with_turn_id_serializes_field() {
        let id = TurnId::mint();
        let id_str = id.as_str().to_owned();
        let ev = AudioEvent::Wake(WakeDetected {
            wake_word: "wintermute".into(),
            confidence: 0.99,
            ts: Timestamp(1),
            turn_id: Some(id),
        });
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        assert_eq!(v["turn_id"], id_str, "turn_id must appear when set");
    }

    #[test]
    fn legacy_wake_payload_deserializes_without_turn_id() {
        // AC5: a pre-PRD payload (no turn_id field) must still deserialize cleanly.
        let raw = r#"{"wake_word":"hey-jarvis","confidence":0.9,"ts":100}"#;
        let w: WakeDetected = serde_json::from_str(raw)
            .expect("legacy wake payload (no turn_id) must deserialize");
        assert!(w.turn_id.is_none(), "absent field must map to None");
        assert_eq!(w.wake_word, "hey-jarvis");
    }

    #[test]
    fn legacy_speech_end_payload_deserializes_without_turn_id() {
        // AC5 for SpeechEnd — confirm backward compat for all speech events.
        let raw = r#"{"duration_ms":500,"ts":200}"#;
        let e: SpeechEnd = serde_json::from_str(raw)
            .expect("legacy speech.end payload must deserialize");
        assert!(e.turn_id.is_none());
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
    fn capture_start_payload_shape() {
        let ev = AudioEvent::CaptureStart(CaptureStart {
            mic: "alsa_input.fake".into(),
            rate: 16_000,
            channels: 1,
            ts: Timestamp(7),
        });
        assert_eq!(ev.topic(), "wm.audio.capture.start");
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        assert_eq!(v["mic"], "alsa_input.fake");
        assert_eq!(v["rate"], 16_000);
        assert_eq!(v["channels"], 1);
        assert_eq!(v["ts"], 7);
    }

    #[test]
    fn capture_end_payload_shape() {
        let ev = AudioEvent::CaptureEnd(CaptureEnd {
            outcome: "ok".into(),
            dur_ms: 1234,
            reason: "shutdown".into(),
            ts: Timestamp(8),
        });
        assert_eq!(ev.topic(), "wm.audio.capture.end");
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        assert_eq!(v["outcome"], "ok");
        assert_eq!(v["dur_ms"], 1234);
        assert_eq!(v["reason"], "shutdown");
        assert_eq!(v["ts"], 8);
    }

    #[test]
    fn audio_error_payload_shape() {
        let ev = AudioEvent::Error(AudioErrorPayload {
            kind: "pw_record_missing".into(),
            message: "spawn failed: /nonexistent".into(),
            ts: Timestamp(9),
        });
        assert_eq!(ev.topic(), "wm.audio.error");
        let v = ev.payload().ok().unwrap_or(serde_json::Value::Null);
        assert_eq!(v["kind"], "pw_record_missing");
        assert_eq!(v["ts"], 9);
    }

    #[test]
    fn is_self_emitted_topic_flags_every_outbound() {
        // Sibling-of-wm-tts defense: every Self::topic() variant must be
        // recognised by the dispatch loop's self-echo filter. If a new
        // outbound topic ships without updating Topics::ALL_OUTBOUND,
        // this test trips before the live recursion does.
        for t in Topics::ALL_OUTBOUND {
            assert!(is_self_emitted_topic(t), "outbound topic {t} must be flagged self-emitted");
        }
        // And every AudioEvent variant's topic must also be in the list.
        let canon = [
            AudioEvent::Wake(WakeDetected {
                wake_word: "hey-jarvis".into(),
                confidence: 0.5,
                ts: Timestamp(0),
                turn_id: None,
            })
            .topic(),
            AudioEvent::SpeechStart(SpeechStart { ts: Timestamp(0), turn_id: None }).topic(),
            AudioEvent::SpeechChunk(SpeechChunk {
                seq: 0,
                pcm_b64: String::new(),
                ts: Timestamp(0),
                turn_id: None,
            })
            .topic(),
            AudioEvent::SpeechEnd(SpeechEnd {
                duration_ms: 0,
                ts: Timestamp(0),
                turn_id: None,
            })
            .topic(),
            AudioEvent::Mute { ts: Timestamp(0) }.topic(),
            AudioEvent::Unmute { ts: Timestamp(0) }.topic(),
            AudioEvent::CaptureStart(CaptureStart {
                mic: String::new(),
                rate: 16_000,
                channels: 1,
                ts: Timestamp(0),
            })
            .topic(),
            AudioEvent::CaptureEnd(CaptureEnd {
                outcome: "ok".into(),
                dur_ms: 0,
                reason: String::new(),
                ts: Timestamp(0),
            })
            .topic(),
            AudioEvent::Error(AudioErrorPayload {
                kind: String::new(),
                message: String::new(),
                ts: Timestamp(0),
            })
            .topic(),
        ];
        for t in canon {
            assert!(is_self_emitted_topic(t), "AudioEvent topic {t} must be self-emitted");
        }
    }

    #[test]
    fn is_self_emitted_topic_does_not_flag_inbound() {
        // wm.audio.reload is inbound only — it's a control event the
        // daemon reacts to, not one it publishes.
        assert!(!is_self_emitted_topic(Topics::RELOAD));
        assert!(!is_self_emitted_topic(Topics::TTS_START));
        assert!(!is_self_emitted_topic(Topics::TTS_END));
        assert!(!is_self_emitted_topic(Topics::DIALOG_MUTE_REQ));
        assert!(!is_self_emitted_topic(Topics::DIALOG_UNMUTE_REQ));
        assert!(!is_self_emitted_topic(""));
        assert!(!is_self_emitted_topic("wm.audio."));
        // Substring-not-equal check: capture.start.detail must not match.
        assert!(!is_self_emitted_topic("wm.audio.capture.start.detail"));
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
