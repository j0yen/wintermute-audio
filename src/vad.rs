//! Voice-activity detection: trait + sliding window + edge tracker.
//!
//! PRD §2.3 step 3 calls for Silero VAD inference every 32 ms on a
//! 512-sample window (32 ms at 16 kHz). This module defines the
//! [`VadDetector`] trait the daemon consumes, [`VadWindow`] (a buffer
//! mirroring [`crate::wake::WakeWindow`] at the VAD cadence), a
//! [`NullVadDetector`] baseline, and [`VadEdgeTracker`] — the state
//! machine that turns per-frame VAD outcomes into the
//! `speech.start` / `speech.end` rising / falling edges PRD §2.3
//! publishes on agorabus.
//!
//! Silero VAD ONNX inference plugs in behind [`VadDetector`] in a
//! follow-up iteration without disturbing the topology built here.

/// Sample rate of the canonical mic stream (mirrors
/// [`crate::source::SAMPLE_RATE_HZ`]).
const SAMPLE_RATE_HZ: usize = 16_000;

/// VAD-window size in i16 samples. 32 ms at 16 kHz = 512 samples,
/// matching Silero VAD's expected input.
pub const VAD_WINDOW_SAMPLES: usize = SAMPLE_RATE_HZ * 32 / 1000;

/// VAD-inference stride in i16 samples. PRD §2.3 step 3 calls for
/// inference "every 32 ms"; non-overlapping windows.
pub const VAD_STRIDE_SAMPLES: usize = VAD_WINDOW_SAMPLES;

/// Default per-frame duration in ms, mirroring [`VAD_STRIDE_SAMPLES`]
/// at 16 kHz.
pub const VAD_FRAME_MS: u32 = 32;

/// PRD §2.3 step 3: emit `speech.end` after this much confirmed silence.
pub const SPEECH_END_HANGOVER_MS: u32 = 500;

/// Outcome of one inference call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadOutcome {
    /// No speech detected on this window.
    Silence,
    /// Speech detected. `probability` is the model score (`0.0..=1.0`).
    Speech {
        /// Model probability of speech, `0.0..=1.0`.
        probability: f32,
    },
}

/// A pluggable VAD backend.
///
/// Implementations must be `Send + Sync` so the daemon can share one
/// instance across the capture loop and any future hot-swap path.
pub trait VadDetector: Send + Sync {
    /// Stable kebab-case label identifying this backend.
    fn label(&self) -> &str;

    /// Score one [`VAD_WINDOW_SAMPLES`]-sized window. Implementations
    /// MUST be deterministic on the same input.
    fn process(&self, window: &[i16]) -> VadOutcome;
}

/// Baseline backend: always returns [`VadOutcome::Silence`].
///
/// Lets the daemon wire its full VAD topology before Silero inference
/// lands. Carries its own label string so the trait method's
/// elided-lifetime return is genuinely bound to `self`.
pub struct NullVadDetector {
    label: String,
}

impl NullVadDetector {
    /// New baseline detector labelled `label`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl Default for NullVadDetector {
    fn default() -> Self {
        Self::new("null")
    }
}

impl VadDetector for NullVadDetector {
    fn label(&self) -> &str {
        &self.label
    }

    fn process(&self, _window: &[i16]) -> VadOutcome {
        VadOutcome::Silence
    }
}

/// Sliding sample buffer that drains complete fixed-size windows on
/// demand. Mirrors [`crate::wake::WakeWindow`] for the VAD cadence.
pub struct VadWindow {
    buf: Vec<i16>,
    window_size: usize,
    stride: usize,
}

impl VadWindow {
    /// Construct a VAD-window buffer.
    ///
    /// `window_size == 0` or `stride == 0` is treated as a safe no-op:
    /// [`next_window`](Self::next_window) returns `None` for any input.
    #[must_use]
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self {
            buf: Vec::with_capacity(window_size.saturating_mul(2)),
            window_size,
            stride,
        }
    }

    /// Convenience constructor with the daemon's defaults.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(VAD_WINDOW_SAMPLES, VAD_STRIDE_SAMPLES)
    }

    /// Append samples to the internal buffer.
    pub fn push(&mut self, samples: &[i16]) {
        self.buf.extend_from_slice(samples);
    }

    /// How many samples are currently buffered.
    #[must_use]
    pub const fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Pop the next complete window, or `None` if not enough samples
    /// are buffered yet. On `Some` return, the internal buffer has
    /// advanced by `stride` samples.
    pub fn next_window(&mut self) -> Option<Vec<i16>> {
        if self.window_size == 0 || self.stride == 0 {
            return None;
        }
        let window = self.buf.get(..self.window_size)?.to_vec();
        let to_drop = self.stride.min(self.buf.len());
        self.buf.drain(..to_drop);
        Some(window)
    }

    /// Drop all buffered samples. Used at mute-edge transitions so the
    /// next inference does not run on stale audio.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

/// Edge transition the tracker emitted on a given step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechEdge {
    /// Rising edge — speech just started.
    Start,
    /// Falling edge — sustained silence for at least
    /// [`VadEdgeTracker::hangover_ms`].
    ///
    /// `duration_ms` measures speech-frame time accumulated *before*
    /// the silence run began (i.e. the actual utterance length, not
    /// including the hangover gap), matching what `wm.audio.speech.end`
    /// publishes per PRD §2.3.
    End {
        /// Speech duration in ms.
        duration_ms: u32,
    },
}

/// State machine: [`VadOutcome`] stream → [`SpeechEdge`] transitions.
///
/// Drives rising-edge → sustained-speech → confirmed-silence → falling-edge.
/// Caller feeds [`VadOutcome`]s from successive frames; the tracker
/// returns one [`SpeechEdge`] (or `None`) per step. Time advances by
/// `frame_duration_ms` per call.
pub struct VadEdgeTracker {
    speech_threshold: f32,
    frame_duration_ms: u32,
    hangover_ms: u32,
    state: TrackerState,
}

#[derive(Debug, Clone, Copy)]
enum TrackerState {
    Silent,
    InSpeech {
        speech_ms: u32,
    },
    PendingEnd {
        speech_ms_at_silence_start: u32,
        silence_run_ms: u32,
    },
}

impl VadEdgeTracker {
    /// Construct with caller-supplied parameters.
    ///
    /// `speech_threshold` is the minimum probability for a
    /// [`VadOutcome::Speech`] frame to count as speech.
    /// `frame_duration_ms` is the per-frame timestep advance (use
    /// [`VAD_FRAME_MS`] for the default 32 ms cadence).
    /// `hangover_ms` is how much consecutive silence must accumulate
    /// before [`SpeechEdge::End`] fires (PRD: 500 ms).
    #[must_use]
    pub const fn new(speech_threshold: f32, frame_duration_ms: u32, hangover_ms: u32) -> Self {
        Self {
            speech_threshold,
            frame_duration_ms,
            hangover_ms,
            state: TrackerState::Silent,
        }
    }

    /// Default tracker: 0.5 threshold, [`VAD_FRAME_MS`] frames,
    /// [`SPEECH_END_HANGOVER_MS`] hangover.
    #[must_use]
    pub const fn with_defaults() -> Self {
        Self::new(0.5, VAD_FRAME_MS, SPEECH_END_HANGOVER_MS)
    }

    /// Feed one outcome; return any edge transition it produced.
    pub fn step(&mut self, outcome: VadOutcome) -> Option<SpeechEdge> {
        let is_speech = matches!(
            outcome,
            VadOutcome::Speech { probability } if probability >= self.speech_threshold
        );
        match (self.state, is_speech) {
            (TrackerState::Silent, false) => None,
            (TrackerState::Silent, true) => {
                self.state = TrackerState::InSpeech {
                    speech_ms: self.frame_duration_ms,
                };
                Some(SpeechEdge::Start)
            }
            (TrackerState::InSpeech { speech_ms }, true) => {
                self.state = TrackerState::InSpeech {
                    speech_ms: speech_ms.saturating_add(self.frame_duration_ms),
                };
                None
            }
            (TrackerState::InSpeech { speech_ms }, false) => {
                if self.frame_duration_ms >= self.hangover_ms {
                    self.state = TrackerState::Silent;
                    Some(SpeechEdge::End {
                        duration_ms: speech_ms,
                    })
                } else {
                    self.state = TrackerState::PendingEnd {
                        speech_ms_at_silence_start: speech_ms,
                        silence_run_ms: self.frame_duration_ms,
                    };
                    None
                }
            }
            (
                TrackerState::PendingEnd {
                    speech_ms_at_silence_start,
                    silence_run_ms,
                },
                false,
            ) => {
                let new_silence = silence_run_ms.saturating_add(self.frame_duration_ms);
                if new_silence >= self.hangover_ms {
                    self.state = TrackerState::Silent;
                    Some(SpeechEdge::End {
                        duration_ms: speech_ms_at_silence_start,
                    })
                } else {
                    self.state = TrackerState::PendingEnd {
                        speech_ms_at_silence_start,
                        silence_run_ms: new_silence,
                    };
                    None
                }
            }
            (
                TrackerState::PendingEnd {
                    speech_ms_at_silence_start,
                    silence_run_ms,
                },
                true,
            ) => {
                let resumed = speech_ms_at_silence_start
                    .saturating_add(silence_run_ms)
                    .saturating_add(self.frame_duration_ms);
                self.state = TrackerState::InSpeech { speech_ms: resumed };
                None
            }
        }
    }

    /// Force a reset — caller may use at mute edges or hot-swap so the
    /// next utterance does not inherit stale state.
    pub const fn reset(&mut self) {
        self.state = TrackerState::Silent;
    }

    /// `true` while mid-utterance (post-[`SpeechEdge::Start`],
    /// pre-[`SpeechEdge::End`]).
    #[must_use]
    pub const fn in_speech(&self) -> bool {
        matches!(
            self.state,
            TrackerState::InSpeech { .. } | TrackerState::PendingEnd { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_constants_match_prd() {
        assert_eq!(VAD_WINDOW_SAMPLES, 512);
        assert_eq!(VAD_STRIDE_SAMPLES, 512);
        assert_eq!(VAD_FRAME_MS, 32);
        assert_eq!(SPEECH_END_HANGOVER_MS, 500);
    }

    #[test]
    fn null_detector_returns_silence() {
        let d = NullVadDetector::default();
        let win = vec![0_i16; VAD_WINDOW_SAMPLES];
        assert_eq!(d.process(&win), VadOutcome::Silence);
        assert_eq!(d.label(), "null");
    }

    #[test]
    fn null_detector_label_round_trip() {
        let d = NullVadDetector::new("silero-v1");
        assert_eq!(d.label(), "silero-v1");
    }

    #[test]
    fn window_buffers_until_full() {
        let mut w = VadWindow::new(4, 4);
        w.push(&[1, 2, 3]);
        assert!(w.next_window().is_none());
        assert_eq!(w.buffered(), 3);
        w.push(&[4]);
        assert_eq!(w.next_window(), Some(vec![1, 2, 3, 4]));
        assert_eq!(w.buffered(), 0);
    }

    #[test]
    fn window_drains_multiple_in_a_row() {
        let mut w = VadWindow::new(2, 2);
        w.push(&[10, 20, 30, 40, 50]);
        assert_eq!(w.next_window(), Some(vec![10, 20]));
        assert_eq!(w.next_window(), Some(vec![30, 40]));
        assert_eq!(w.next_window(), None);
        assert_eq!(w.buffered(), 1);
    }

    #[test]
    fn window_clear_drops_buffer() {
        let mut w = VadWindow::new(2, 2);
        w.push(&[1, 2, 3]);
        w.clear();
        assert_eq!(w.buffered(), 0);
        assert!(w.next_window().is_none());
    }

    #[test]
    fn window_zero_size_is_safe_noop() {
        let mut w = VadWindow::new(0, 0);
        w.push(&[1, 2, 3]);
        assert!(w.next_window().is_none());
    }

    #[test]
    fn tracker_rises_on_first_speech_frame() {
        let mut t = VadEdgeTracker::with_defaults();
        assert_eq!(t.step(VadOutcome::Silence), None);
        assert!(!t.in_speech());
        assert_eq!(
            t.step(VadOutcome::Speech { probability: 0.9 }),
            Some(SpeechEdge::Start)
        );
        assert!(t.in_speech());
    }

    #[test]
    fn tracker_below_threshold_counts_as_silence() {
        let mut t = VadEdgeTracker::with_defaults();
        assert_eq!(t.step(VadOutcome::Speech { probability: 0.1 }), None);
        assert!(!t.in_speech());
    }

    #[test]
    fn tracker_falls_after_full_hangover_silence() {
        let mut t = VadEdgeTracker::with_defaults();
        assert_eq!(
            t.step(VadOutcome::Speech { probability: 0.9 }),
            Some(SpeechEdge::Start)
        );
        for _ in 0..9 {
            assert_eq!(t.step(VadOutcome::Speech { probability: 0.8 }), None);
        }
        // 10 speech frames × 32 ms = 320 ms accumulated speech.
        // 15 silence frames × 32 ms = 480 ms < 500 ms hangover → no fall yet.
        for i in 0..15 {
            assert_eq!(t.step(VadOutcome::Silence), None, "frame {i}");
        }
        // 16th silence frame pushes the run to 512 ms ≥ 500 → End.
        assert_eq!(
            t.step(VadOutcome::Silence),
            Some(SpeechEdge::End { duration_ms: 320 })
        );
        assert!(!t.in_speech());
    }

    #[test]
    fn tracker_brief_silence_does_not_end_speech() {
        let mut t = VadEdgeTracker::with_defaults();
        assert_eq!(
            t.step(VadOutcome::Speech { probability: 0.9 }),
            Some(SpeechEdge::Start)
        );
        for _ in 0..5 {
            assert_eq!(t.step(VadOutcome::Silence), None);
        }
        assert!(t.in_speech());
        assert_eq!(t.step(VadOutcome::Speech { probability: 0.8 }), None);
        assert!(t.in_speech());
    }

    #[test]
    fn tracker_reset_returns_to_silent() {
        let mut t = VadEdgeTracker::with_defaults();
        let _ = t.step(VadOutcome::Speech { probability: 0.9 });
        assert!(t.in_speech());
        t.reset();
        assert!(!t.in_speech());
        assert_eq!(t.step(VadOutcome::Silence), None);
    }

    #[test]
    fn tracker_short_hangover_ends_on_first_silence_frame() {
        // hangover_ms ≤ frame_duration_ms → fall fires on the first
        // silence frame after speech (no PendingEnd dwell).
        let mut t = VadEdgeTracker::new(0.5, 32, 16);
        assert_eq!(
            t.step(VadOutcome::Speech { probability: 0.9 }),
            Some(SpeechEdge::Start)
        );
        assert_eq!(t.step(VadOutcome::Speech { probability: 0.9 }), None);
        assert_eq!(
            t.step(VadOutcome::Silence),
            Some(SpeechEdge::End { duration_ms: 64 })
        );
        assert!(!t.in_speech());
    }
}
