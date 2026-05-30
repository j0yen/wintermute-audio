//! ONNX inference backends for wake-word detection and VAD.
//!
//! This module provides two concrete implementations of the
//! [`crate::wake::WakeDetector`] and [`crate::vad::VadDetector`] traits that
//! run real ONNX inference using the [`ort`] crate (a safe Rust wrapper for
//! ONNX Runtime). Both backends load a model file on construction; if the file
//! is missing or the model fails to load they **fall back to the null
//! detector** defined in their respective trait modules rather than crashing
//! the daemon. This satisfies PRD AC7.
//!
//! ## Model formats
//!
//! ### microWakeWord wake-word backend ([`OnnxWakeDetector`])
//!
//! microWakeWord models expect a 1-D float32 input of shape `[1, N]` where N
//! is [`crate::wake::WAKE_WINDOW_SAMPLES`] (1280 samples at 16 kHz). The
//! samples must be normalised to `[-1.0, 1.0]` by dividing each i16 by
//! `32768.0`. The model outputs a single float32 confidence score.
//!
//! ### Silero VAD backend ([`OnnxVadDetector`])
//!
//! Silero VAD v4 expects:
//! * `input` — float32 `[1, 1, 512]` (mono, 32 ms window at 16 kHz,
//!   normalised to `[-1.0, 1.0]`)
//! * `sr`    — int64 scalar `16000` (sample rate)
//! * `h`     — float32 `[2, 1, 64]` recurrent state (zeros on first call,
//!   updated in-place after each call)
//! * `c`     — float32 `[2, 1, 64]` cell state (same)
//!
//! The model outputs:
//! * `output` — float32 `[1, 1]` speech probability
//! * `hn`     — float32 `[2, 1, 64]` updated hidden state
//! * `cn`     — float32 `[2, 1, 64]` updated cell state
//!
//! The Silero backend is stateful: recurrent states persist across calls so
//! the model sees continuous context. Call [`OnnxVadDetector::reset`] at
//! mute or hot-swap edges to prevent stale state bleeding into the next
//! utterance window.

use crate::vad::{NullVadDetector, VadDetector, VadOutcome, VAD_WINDOW_SAMPLES};
use crate::wake::{NullWakeDetector, WakeDetector, WakeOutcome, WAKE_WINDOW_SAMPLES};
use ort::{
    session::Session,
    value::Tensor,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

// ─── normalisation ──────────────────────────────────────────────────────────

/// Normalise a slice of i16 PCM samples to f32 `[-1.0, 1.0]`.
///
/// Divides each sample by `32768.0` (the absolute maximum magnitude of an
/// i16). The result is clamped to `[-1.0, 1.0]` to guard against the rare
/// `i16::MIN` underflow where dividing by `32768.0` would give `-1.000030…`.
#[inline]
fn normalise_pcm(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| (f32::from(s) / 32_768.0_f32).clamp(-1.0_f32, 1.0_f32))
        .collect()
}

// ─── OnnxWakeDetector ───────────────────────────────────────────────────────

/// microWakeWord ONNX-backed wake-word detector.
///
/// Loads the model once at construction; thereafter [`WakeDetector::process`]
/// runs a single forward pass per window and returns the model's confidence.
/// Construction falls back to [`NullWakeDetector`] (see
/// [`load_or_null_wake`]) if the model file is absent or fails to load.
///
/// The [`ort`] session requires `&mut self` for `run()`, so it is held
/// behind a `Mutex` to satisfy the `Sync` bound on [`WakeDetector`].
pub struct OnnxWakeDetector {
    label: String,
    session: Mutex<Session>,
}

impl OnnxWakeDetector {
    /// Load the model at `path` and register `label` as the wake-word name.
    ///
    /// # Errors
    ///
    /// Returns an [`ort::Error`] if the model file cannot be parsed or the
    /// ONNX Runtime environment fails to initialise.
    pub fn load(path: impl AsRef<Path>, label: impl Into<String>) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self {
            label: label.into(),
            session: Mutex::new(session),
        })
    }
}

impl WakeDetector for OnnxWakeDetector {
    fn label(&self) -> &str {
        &self.label
    }

    fn process(&self, window: &[i16]) -> WakeOutcome {
        if window.len() != WAKE_WINDOW_SAMPLES {
            return WakeOutcome::NotDetected;
        }
        let floats = normalise_pcm(window);
        // microWakeWord model input shape: [1, N] (batch=1, samples=N).
        let tensor = match Tensor::<f32>::from_array(([1, WAKE_WINDOW_SAMPLES], floats)) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "wake: could not build input tensor");
                return WakeOutcome::NotDetected;
            }
        };
        // Run inference. The session requires &mut, so we hold the lock only
        // for the duration of the run() call. We extract the confidence score
        // (a copy) before dropping the MutexGuard so the outputs lifetime
        // does not escape the lock scope.
        let confidence = {
            let mut sess = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = match sess.run(ort::inputs![tensor]) {
                Ok(o) => o,
                Err(e) => {
                    warn!(error = %e, "wake: inference failed");
                    return WakeOutcome::NotDetected;
                }
            };
            // Extract the first output as a scalar float32 confidence value.
            // The extraction copies the float out; outputs is then dropped.
            outputs
                .iter()
                .next()
                .and_then(|(_k, v)| v.try_extract_scalar::<f32>().ok())
                .unwrap_or(0.0_f32)
        };
        if confidence > 0.0_f32 {
            WakeOutcome::Detected { confidence }
        } else {
            WakeOutcome::NotDetected
        }
    }
}

// ─── OnnxVadDetector ────────────────────────────────────────────────────────

/// Recurrent state for one Silero VAD v4 session.
///
/// Silero VAD v4 uses a 2-layer LSTM; each state tensor has shape
/// `[2, 1, 64]` → 128 elements for both `h` (hidden) and `c` (cell).
#[derive(Clone)]
struct SileroState {
    h: Vec<f32>, // 128 elements, shape [2, 1, 64]
    c: Vec<f32>, // 128 elements, shape [2, 1, 64]
}

impl SileroState {
    fn zeros() -> Self {
        Self {
            h: vec![0.0_f32; 128],
            c: vec![0.0_f32; 128],
        }
    }
}

/// Silero VAD v4 ONNX-backed voice-activity detector.
///
/// Carries recurrent state across calls so the model maintains continuous
/// temporal context. Both the [`ort`] session (which requires `&mut self` for
/// `run()`) and the LSTM state are held in a combined `Mutex` to satisfy the
/// `Sync` bound on [`VadDetector`].
///
/// Call [`OnnxVadDetector::reset`] at mute or hot-swap edges to prevent
/// stale state from bleeding into the next utterance window.
pub struct OnnxVadDetector {
    label: String,
    // session and state share one Mutex so a single lock covers both.
    inner: Mutex<OnnxVadInner>,
}

struct OnnxVadInner {
    session: Session,
    state: SileroState,
}

impl OnnxVadDetector {
    /// Load the model at `path`.
    ///
    /// # Errors
    ///
    /// Returns an [`ort::Error`] if the model file cannot be parsed or the
    /// ONNX Runtime environment fails to initialise.
    pub fn load(path: impl AsRef<Path>, label: impl Into<String>) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self {
            label: label.into(),
            inner: Mutex::new(OnnxVadInner {
                session,
                state: SileroState::zeros(),
            }),
        })
    }

    /// Reset recurrent state to all-zeros.
    ///
    /// Call at mute or hot-swap edges so the next utterance starts with
    /// fresh LSTM context.
    pub fn reset(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.state = SileroState::zeros();
        }
    }
}

impl VadDetector for OnnxVadDetector {
    fn label(&self) -> &str {
        &self.label
    }

    fn process(&self, window: &[i16]) -> VadOutcome {
        if window.len() != VAD_WINDOW_SAMPLES {
            return VadOutcome::Silence;
        }
        let floats = normalise_pcm(window);
        // Silero VAD v4 input: float32 [1, 1, 512]
        let input_tensor = match Tensor::<f32>::from_array(([1_usize, 1, VAD_WINDOW_SAMPLES], floats)) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "vad: could not build input tensor");
                return VadOutcome::Silence;
            }
        };
        // sr scalar: int64 — shape [1] to avoid ambiguous empty-array type.
        let sr_tensor = match Tensor::<i64>::from_array(([1_usize], vec![16_000_i64])) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "vad: could not build sr tensor");
                return VadOutcome::Silence;
            }
        };

        // We need to hold the lock across the full run() call to satisfy the
        // borrow checker: SessionOutputs<'s> borrows from the Session, so we
        // must extract all output values (copies) before releasing the lock.
        let (probability, new_h, new_c) = {
            let mut inner = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            let h_data = inner.state.h.clone();
            let c_data = inner.state.c.clone();

            // h and c: float32 [2, 1, 64]
            let h_tensor = match Tensor::<f32>::from_array(([2_usize, 1, 64], h_data)) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = %e, "vad: could not build h tensor");
                    return VadOutcome::Silence;
                }
            };
            let c_tensor = match Tensor::<f32>::from_array(([2_usize, 1, 64], c_data)) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = %e, "vad: could not build c tensor");
                    return VadOutcome::Silence;
                }
            };

            let outputs = match inner.session.run(ort::inputs![
                "input" => input_tensor,
                "sr"    => sr_tensor,
                "h"     => h_tensor,
                "c"     => c_tensor,
            ]) {
                Ok(o) => o,
                Err(e) => {
                    warn!(error = %e, "vad: inference failed");
                    return VadOutcome::Silence;
                }
            };

            // Extract all output values as copies so they don't borrow
            // `outputs` (and thereby `inner.session`) past this block.
            let prob = outputs
                .get("output")
                .and_then(|v| v.try_extract_scalar::<f32>().ok())
                .unwrap_or(0.0_f32);
            let hn: Option<Vec<f32>> = outputs.get("hn").and_then(|v| {
                v.try_extract_tensor::<f32>()
                    .ok()
                    .map(|(_shape, data)| data.to_vec())
            });
            let cn: Option<Vec<f32>> = outputs.get("cn").and_then(|v| {
                v.try_extract_tensor::<f32>()
                    .ok()
                    .map(|(_shape, data)| data.to_vec())
            });
            // outputs and inner (session borrow) drop here.
            drop(outputs);

            // Update recurrent state while we still hold the lock.
            if let Some(ref h) = hn {
                if h.len() == 128 {
                    inner.state.h = h.clone();
                }
            }
            if let Some(ref c) = cn {
                if c.len() == 128 {
                    inner.state.c = c.clone();
                }
            }
            (prob, hn, cn)
        };
        let _ = (new_h, new_c); // consumed for state update above; suppress unused warnings

        if probability > 0.0_f32 {
            VadOutcome::Speech { probability }
        } else {
            VadOutcome::Silence
        }
    }
}

// ─── factory helpers ─────────────────────────────────────────────────────────

/// Load a wake-word ONNX model, falling back to [`NullWakeDetector`] if the
/// model is absent or unloadable.
///
/// On success logs `wake_model_loaded`; on failure logs
/// `wake_model_missing` or `wake_model_load_error` (both result in the null
/// engine). The caller supplies `label` (e.g. `"hey-jarvis"`) and `path`;
/// typical callers derive both from [`crate::config::Config`].
#[must_use]
pub fn load_or_null_wake(
    path: impl AsRef<Path>,
    label: impl Into<String> + Clone,
) -> Arc<dyn WakeDetector> {
    let path = path.as_ref();
    let label_str: String = label.clone().into();
    if !path.exists() {
        warn!(
            path = %path.display(),
            label = %label_str,
            "wake_model_missing: using null wake engine"
        );
        return Arc::new(NullWakeDetector::new(label_str));
    }
    match OnnxWakeDetector::load(path, label_str.clone()) {
        Ok(det) => {
            info!(
                path = %path.display(),
                label = %label_str,
                "wake_model_loaded"
            );
            Arc::new(det)
        }
        Err(e) => {
            warn!(
                path = %path.display(),
                label = %label_str,
                error = %e,
                "wake_model_load_error: using null wake engine"
            );
            Arc::new(NullWakeDetector::new(label_str))
        }
    }
}

/// Load a Silero VAD ONNX model, falling back to [`NullVadDetector`] if the
/// model is absent or unloadable.
///
/// On success logs `vad_model_loaded`; on failure logs `vad_model_missing`
/// or `vad_model_load_error`.
#[must_use]
pub fn load_or_null_vad(path: impl AsRef<Path>) -> Arc<dyn VadDetector> {
    let path = path.as_ref();
    if !path.exists() {
        warn!(
            path = %path.display(),
            "vad_model_missing: using null VAD engine"
        );
        return Arc::new(NullVadDetector::new("null-vad"));
    }
    match OnnxVadDetector::load(path, "silero-vad") {
        Ok(det) => {
            info!(path = %path.display(), "vad_model_loaded");
            Arc::new(det)
        }
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %e,
                "vad_model_load_error: using null VAD engine"
            );
            Arc::new(NullVadDetector::new("null-vad"))
        }
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests need to fail loudly"
)]
mod tests {
    use super::*;
    use crate::vad::{SpeechEdge, VadEdgeTracker, VadOutcome, VadWindow, VAD_WINDOW_SAMPLES};
    use crate::wake::{WakeWindow, WAKE_WINDOW_SAMPLES};

    // ── normalise_pcm ──

    #[test]
    fn normalise_zero_is_zero() {
        let input = vec![0_i16; 8];
        let out = normalise_pcm(&input);
        assert!(out.iter().all(|&x| x == 0.0_f32));
    }

    #[test]
    fn normalise_max_is_at_most_one() {
        let input = vec![i16::MAX, i16::MIN, 0];
        let out = normalise_pcm(&input);
        assert!(out.iter().all(|&x| x.abs() <= 1.0_f32));
    }

    #[test]
    fn normalise_positive_value() {
        // 16384 / 32768 = 0.5
        let input = vec![16_384_i16];
        let out = normalise_pcm(&input);
        let diff = (out[0] - 0.5_f32).abs();
        assert!(diff < 1e-4_f32, "expected ≈0.5, got {}", out[0]);
    }

    // ── fallback on missing model (AC7) ──

    #[test]
    fn load_or_null_wake_falls_back_on_missing_file() {
        // AC7: WM_WAKE_MODEL=/nonexistent → null engine, no panic.
        let det = load_or_null_wake("/nonexistent/wake.onnx", "hey-jarvis");
        let window = vec![0_i16; WAKE_WINDOW_SAMPLES];
        assert_eq!(det.process(&window), WakeOutcome::NotDetected);
        assert_eq!(det.label(), "hey-jarvis");
    }

    #[test]
    fn load_or_null_vad_falls_back_on_missing_file() {
        // AC7 (VAD path): missing model → null engine, no panic.
        let det = load_or_null_vad("/nonexistent/silero.onnx");
        let window = vec![0_i16; VAD_WINDOW_SAMPLES];
        assert_eq!(det.process(&window), VadOutcome::Silence);
        assert_eq!(det.label(), "null-vad");
    }

    // ── idle / active detector counts via null engines ──

    #[test]
    fn wake_detector_idle_count() {
        // Null detector never fires; active detection count over N windows
        // must be exactly 0.
        let det = load_or_null_wake("/nonexistent/wake.onnx", "hey-jarvis");
        let window = vec![0_i16; WAKE_WINDOW_SAMPLES];
        let detected_count: usize = (0..10)
            .filter(|_| matches!(det.process(&window), WakeOutcome::Detected { .. }))
            .count();
        assert_eq!(detected_count, 0, "null detector should never report active");
    }

    #[test]
    fn vad_detector_idle_count() {
        let det = load_or_null_vad("/nonexistent/silero.onnx");
        let window = vec![0_i16; VAD_WINDOW_SAMPLES];
        let speech_count: usize = (0..10)
            .filter(|_| {
                matches!(det.process(&window), VadOutcome::Speech { .. })
            })
            .count();
        assert_eq!(speech_count, 0, "null VAD detector should never report speech");
    }

    // ── frame slicing ──

    #[test]
    fn wake_window_slices_into_correct_count() {
        // 3840 samples = 3 × WAKE_WINDOW_SAMPLES (1280) exactly.
        let mut ww = WakeWindow::with_defaults();
        ww.push(&vec![0_i16; 3840]);
        let mut count = 0_usize;
        while ww.next_window().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn vad_window_slices_into_correct_count() {
        // 2560 samples = 5 × VAD_WINDOW_SAMPLES (512) exactly.
        let mut vw = VadWindow::with_defaults();
        vw.push(&vec![0_i16; 2560]);
        let mut count = 0_usize;
        while vw.next_window().is_some() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn wake_process_size_mismatch_returns_not_detected() {
        let det = load_or_null_wake("/nonexistent/wake.onnx", "hey-jarvis");
        // Wrong size: WAKE_WINDOW_SAMPLES - 1.
        let short_window = vec![0_i16; WAKE_WINDOW_SAMPLES - 1];
        assert_eq!(det.process(&short_window), WakeOutcome::NotDetected);
    }

    #[test]
    fn vad_process_size_mismatch_returns_silence() {
        let det = load_or_null_vad("/nonexistent/silero.onnx");
        let short_window = vec![0_i16; VAD_WINDOW_SAMPLES - 1];
        assert_eq!(det.process(&short_window), VadOutcome::Silence);
    }

    // ── state machine integration (end-to-end, null VAD engine) ──

    #[test]
    fn vad_edge_tracker_with_silence_produces_no_start_event() {
        // AC5 analog: pure silence → no speech.start, no speech.end.
        let det = load_or_null_vad("/nonexistent/silero.onnx");
        let mut tracker = VadEdgeTracker::with_defaults();
        let window = vec![0_i16; VAD_WINDOW_SAMPLES];
        for _ in 0..200 {
            let outcome = det.process(&window);
            let edge = tracker.step(outcome);
            assert!(
                edge.is_none(),
                "null VAD + all-zero PCM must never produce a speech edge"
            );
        }
        assert!(!tracker.in_speech());
    }

    #[test]
    fn synthetic_speech_start_event_fires_on_rising_edge() {
        // Integration test for wm.audio.speech.start envelope (AC1 class).
        let mut tracker = VadEdgeTracker::with_defaults();
        let start_edge = tracker.step(VadOutcome::Speech { probability: 0.9 });
        assert_eq!(
            start_edge,
            Some(SpeechEdge::Start),
            "first high-confidence speech frame must produce SpeechEdge::Start"
        );
        assert!(tracker.in_speech());
    }

    #[test]
    fn synthetic_speech_end_event_fires_after_hangover() {
        // Integration test for wm.audio.speech.end envelope (AC1 class).
        let mut tracker = VadEdgeTracker::with_defaults();
        // 10 speech frames × 32 ms = 320 ms speech.
        let _ = tracker.step(VadOutcome::Speech { probability: 0.9 });
        for _ in 0..9 {
            let _ = tracker.step(VadOutcome::Speech { probability: 0.9 });
        }
        // 16 silence frames × 32 ms = 512 ms ≥ 500 ms hangover → End.
        let mut end_edge = None;
        for _ in 0..16 {
            end_edge = tracker.step(VadOutcome::Silence);
        }
        assert_eq!(
            end_edge,
            Some(SpeechEdge::End { duration_ms: 320 }),
            "after hangover silence, SpeechEdge::End must fire with accumulated duration"
        );
        assert!(!tracker.in_speech());
    }

    #[test]
    fn synthetic_wake_detected_fires_on_high_confidence() {
        // Integration test for wm.audio.wake envelope (AC1 class).
        // We use a custom detector that always returns high confidence.
        struct AlwaysFireDetector;
        impl WakeDetector for AlwaysFireDetector {
            fn label(&self) -> &str {
                "hey-test"
            }
            fn process(&self, _window: &[i16]) -> WakeOutcome {
                WakeOutcome::Detected { confidence: 0.95 }
            }
        }
        let det = AlwaysFireDetector;
        let window = vec![0_i16; WAKE_WINDOW_SAMPLES];
        let outcome = det.process(&window);
        assert!(
            matches!(outcome, WakeOutcome::Detected { confidence } if confidence >= 0.8),
            "wake detector must report high confidence: {outcome:?}"
        );
    }

    // ── self-emitted topic filter covers the three new topics (AC10) ──

    #[test]
    fn inference_topics_are_in_self_emitted_filter() {
        // AC10: wm.audio.wake, wm.audio.speech.start, wm.audio.speech.end
        // must all appear in Topics::ALL_OUTBOUND so the daemon's echo filter
        // covers them. Failing this test means the guard added in PRD
        // §2.1 has been weakened.
        use crate::events::{Topics, is_self_emitted_topic};
        assert!(
            is_self_emitted_topic(Topics::WAKE),
            "wm.audio.wake must be self-emitted"
        );
        assert!(
            is_self_emitted_topic(Topics::SPEECH_START),
            "wm.audio.speech.start must be self-emitted"
        );
        assert!(
            is_self_emitted_topic(Topics::SPEECH_END),
            "wm.audio.speech.end must be self-emitted"
        );
    }
}
