//! Wake-word detection: trait + sliding window + null baseline.
//!
//! PRD §2.3 step 3 calls for `microWakeWord` ONNX inference every
//! 80 ms on a 1280-sample window (80 ms at 16 kHz). This module
//! defines the [`WakeDetector`] trait the daemon consumes plus the
//! [`WakeWindow`] buffer that slices the canonical 16 kHz mono PCM
//! into fixed-size, fixed-stride windows for the detector.
//!
//! The default backend ([`NullWakeDetector`]) never fires, so the
//! daemon stays compilable and integration-testable without the
//! `ort` / model binary dependency. The real `microWakeWord` impl
//! plugs in behind this trait in a follow-up iteration.

/// Sample rate of the canonical mic stream (mirrors
/// [`crate::source::SAMPLE_RATE_HZ`]).
const SAMPLE_RATE_HZ: usize = 16_000;

/// Wake-window size in i16 samples. 80 ms at 16 kHz = 1280 samples —
/// the input shape `microWakeWord`'s shipped models expect.
pub const WAKE_WINDOW_SAMPLES: usize = SAMPLE_RATE_HZ * 80 / 1000;

/// Wake-inference stride in i16 samples. PRD §2.3 step 3 calls for
/// inference "every 80 ms", i.e. non-overlapping windows.
pub const WAKE_STRIDE_SAMPLES: usize = WAKE_WINDOW_SAMPLES;

/// Outcome of one inference call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WakeOutcome {
    /// No wake fired on this window.
    NotDetected,
    /// Wake fired. `confidence` is the model's per-window score
    /// (`0.0..=1.0`). The daemon's publish gate compares this against
    /// [`crate::Config::wake_threshold`].
    Detected {
        /// Model confidence, `0.0..=1.0`.
        confidence: f32,
    },
}

/// A pluggable wake-word backend.
///
/// Implementations must be `Send + Sync` so the daemon can share one
/// instance across the capture loop and any future hot-swap path.
pub trait WakeDetector: Send + Sync {
    /// Stable kebab-case label this backend recognises. Published as
    /// the `wake_word` field of `wm.audio.wake`.
    fn label(&self) -> &str;

    /// Score one [`WAKE_WINDOW_SAMPLES`]-sized window. Implementations
    /// MUST be deterministic on the same input.
    fn process(&self, window: &[i16]) -> WakeOutcome;
}

/// Baseline backend: always returns [`WakeOutcome::NotDetected`].
///
/// Lets the daemon wire its full wake topology before ONNX inference
/// lands. Carries the configured wake-word label so published events
/// (none, in practice) would still name the right model.
pub struct NullWakeDetector {
    label: String,
}

impl NullWakeDetector {
    /// New baseline detector labelled `label`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl WakeDetector for NullWakeDetector {
    fn label(&self) -> &str {
        &self.label
    }

    fn process(&self, _window: &[i16]) -> WakeOutcome {
        WakeOutcome::NotDetected
    }
}

/// Sliding sample buffer that drains complete fixed-size windows on
/// demand.
///
/// Push raw PCM samples in any quantum; pull
/// `(window_size)`-sample views back out, advanced by `stride` per
/// pop. Internal allocation is a single `Vec<i16>`; `next_window`
/// drains the stride from the front so steady-state memory is
/// bounded by `window_size + max_push_quantum`.
pub struct WakeWindow {
    buf: Vec<i16>,
    window_size: usize,
    stride: usize,
}

impl WakeWindow {
    /// Construct a window buffer.
    ///
    /// `window_size` is how many samples each call to
    /// [`next_window`](Self::next_window) returns. `stride` is how
    /// far the buffer advances after each pop — `stride == window_size`
    /// gives non-overlapping windows; `stride < window_size` gives
    /// overlap.
    ///
    /// # Panics
    ///
    /// Returns a zero-size buffer if `window_size == 0` or
    /// `stride == 0`; the construction itself does not panic. Callers
    /// SHOULD pass non-zero values.
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
        Self::new(WAKE_WINDOW_SAMPLES, WAKE_STRIDE_SAMPLES)
    }

    /// Append samples to the internal buffer.
    pub fn push(&mut self, samples: &[i16]) {
        self.buf.extend_from_slice(samples);
    }

    /// How many samples are currently buffered.
    #[must_use]
    pub fn buffered(&self) -> usize {
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

    /// Drop all buffered samples. Used at mute-edge transitions so
    /// the next inference doesn't run on stale audio.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]

    use super::*;

    #[test]
    fn window_size_matches_prd() {
        // PRD §2.3 step 3 wake: 80 ms at 16 kHz.
        assert_eq!(WAKE_WINDOW_SAMPLES, 1280);
        assert_eq!(WAKE_STRIDE_SAMPLES, 1280);
    }

    #[test]
    fn null_detector_never_fires() {
        let d = NullWakeDetector::new("hey-jarvis");
        assert_eq!(d.label(), "hey-jarvis");
        let win = vec![0_i16; WAKE_WINDOW_SAMPLES];
        assert_eq!(d.process(&win), WakeOutcome::NotDetected);
    }

    #[test]
    fn window_buffers_until_full() {
        let mut w = WakeWindow::new(4, 4);
        w.push(&[1, 2, 3]);
        assert!(w.next_window().is_none());
        assert_eq!(w.buffered(), 3);
        w.push(&[4]);
        let got = w.next_window();
        assert_eq!(got, Some(vec![1, 2, 3, 4]));
        assert_eq!(w.buffered(), 0);
        assert!(w.next_window().is_none());
    }

    #[test]
    fn window_drains_multiple_in_a_row() {
        let mut w = WakeWindow::new(2, 2);
        w.push(&[10, 20, 30, 40, 50, 60, 70]);
        // Three full windows of 2 samples; one trailing sample stays buffered.
        let a = w.next_window();
        let b = w.next_window();
        let c = w.next_window();
        let d = w.next_window();
        assert_eq!(a, Some(vec![10, 20]));
        assert_eq!(b, Some(vec![30, 40]));
        assert_eq!(c, Some(vec![50, 60]));
        assert_eq!(d, None);
        assert_eq!(w.buffered(), 1);
    }

    #[test]
    fn window_supports_overlap() {
        // window_size=4, stride=2 → overlapping windows.
        let mut w = WakeWindow::new(4, 2);
        w.push(&[1, 2, 3, 4, 5, 6]);
        let a = w.next_window();
        let b = w.next_window();
        let c = w.next_window();
        assert_eq!(a, Some(vec![1, 2, 3, 4]));
        assert_eq!(b, Some(vec![3, 4, 5, 6]));
        assert_eq!(c, None);
        assert_eq!(w.buffered(), 2);
    }

    #[test]
    fn clear_drops_buffer() {
        let mut w = WakeWindow::new(2, 2);
        w.push(&[1, 2, 3]);
        w.clear();
        assert_eq!(w.buffered(), 0);
        assert!(w.next_window().is_none());
    }

    #[test]
    fn zero_size_is_safe_noop() {
        let mut w = WakeWindow::new(0, 0);
        w.push(&[1, 2, 3]);
        assert!(w.next_window().is_none());
    }
}
