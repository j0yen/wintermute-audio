//! Log-mel feature front-end for the `microWakeWord` wake path.
//!
//! ## Why this module exists
//!
//! The trained wake model (`wintermute.onnx`) declares input shape
//! `[1, 186, 40]` f32 — 186 time frames by 40 log-mel channels — not raw PCM.
//!
//! The previous wake path fed the model a raw-PCM tensor `[1, 1280]`, which
//! shape-mismatches against any real `microWakeWord` model and produced
//! `NotDetected` on every frame (the contract-mismatch wall this PRD fixes).
//! This module produces the `[186, 40]` feature matrix the model requires.
//!
//! ## Ground-truth feature spec (sourced, not guessed)
//!
//! The training preprocessor is `OHF-Voice/micro-wake-word`'s
//! `generate_features_for_clip` (`microwakeword/audio/audio_utils.py`), which
//! drives the TFLM `audio_microfrontend` / `pymicro_features.MicroFrontend`
//! with these settings (read from that file, not assumed):
//!
//! | param              | value                |
//! |--------------------|----------------------|
//! | `sample_rate`      | 16 000 Hz            |
//! | `window_size`      | 30 ms (480 samples)  |
//! | `window_step`      | 10 ms (160 samples)  |
//! | `num_channels`     | 40 mel bins          |
//! | `lower_band_limit` | 125 Hz               |
//! | `upper_band_limit` | 7 500 Hz             |
//! | output dtype       | `uint16`, then ×0.0390625 → f32 |
//!
//! The model is fed `microfrontend_uint16 * 0.0390625` (= 1 / 25.6) as f32 —
//! see `microwakeword/data.py` and `inference.py` (`spectrogram.astype(
//! np.float32) * 0.0390625`). A 186-frame window corresponds to exactly
//! **30 240 i16 samples** (1 890 ms): `30240 / 160 - 3` warmup frames `= 186`.
//!
//! ## Parity caveat (AC2)
//!
//! The TFLM microfrontend applies PCAN auto-gain-control plus a noise-reduction
//! stage in fixed-point arithmetic; reproducing it bit-for-bit in Rust is a
//! follow-up task. [`mel_window`] here implements the same *geometry* (30 ms
//! Hann-windowed frames, 10 ms hop, 40 triangular mel bins over 125–7 500 Hz,
//! log compression) and emits the correct **shape and dtype** so the
//! `[1, 186, 40]` contract (AC1) is satisfied and the inference path is
//! unblocked. Exact numeric parity against the golden vector
//! (`tests/golden/mel_440hz_8000amp.json`) is gated by AC2, which remains a
//! `#[ignore]`d test until the PCAN/noise-reduction stages are ported.
//!
//! ## AC2 golden provenance — UNVERIFIED (do not chase, 2026-06-03)
//!
//! The committed golden file's `source` field claims it was exported from
//! `pymicro_features.MicroFrontend` (uint16 output ×0.0390625). That claim does
//! **not** hold up: a genuine uint16×(1/25.6) export must yield values that are
//! exact integer multiples of `0.0390625`, but 2378 of the 7440 committed values
//! are fractional multiples (e.g. `0.81939697 / 0.0390625 = 20.9766`, not an
//! integer). `pymicro_features` is also not installed on this machine, so the
//! export could not have been reproduced here. The golden vector is therefore a
//! float-DSP artifact of unknown origin, **not** ground truth from the training
//! preprocessor — the [[agent-written-fixtures-tautology]] failure mode the PRD
//! (§2 / AC2) explicitly forbids. Porting PCAN/noise-reduction to match it would
//! prove nothing. AC2 is BLOCKED on regenerating this golden on a machine with
//! the real `pymicro_features.MicroFrontend` (use_c=True) before any parity port.

// This is a DSP module: float arithmetic and i->f / f->i conversions are
// intrinsic to spectrogram computation. The numeric lints are warn-level in
// the crate and are allowed here rather than scattered per-expression; the
// `as` conversions are all bounded (indices, sample magnitudes) and documented
// at the call sites.
#![allow(
    clippy::float_arithmetic,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "intrinsic to DSP; conversions are bounded and documented"
)]

use std::f32::consts::PI;

/// Number of mel channels per frame (microfrontend `num_channels`).
pub const NUM_MEL_BINS: usize = 40;

/// Number of time frames in one model input window (model declares 186).
pub const NUM_FRAMES: usize = 186;

/// Sample rate of the canonical mic stream.
pub const SAMPLE_RATE_HZ: usize = 16_000;

/// Analysis window length in samples (30 ms @ 16 kHz).
pub const WINDOW_SAMPLES: usize = SAMPLE_RATE_HZ * 30 / 1000; // 480

/// Hop / step between frames in samples (10 ms @ 16 kHz).
pub const HOP_SAMPLES: usize = SAMPLE_RATE_HZ * 10 / 1000; // 160

/// FFT size: smallest power of two ≥ [`WINDOW_SAMPLES`] (matches the TFLM
/// microfrontend, which zero-pads the 480-sample window up to 512).
pub const FFT_SIZE: usize = 512;

/// Lower edge of the mel filterbank (microfrontend `lower_band_limit`).
pub const LOWER_BAND_LIMIT_HZ: f32 = 125.0;

/// Upper edge of the mel filterbank (microfrontend `upper_band_limit`).
pub const UPPER_BAND_LIMIT_HZ: f32 = 7_500.0;

/// Feature scale (`uint16 * 0.0390625`, i.e. `1 / 25.6`).
///
/// Applied to the integer microfrontend output before the model consumes it.
/// Kept as a named constant so the parity work (AC2) and the geometry path
/// agree on scaling.
pub const FEATURE_SCALE: f32 = 0.039_062_5;

/// Exact i16 sample count to fill one [`NUM_FRAMES`]-frame window.
///
/// `(NUM_FRAMES + warmup) * HOP_SAMPLES` where warmup = 3 frames (the 30 ms
/// window spans 3 hops): `(186 + 3) * 160 = 30 240`. Verified against the
/// training preprocessor — 30 240 samples yields a `(186, 40)` spectrogram.
pub const MEL_WINDOW_SAMPLES: usize = (NUM_FRAMES + 3) * HOP_SAMPLES; // 30_240

/// Stride between successive mel windows in samples. One inference per
/// 160 ms keeps the wake path responsive without re-running on every hop.
pub const MEL_STRIDE_SAMPLES: usize = HOP_SAMPLES * 16; // 2_560 (160 ms)

#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    // HTK mel scale (matches the microfrontend's `FreqToMel`).
    1127.0 * (hz / 700.0).ln_1p()
}

#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (mel / 1127.0).exp_m1()
}

/// Precomputed triangular mel filterbank: for each of [`NUM_MEL_BINS`] bins,
/// the `(fft_bin_index, weight)` pairs that contribute to it. Built once per
/// [`MelFrontend`] so the per-frame hot path is a sparse dot product.
struct MelFilterbank {
    /// `bins[m]` = contributions to mel channel `m`.
    bins: Vec<Vec<(usize, f32)>>,
}

impl MelFilterbank {
    fn new() -> Self {
        let num_fft_bins = FFT_SIZE / 2 + 1; // rfft length
        let bin_hz = SAMPLE_RATE_HZ as f32 / FFT_SIZE as f32;

        let mel_lo = hz_to_mel(LOWER_BAND_LIMIT_HZ);
        let mel_hi = hz_to_mel(UPPER_BAND_LIMIT_HZ);
        // NUM_MEL_BINS triangles need NUM_MEL_BINS + 2 edge points.
        let mut edges = Vec::with_capacity(NUM_MEL_BINS + 2);
        for i in 0..NUM_MEL_BINS + 2 {
            let frac = i as f32 / (NUM_MEL_BINS + 1) as f32;
            edges.push(mel_to_hz(frac.mul_add(mel_hi - mel_lo, mel_lo)));
        }

        let mut bins = Vec::with_capacity(NUM_MEL_BINS);
        for window in edges.windows(3) {
            // windows(3) yields [low_edge, center, high_edge] for each bin.
            let (edge_lo, center, edge_hi) = match window {
                [a, b, c] => (*a, *b, *c),
                _ => continue,
            };
            let mut contrib = Vec::new();
            for k in 0..num_fft_bins {
                let freq = k as f32 * bin_hz;
                let weight = if freq >= edge_lo && freq <= center && (center - edge_lo) > 0.0 {
                    (freq - edge_lo) / (center - edge_lo)
                } else if freq > center && freq <= edge_hi && (edge_hi - center) > 0.0 {
                    (edge_hi - freq) / (edge_hi - center)
                } else {
                    0.0
                };
                if weight > 0.0 {
                    contrib.push((k, weight));
                }
            }
            bins.push(contrib);
        }
        Self { bins }
    }
}

/// Stateless (per-call) log-mel front-end producing the `[186, 40]` matrix.
///
/// Construction precomputes the Hann window and the mel filterbank; call
/// [`MelFrontend::mel_window`] with a [`MEL_WINDOW_SAMPLES`]-length PCM slice.
pub struct MelFrontend {
    hann: Vec<f32>,
    filterbank: MelFilterbank,
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontend {
    /// Build the front-end (precomputes the Hann window + mel filterbank).
    #[must_use]
    pub fn new() -> Self {
        let mut hann = Vec::with_capacity(WINDOW_SAMPLES);
        for n in 0..WINDOW_SAMPLES {
            let phase = 2.0 * PI * n as f32 / WINDOW_SAMPLES as f32;
            hann.push(0.5_f32.mul_add(-phase.cos(), 0.5));
        }
        Self {
            hann,
            filterbank: MelFilterbank::new(),
        }
    }

    /// Convert a [`MEL_WINDOW_SAMPLES`]-length PCM window into the
    /// `[186][40]` log-mel feature matrix the model consumes.
    ///
    /// Returns [`NUM_FRAMES`] rows of [`NUM_MEL_BINS`] channels as a boxed
    /// slice. Shorter inputs are zero-padded; longer inputs use only the
    /// leading [`MEL_WINDOW_SAMPLES`] samples, so the row count is always
    /// exactly [`NUM_FRAMES`].
    #[must_use]
    pub fn mel_window(&self, pcm: &[i16]) -> Box<[[f32; NUM_MEL_BINS]]> {
        let mut out = vec![[0.0_f32; NUM_MEL_BINS]; NUM_FRAMES];
        let mut frame = vec![0.0_f32; FFT_SIZE];
        for (f, row) in out.iter_mut().enumerate() {
            let start = f * HOP_SAMPLES;
            // Window the 30 ms span and zero-pad the rest of the FFT buffer.
            for (s, slot) in frame.iter_mut().enumerate() {
                *slot = if s < WINDOW_SAMPLES {
                    let sample = pcm.get(start + s).copied().unwrap_or(0);
                    // hann has exactly WINDOW_SAMPLES entries; s < WINDOW_SAMPLES.
                    f32::from(sample) * self.hann.get(s).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
            }
            let power = rfft_power(&frame);
            for (m, cell) in row.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                if let Some(contrib) = self.filterbank.bins.get(m) {
                    for &(k, w) in contrib {
                        acc += w * power.get(k).copied().unwrap_or(0.0);
                    }
                }
                // Log compression (avoids log(0)), then scale into the same
                // range as the integer microfrontend output.
                *cell = acc.ln_1p() * FEATURE_SCALE;
            }
        }
        out.into_boxed_slice()
    }
}

/// Free-function entry point matching the PRD's `mel_window` signature.
///
/// Allocates a fresh [`MelFrontend`] per call. For the daemon hot path,
/// construct a [`MelFrontend`] once and reuse it via
/// [`MelFrontend::mel_window`].
#[must_use]
pub fn mel_window(pcm: &[i16]) -> Box<[[f32; NUM_MEL_BINS]]> {
    MelFrontend::new().mel_window(pcm)
}

/// Direct real-input DFT power spectrum (`|X[k]|²`) for `k in 0..=FFT_SIZE/2`.
///
/// A naïve O(N²) DFT — adequate for a 512-point transform run a few times per
/// second and keeps the wake path dependency-free (no `rustfft`). Returns
/// `FFT_SIZE/2 + 1` real power values.
fn rfft_power(frame: &[f32]) -> Vec<f32> {
    let n = FFT_SIZE;
    let half = n / 2 + 1;
    let mut power = vec![0.0_f32; half];
    let two_pi_over_n = 2.0 * PI / n as f32;
    for (k, p) in power.iter_mut().enumerate() {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (t, &x) in frame.iter().enumerate() {
            let angle = two_pi_over_n * (k as f32) * (t as f32);
            re = x.mul_add(angle.cos(), re);
            im = (-x).mul_add(angle.sin(), im);
        }
        *p = re.mul_add(re, im * im);
    }
    power
}

/// Sliding PCM accumulator that drains [`MEL_WINDOW_SAMPLES`]-length windows
/// advanced by [`MEL_STRIDE_SAMPLES`] per pop.
///
/// This is the wake-path analogue of [`crate::wake::WakeWindow`] but sized for
/// the mel front-end: it accumulates ~1.89 s of PCM before yielding the first
/// window so the model always sees a full `[186, 40]` feature matrix.
pub struct MelWindowBuffer {
    buf: Vec<i16>,
    window_size: usize,
    stride: usize,
}

impl Default for MelWindowBuffer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl MelWindowBuffer {
    /// Construct with explicit window/stride sizes (samples).
    #[must_use]
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self {
            buf: Vec::with_capacity(window_size.saturating_mul(2)),
            window_size,
            stride,
        }
    }

    /// Construct with the mel-window defaults
    /// ([`MEL_WINDOW_SAMPLES`] / [`MEL_STRIDE_SAMPLES`]).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(MEL_WINDOW_SAMPLES, MEL_STRIDE_SAMPLES)
    }

    /// Append PCM samples.
    pub fn push(&mut self, samples: &[i16]) {
        self.buf.extend_from_slice(samples);
    }

    /// Currently buffered sample count.
    #[must_use]
    pub const fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Pop the next complete [`MEL_WINDOW_SAMPLES`]-length window, advancing the
    /// buffer by `stride`. Returns `None` until a full window is buffered.
    pub fn next_window(&mut self) -> Option<Vec<i16>> {
        if self.window_size == 0 || self.stride == 0 {
            return None;
        }
        let window = self.buf.get(..self.window_size)?.to_vec();
        let to_drop = self.stride.min(self.buf.len());
        self.buf.drain(..to_drop);
        Some(window)
    }

    /// Drop all buffered samples (mute-edge reset).
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_arithmetic,
    reason = "tests need to fail loudly and do arithmetic on outputs"
)]
mod tests {
    use super::*;

    #[test]
    fn window_sample_count_matches_training() {
        // Verified against generate_features_for_clip: 30 240 samples → 186
        // frames. (186 + 3 warmup) * 160 hop = 30 240.
        assert_eq!(MEL_WINDOW_SAMPLES, 30_240);
        assert_eq!(WINDOW_SAMPLES, 480);
        assert_eq!(HOP_SAMPLES, 160);
    }

    #[test]
    fn mel_window_shape_is_186x40() {
        let pcm = vec![0_i16; MEL_WINDOW_SAMPLES];
        let feat = mel_window(&pcm);
        assert_eq!(feat.len(), NUM_FRAMES);
        assert_eq!(feat[0].len(), NUM_MEL_BINS);
    }

    #[test]
    fn mel_window_zero_pcm_is_finite_and_nonnegative() {
        // Silence must not produce NaN/Inf; log(power+1) is ≥ 0.
        let feat = mel_window(&vec![0_i16; MEL_WINDOW_SAMPLES]);
        for row in feat.iter() {
            for &c in row {
                assert!(c.is_finite(), "feature must be finite");
                assert!(c >= 0.0, "log(power+1)*scale must be non-negative");
            }
        }
    }

    #[test]
    fn mel_window_tone_has_energy() {
        // A 440 Hz tone must put nonzero energy into at least one mel bin
        // (genuinely exercises the STFT + filterbank path, not just shape).
        let mut pcm = vec![0_i16; MEL_WINDOW_SAMPLES];
        for (i, s) in pcm.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let v = (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE_HZ as f32).sin() * 8000.0;
            *s = v as i16;
        }
        let feat = mel_window(&pcm);
        let total: f32 = feat.iter().flat_map(|r| r.iter()).sum();
        assert!(total > 0.0, "tone must produce nonzero mel energy");
    }

    #[test]
    fn short_input_is_zero_padded_to_full_shape() {
        let feat = mel_window(&[0_i16; 100]);
        assert_eq!(feat.len(), NUM_FRAMES);
        assert_eq!(feat[0].len(), NUM_MEL_BINS);
    }

    #[test]
    fn buffer_drains_full_window() {
        let mut b = MelWindowBuffer::new(4, 2);
        b.push(&[1, 2, 3]);
        assert!(b.next_window().is_none());
        b.push(&[4, 5]);
        assert_eq!(b.next_window(), Some(vec![1, 2, 3, 4]));
        assert_eq!(b.buffered(), 3);
    }

    #[test]
    fn buffer_clear_resets() {
        let mut b = MelWindowBuffer::with_defaults();
        b.push(&[7_i16; 100]);
        b.clear();
        assert_eq!(b.buffered(), 0);
    }

    /// AC2 — mel parity with the training preprocessor.
    ///
    /// IGNORED: the geometry-only [`mel_window`] does not yet reproduce the
    /// TFLM microfrontend's PCAN auto-gain + noise-reduction fixed-point stages.
    /// More importantly, the committed golden vector at
    /// `tests/golden/mel_440hz_8000amp.json` is of UNVERIFIED provenance — its
    /// `source` claims a `pymicro_features.MicroFrontend` uint16×0.0390625
    /// export, but 2378/7440 values are not integer multiples of 0.0390625, so
    /// it cannot be that export (see module doc). It must NOT be treated as
    /// ground truth.
    ///
    /// TODO(AC2): FIRST regenerate the golden on a machine with the real
    /// `pymicro_features.MicroFrontend` (use_c=True) and commit the export
    /// script for reproducibility; THEN port the PCAN gain-control +
    /// noise-reduction stages from `OHF-Voice/micro-wake-word`'s C microfrontend
    /// and drop `#[ignore]` to assert parity ≤1e-3.
    #[test]
    #[ignore = "AC2 blocked: golden provenance unverified (not a real pymicro export) + PCAN port pending"]
    fn ac2_mel_parity_with_training_golden() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/mel_440hz_8000amp.json"
        ))
        .expect("golden vector must be present");
        let golden: serde_json::Value =
            serde_json::from_str(&raw).expect("golden must be valid JSON");
        let n = golden["n_samples"].as_u64().unwrap() as usize;
        let freq = golden["freq_hz"].as_f64().unwrap() as f32;
        let amp = golden["amplitude"].as_f64().unwrap() as f32;
        let mut pcm = vec![0_i16; n];
        for (i, s) in pcm.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let v = (2.0 * PI * freq * i as f32 / SAMPLE_RATE_HZ as f32).sin() * amp;
            *s = v as i16;
        }
        let feat = mel_window(&pcm);
        let rows = golden["features"].as_array().unwrap();
        let mut max_err = 0.0_f32;
        for (fi, row) in rows.iter().enumerate() {
            for (mi, gv) in row.as_array().unwrap().iter().enumerate() {
                let g = gv.as_f64().unwrap() as f32;
                max_err = max_err.max((feat[fi][mi] - g).abs());
            }
        }
        assert!(max_err <= 1e-3, "mel parity max error {max_err} > 1e-3");
    }
}
