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
//! ## Bit-exact port of the TFLM microfrontend (AC2)
//!
//! [`MelFrontend::mel_window`] is a **bit-exact** Rust port of the
//! `OHF-Voice/micro-wake-word` reference preprocessor — the TFLM
//! `audio_microfrontend` shipped as `pymicro_features.MicroFrontend`. It
//! reproduces the reference's fixed-point pipeline stage-for-stage:
//!
//! 1. **Window**: 30 ms (480-sample) frame × a Hann window quantised to
//!    Q12 fixed point. The window coefficients are computed in **`f32`**
//!    (matching the reference's `cosf`/`M_PI` float math) — using `f64`
//!    here diverges by ±1 LSB on at least one coefficient and cascades
//!    into a multi-frame mismatch, so `f32` is load-bearing, not a style
//!    choice.
//! 2. **FFT**: 512-point fixed-point real FFT (`kiss_fftr`, `FIXED_POINT=16`),
//!    ported butterfly-for-butterfly (radix-4/2) with the exact `sround`
//!    rounding and `C_FIXDIV` scaling.
//! 3. **Filterbank**: 40 triangular mel channels over 125–7 500 Hz, with the
//!    reference's overlap-carry accumulator and 64→32-bit integer sqrt.
//! 4. **Noise reduction**: per-channel exponential noise estimate +
//!    spectral subtraction (the one stateful stage; the estimate carries
//!    across the 186 frames of one window).
//! 5. **PCAN auto-gain control**: wide-dynamic-range gain LUT lookup
//!    (strength 0.95, offset 80) applied per channel.
//! 6. **Log + scale**: fixed-point natural-log LUT, scaled into uint16.
//!
//! The uint16 output is multiplied by [`FEATURE_SCALE`] (`0.0390625` = 1/25.6)
//! to f32 — exactly what `microwakeword/data.py` and `inference.py` feed the
//! model (`spectrogram.astype(np.float32) * 0.0390625`).
//!
//! ## AC2 golden parity — VERIFIED bit-exact
//!
//! The committed golden (`tests/golden/mel_440hz_8000amp.json`) is a genuine
//! `pymicro_features.MicroFrontend` export, reproducible bit-exactly via
//! `contrib/gen_golden_mel.py --verify` (maxabs = 0). [`mel_window`] now
//! reproduces that golden to maxabs = 0 as well (see
//! `ac2_mel_parity_with_training_golden` below — no longer `#[ignore]`d). The
//! port was validated against the reference C frontend directly: all 186
//! frames of the canonical 440 Hz / amp-8000 / 30 240-sample buffer match
//! the reference uint16 output bit-for-bit before scaling.

// This is a DSP module: float arithmetic and i->f / f->i conversions are
// intrinsic to spectrogram computation. The numeric lints are warn-level in
// the crate and are allowed here rather than scattered per-expression; the
// `as` conversions are all bounded (indices, sample magnitudes) and documented
// at the call sites.
#![allow(
    clippy::float_arithmetic,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "intrinsic to a fixed-point DSP port; conversions are bounded and documented"
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

/// FFT size: smallest power of two ≥ [`WINDOW_SAMPLES`] (the TFLM microfrontend
/// zero-pads the 480-sample window up to 512).
pub const FFT_SIZE: usize = 512;

/// Lower edge of the mel filterbank (microfrontend `lower_band_limit`).
pub const LOWER_BAND_LIMIT_HZ: f32 = 125.0;

/// Upper edge of the mel filterbank (microfrontend `upper_band_limit`).
pub const UPPER_BAND_LIMIT_HZ: f32 = 7_500.0;

/// Feature scale (`uint16 * 0.0390625`, i.e. `1 / 25.6`).
///
/// Applied to the integer microfrontend output before the model consumes it
/// (`microwakeword.data` / `inference`).
pub const FEATURE_SCALE: f32 = 0.039_062_5;

/// Exact i16 sample count to fill one [`NUM_FRAMES`]-frame window.
///
/// `(NUM_FRAMES + warmup) * HOP_SAMPLES` where warmup = 3 frames (the 30 ms
/// window spans 3 hops): `(186 + 3) * 160 = 30 240`. Verified against the
/// reference preprocessor — 30 240 samples yields a `(186, 40)` spectrogram.
pub const MEL_WINDOW_SAMPLES: usize = (NUM_FRAMES + 3) * HOP_SAMPLES; // 30_240

/// Stride between successive mel windows in samples. One inference per
/// 160 ms keeps the wake path responsive without re-running on every hop.
pub const MEL_STRIDE_SAMPLES: usize = HOP_SAMPLES * 16; // 2_560 (160 ms)

// ---------------------------------------------------------------------------
// Fixed-point constants (from the TFLM microfrontend headers).
// ---------------------------------------------------------------------------

const WINDOW_BITS: u32 = 12; // kFrontendWindowBits
const FILTERBANK_BITS: i32 = 12; // kFilterbankBits
const NOISE_REDUCTION_BITS: u32 = 14; // kNoiseReductionBits
const PCAN_SNR_BITS: i32 = 12; // kPcanSnrBits
const PCAN_OUTPUT_BITS: i32 = 6; // kPcanOutputBits
const PCAN_GAIN_BITS: i32 = 21;
const PCAN_STRENGTH: f32 = 0.95;
const PCAN_OFFSET: f32 = 80.0;
const SMOOTHING_BITS: u32 = 10;
const EVEN_SMOOTHING: f32 = 0.025;
const ODD_SMOOTHING: f32 = 0.06;
const MIN_SIGNAL_REMAINING: f32 = 0.05;
const LOG_SCALE_SHIFT: u32 = 6;

const LOG_SCALE_LOG2: u32 = 16;
const LOG_SEGMENTS_LOG2: u32 = 7;
const LOG_SCALE: u32 = 65_536;
const LOG_COEFF: u64 = 45_426;

/// Natural-log fixed-point LUT (`kLogLut`, 130 entries).
const LOG_LUT: [u16; 130] = [
    0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163, 2329, 2490, 2646, 2797, 2944,
    3087, 3224, 3358, 3487, 3611, 3732, 3848, 3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714,
    4791, 4864, 4934, 5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507, 5533,
    5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633, 5626, 5615, 5602, 5586, 5568,
    5547, 5524, 5498, 5470, 5439, 5406, 5370, 5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944,
    4885, 4825, 4762, 4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848, 3759,
    3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659, 2549, 2437, 2323, 2207, 2090,
    1971, 1851, 1729, 1605, 1480, 1353, 1224, 1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
];

/// `MostSignificantBit32`: `32 - clz(n)`, with `0 -> 0`.
#[inline]
fn msb32(n: u32) -> u32 {
    32 - n.leading_zeros()
}

/// `MostSignificantBit64`: `64 - clz(n)`, with `0 -> 0`.
#[inline]
fn msb64(n: u64) -> u32 {
    64 - n.leading_zeros()
}

// ---------------------------------------------------------------------------
// Fixed-point complex FFT (kiss_fftr, FIXED_POINT=16) — bit-exact port.
// ---------------------------------------------------------------------------

const FRACBITS: i32 = 15;
const SAMP_MAX: i32 = 32_767;

#[derive(Clone, Copy, Default)]
struct Cpx {
    r: i16,
    i: i16,
}

#[inline]
fn sround(x: i32) -> i16 {
    // (x + (1 << (FRACBITS-1))) >> FRACBITS, arithmetic shift on signed.
    ((x + (1 << (FRACBITS - 1))) >> FRACBITS) as i16
}

#[inline]
fn s_mul(a: i16, b: i16) -> i16 {
    sround((a as i32) * (b as i32))
}

#[inline]
fn c_mul(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: sround((a.r as i32) * (b.r as i32) - (a.i as i32) * (b.i as i32)),
        i: sround((a.r as i32) * (b.i as i32) + (a.i as i32) * (b.r as i32)),
    }
}

#[inline]
fn c_fixdiv(c: Cpx, div: i32) -> Cpx {
    // DIVSCALAR(x,k): x = sround(smul(x, SAMP_MAX/k)).
    let s = (SAMP_MAX / div) as i16;
    Cpx {
        r: s_mul(c.r, s),
        i: s_mul(c.i, s),
    }
}

#[inline]
fn c_add(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r.wrapping_add(b.r),
        i: a.i.wrapping_add(b.i),
    }
}

#[inline]
fn c_sub(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r.wrapping_sub(b.r),
        i: a.i.wrapping_sub(b.i),
    }
}

#[inline]
fn half_of(x: i16) -> i16 {
    x >> 1
}

#[inline]
fn kf_cexp(phase: f64) -> Cpx {
    Cpx {
        r: (0.5 + (SAMP_MAX as f64) * phase.cos()).floor() as i16,
        i: (0.5 + (SAMP_MAX as f64) * phase.sin()).floor() as i16,
    }
}

/// `kf_factor`: factor `n` into the radix list kiss_fft consumes.
fn kf_factor(mut n: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    let mut p = 4;
    let floor_sqrt = (n as f64).sqrt().floor() as i32;
    loop {
        while n % p != 0 {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > floor_sqrt {
                p = n;
            }
        }
        n /= p;
        out.push((p, n));
        if n <= 1 {
            break;
        }
    }
    out
}

/// Twiddle table + factors for a complex FFT of length `nfft` (forward).
struct KissCfg {
    twiddles: Vec<Cpx>,
    factors: Vec<(i32, i32)>,
}

impl KissCfg {
    fn new(nfft: usize) -> Self {
        let mut twiddles = Vec::with_capacity(nfft);
        for i in 0..nfft {
            let phase = -2.0 * std::f64::consts::PI * (i as f64) / (nfft as f64);
            twiddles.push(kf_cexp(phase));
        }
        Self {
            twiddles,
            factors: kf_factor(nfft as i32),
        }
    }
}

fn kf_bfly2(fout: &mut [Cpx], foff: usize, fstride: usize, tw: &[Cpx], m: usize) {
    for i in 0..m {
        let a = c_fixdiv(fout[foff + i], 2);
        let b = c_fixdiv(fout[foff + i + m], 2);
        let t = c_mul(b, tw[i * fstride]);
        fout[foff + i + m] = c_sub(a, t);
        fout[foff + i] = c_add(a, t);
    }
}

fn kf_bfly4(fout: &mut [Cpx], foff: usize, fstride: usize, tw: &[Cpx], m: usize) {
    let m2 = 2 * m;
    let m3 = 3 * m;
    for k in 0..m {
        let i = foff + k;
        let f0 = c_fixdiv(fout[i], 4);
        let f1 = c_fixdiv(fout[i + m], 4);
        let f2 = c_fixdiv(fout[i + m2], 4);
        let f3 = c_fixdiv(fout[i + m3], 4);
        let s0 = c_mul(f1, tw[k * fstride]);
        let s1 = c_mul(f2, tw[2 * k * fstride]);
        let s2 = c_mul(f3, tw[3 * k * fstride]);
        let s5 = c_sub(f0, s1);
        let f0 = c_add(f0, s1);
        let s3 = c_add(s0, s2);
        let s4 = c_sub(s0, s2);
        let fm2 = c_sub(f0, s3);
        let f0 = c_add(f0, s3);
        // Forward transform (st->inverse == 0).
        let fm = Cpx {
            r: s5.r.wrapping_add(s4.i),
            i: s5.i.wrapping_sub(s4.r),
        };
        let fm3 = Cpx {
            r: s5.r.wrapping_sub(s4.i),
            i: s5.i.wrapping_add(s4.r),
        };
        fout[i] = f0;
        fout[i + m] = fm;
        fout[i + m2] = fm2;
        fout[i + m3] = fm3;
    }
}

#[allow(clippy::too_many_arguments)]
fn kf_work(
    fout: &mut [Cpx],
    foff: usize,
    fin: &[Cpx],
    finoff: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[(i32, i32)],
    fidx: usize,
    tw: &[Cpx],
) {
    let (p, m) = factors[fidx];
    let p = p as usize;
    let m = m as usize;
    let fout_end = p * m;
    if m == 1 {
        let mut fi = finoff;
        for k in 0..fout_end {
            fout[foff + k] = fin[fi];
            fi += fstride * in_stride;
        }
    } else {
        let mut fi = finoff;
        let mut off = 0;
        while off != fout_end {
            kf_work(
                fout,
                foff + off,
                fin,
                fi,
                fstride * p,
                in_stride,
                factors,
                fidx + 1,
                tw,
            );
            fi += fstride * in_stride;
            off += m;
        }
    }
    // FFT_SIZE = 512 → ncfft = 256 = 4·4·4·4, so kf_factor only ever yields
    // radix-4 and radix-2 stages. Other radices are unsupported by this
    // minimal port; a `debug_assert` documents the invariant without the
    // `unreachable!` macro (which the crate denies under clippy).
    debug_assert!(p == 2 || p == 4, "unsupported FFT radix {p}");
    if p == 4 {
        kf_bfly4(fout, foff, fstride, tw, m);
    } else {
        kf_bfly2(fout, foff, fstride, tw, m);
    }
}

fn kiss_fft(cfg: &KissCfg, fin: &[Cpx]) -> Vec<Cpx> {
    let nfft = fin.len();
    let mut fout = vec![Cpx::default(); nfft];
    kf_work(&mut fout, 0, fin, 0, 1, 1, &cfg.factors, 0, &cfg.twiddles);
    fout
}

/// Fixed-point real FFT (`kiss_fftr`) for an even `nfft`.
struct KissFftr {
    ncfft: usize,
    substate: KissCfg,
    super_twiddles: Vec<Cpx>,
}

impl KissFftr {
    fn new(nfft: usize) -> Self {
        assert!(nfft % 2 == 0, "kiss_fftr requires even nfft");
        let ncfft = nfft / 2;
        let substate = KissCfg::new(ncfft);
        let mut super_twiddles = Vec::with_capacity(ncfft / 2);
        for i in 0..ncfft / 2 {
            let phase =
                -std::f64::consts::PI * ((i as f64 + 1.0) / (ncfft as f64) + 0.5);
            super_twiddles.push(kf_cexp(phase));
        }
        Self {
            ncfft,
            substate,
            super_twiddles,
        }
    }

    /// Run the real FFT on `timedata` (length `2*ncfft`), returning
    /// `ncfft + 1` complex bins.
    fn run(&self, timedata: &[i16]) -> Vec<Cpx> {
        let ncfft = self.ncfft;
        let cin: Vec<Cpx> = (0..ncfft)
            .map(|k| Cpx {
                r: timedata[2 * k],
                i: timedata[2 * k + 1],
            })
            .collect();
        let tmp = kiss_fft(&self.substate, &cin);
        let mut freq = vec![Cpx::default(); ncfft + 1];
        let tdc = c_fixdiv(tmp[0], 2);
        freq[0] = Cpx {
            r: tdc.r.wrapping_add(tdc.i),
            i: 0,
        };
        freq[ncfft] = Cpx {
            r: tdc.r.wrapping_sub(tdc.i),
            i: 0,
        };
        for k in 1..=ncfft / 2 {
            let fpk = tmp[k];
            let fpnk = Cpx {
                r: tmp[ncfft - k].r,
                i: (tmp[ncfft - k].i as i32).wrapping_neg() as i16,
            };
            let fpk = c_fixdiv(fpk, 2);
            let fpnk = c_fixdiv(fpnk, 2);
            let f1k = c_add(fpk, fpnk);
            let f2k = c_sub(fpk, fpnk);
            let tw = c_mul(f2k, self.super_twiddles[k - 1]);
            freq[k] = Cpx {
                r: half_of(f1k.r.wrapping_add(tw.r)),
                i: half_of(f1k.i.wrapping_add(tw.i)),
            };
            freq[ncfft - k] = Cpx {
                r: half_of(f1k.r.wrapping_sub(tw.r)),
                i: half_of(tw.i.wrapping_sub(f1k.i)),
            };
        }
        freq
    }
}

// ---------------------------------------------------------------------------
// Mel filterbank setup (FilterbankPopulateState) — integer weights.
// ---------------------------------------------------------------------------

const FILTERBANK_INDEX_ALIGNMENT: usize = 4;
const FILTERBANK_CHANNEL_BLOCK_SIZE: usize = 4;

#[inline]
fn freq_to_mel(f: f32) -> f32 {
    1127.0 * (f / 700.0).ln_1p()
}

struct Filterbank {
    start_index: usize,
    end_index: usize,
    channel_frequency_starts: Vec<usize>,
    channel_weight_starts: Vec<usize>,
    channel_widths: Vec<usize>,
    weights: Vec<i32>,
    unweights: Vec<i32>,
}

impl Filterbank {
    fn new() -> Self {
        let num_channels = NUM_MEL_BINS;
        let ncp1 = num_channels + 1;
        let spectrum_size = FFT_SIZE / 2 + 1;
        let index_alignment = if FILTERBANK_INDEX_ALIGNMENT < 2 {
            1
        } else {
            FILTERBANK_INDEX_ALIGNMENT / 2
        };

        let mel_low = freq_to_mel(LOWER_BAND_LIMIT_HZ);
        let mel_hi = freq_to_mel(UPPER_BAND_LIMIT_HZ);
        let mel_span = mel_hi - mel_low;
        let mel_spacing = mel_span / (ncp1 as f32);
        let center: Vec<f32> = (0..ncp1)
            .map(|i| mel_low + mel_spacing * ((i + 1) as f32))
            .collect();

        let hz_per_sbin = 0.5 * (SAMPLE_RATE_HZ as f32) / ((spectrum_size - 1) as f32);
        let start_index = (1.5 + LOWER_BAND_LIMIT_HZ / hz_per_sbin) as usize;
        let mut end_index = 0_usize;

        let mut chan_freq_index_start = start_index;
        let mut weight_index_start = 0_usize;
        let mut needs_zeros = false;
        let mut channel_frequency_starts = vec![0_usize; ncp1];
        let mut channel_weight_starts = vec![0_usize; ncp1];
        let mut channel_widths = vec![0_usize; ncp1];
        let mut actual_channel_starts = vec![0_usize; ncp1];
        let mut actual_channel_widths = vec![0_usize; ncp1];

        for chan in 0..ncp1 {
            let mut freq_index = chan_freq_index_start;
            while freq_to_mel((freq_index as f32) * hz_per_sbin) <= center[chan] {
                freq_index += 1;
            }
            let width = freq_index - chan_freq_index_start;
            actual_channel_starts[chan] = chan_freq_index_start;
            actual_channel_widths[chan] = width;

            if width == 0 {
                channel_frequency_starts[chan] = 0;
                channel_weight_starts[chan] = 0;
                channel_widths[chan] = FILTERBANK_CHANNEL_BLOCK_SIZE;
                if !needs_zeros {
                    needs_zeros = true;
                    for w in channel_weight_starts.iter_mut().take(chan) {
                        *w += FILTERBANK_CHANNEL_BLOCK_SIZE;
                    }
                    weight_index_start += FILTERBANK_CHANNEL_BLOCK_SIZE;
                }
            } else {
                let aligned_start = (chan_freq_index_start / index_alignment) * index_alignment;
                let aligned_width = chan_freq_index_start - aligned_start + width;
                let padded_width = (((aligned_width - 1) / FILTERBANK_CHANNEL_BLOCK_SIZE) + 1)
                    * FILTERBANK_CHANNEL_BLOCK_SIZE;
                channel_frequency_starts[chan] = aligned_start;
                channel_weight_starts[chan] = weight_index_start;
                channel_widths[chan] = padded_width;
                weight_index_start += padded_width;
            }
            chan_freq_index_start = freq_index;
        }

        let mut weights = vec![0_i32; weight_index_start];
        let mut unweights = vec![0_i32; weight_index_start];
        let mel_low2 = freq_to_mel(LOWER_BAND_LIMIT_HZ);
        for chan in 0..ncp1 {
            let mut frequency = actual_channel_starts[chan];
            let num_freq = actual_channel_widths[chan];
            let frequency_offset = frequency - channel_frequency_starts[chan];
            let weight_start = channel_weight_starts[chan];
            let denom_val = if chan == 0 { mel_low2 } else { center[chan - 1] };
            for j in 0..num_freq {
                let w = (center[chan] - freq_to_mel((frequency as f32) * hz_per_sbin))
                    / (center[chan] - denom_val);
                let wi = weight_start + frequency_offset + j;
                weights[wi] = (w * ((1 << FILTERBANK_BITS) as f32) + 0.5).floor() as i32;
                unweights[wi] =
                    ((1.0 - w) * ((1 << FILTERBANK_BITS) as f32) + 0.5).floor() as i32;
                frequency += 1;
            }
            if frequency > end_index {
                end_index = frequency;
            }
        }

        Self {
            start_index,
            end_index,
            channel_frequency_starts,
            channel_weight_starts,
            channel_widths,
            weights,
            unweights,
        }
    }

    /// `FilterbankConvertFftComplexToEnergy` → energy spectrum (`u32`),
    /// only `[start_index, end_index)` written (the rest stays as the C
    /// stale-buffer aliasing would leave it; those bins are never read by an
    /// in-range channel so zero is equivalent for our channels).
    fn energy(&self, fft: &[Cpx]) -> Vec<u32> {
        let mut energy = vec![0_u32; FFT_SIZE / 2 + 1];
        for i in self.start_index..self.end_index {
            let real = fft[i].r as i64;
            let imag = fft[i].i as i64;
            energy[i] = ((real * real + imag * imag) as u32).wrapping_add(0);
        }
        energy
    }

    /// `FilterbankAccumulateChannels` → 64-bit work accumulator with the
    /// overlap-carry between adjacent channels.
    fn accumulate(&self, energy: &[u32]) -> Vec<u64> {
        let mut work = vec![0_u64; NUM_MEL_BINS + 1];
        let mut weight_acc: u64 = 0;
        let mut unweight_acc: u64 = 0;
        for i in 0..NUM_MEL_BINS + 1 {
            let mag_start = self.channel_frequency_starts[i];
            let w_start = self.channel_weight_starts[i];
            let width = self.channel_widths[i];
            for j in 0..width {
                let mag = energy[mag_start + j] as u64;
                weight_acc = weight_acc.wrapping_add((self.weights[w_start + j] as u64) * mag);
                unweight_acc =
                    unweight_acc.wrapping_add((self.unweights[w_start + j] as u64) * mag);
            }
            work[i] = weight_acc;
            weight_acc = unweight_acc;
            unweight_acc = 0;
        }
        work
    }
}

/// `Sqrt32` from the microfrontend.
fn sqrt32(mut num: u32) -> u16 {
    if num == 0 {
        return 0;
    }
    let mut res: u32 = 0;
    let mut max_bit_number = (32 - msb32(num)) as i32;
    max_bit_number |= 1;
    let mut bit: u32 = 1 << (31 - max_bit_number);
    let mut iterations = (31 - max_bit_number) / 2 + 1;
    while iterations > 0 {
        iterations -= 1;
        if num >= res + bit {
            num -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    if num > res && res != 0xFFFF {
        res += 1;
    }
    res as u16
}

/// `Sqrt64` from the microfrontend.
fn sqrt64(mut num: u64) -> u32 {
    if (num >> 32) == 0 {
        return u32::from(sqrt32(num as u32));
    }
    let mut res: u64 = 0;
    let mut max_bit_number = (64 - msb64(num)) as i32;
    max_bit_number |= 1;
    let mut bit: u64 = 1 << (63 - max_bit_number);
    let mut iterations = (63 - max_bit_number) / 2 + 1;
    while iterations > 0 {
        iterations -= 1;
        if num >= res + bit {
            num -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    if num > res && res != 0xFFFF_FFFF {
        res += 1;
    }
    res as u32
}

// ---------------------------------------------------------------------------
// PCAN gain control (wide-dynamic-range LUT).
// ---------------------------------------------------------------------------

const WIDE_DYNAMIC_FUNCTION_BITS: usize = 32;
const WIDE_DYNAMIC_FUNCTION_LUT_SIZE: usize = 4 * WIDE_DYNAMIC_FUNCTION_BITS - 3;

struct Pcan {
    /// Gain LUT stored with a +6 index bias so the C pointer arithmetic
    /// (`lut += 4*interval - 6`) maps to plain indexing here.
    gain_lut: Vec<i16>,
    snr_shift: i32,
}

impl Pcan {
    fn new(input_correction_bits: i32) -> Self {
        let input_bits = (SMOOTHING_BITS as i32) - input_correction_bits;
        let mut base = vec![0_i16; WIDE_DYNAMIC_FUNCTION_LUT_SIZE + 6];
        base[6] = Self::lookup(input_bits, 0);
        base[1 + 6] = Self::lookup(input_bits, 1);
        for interval in 2..=WIDE_DYNAMIC_FUNCTION_BITS {
            let x0: u32 = 1 << (interval - 1);
            let x1 = x0 + (x0 >> 1);
            let x2 = if interval == WIDE_DYNAMIC_FUNCTION_BITS {
                x0 + (x0 - 1)
            } else {
                2 * x0
            };
            let y0 = Self::lookup(input_bits, x0) as i32;
            let y1 = Self::lookup(input_bits, x1) as i32;
            let y2 = Self::lookup(input_bits, x2) as i32;
            let diff1 = y1 - y0;
            let diff2 = y2 - y0;
            let a1 = 4 * diff1 - diff2;
            let a2 = diff2 - a1;
            base[4 * interval] = y0 as i16;
            base[4 * interval + 1] = a1 as i16;
            base[4 * interval + 2] = a2 as i16;
        }
        let snr_shift = PCAN_GAIN_BITS - input_correction_bits - PCAN_SNR_BITS;
        Self {
            gain_lut: base,
            snr_shift,
        }
    }

    fn lookup(input_bits: i32, x: u32) -> i16 {
        let x_as_float = (x as f32) / ((1_u32 << input_bits) as f32);
        let gain_as_float = ((1_u32 << PCAN_GAIN_BITS) as f32)
            * (x_as_float + PCAN_OFFSET).powf(-PCAN_STRENGTH);
        if gain_as_float > (SAMP_MAX as f32) {
            return SAMP_MAX as i16;
        }
        (gain_as_float + 0.5) as i16
    }

    fn wide_dynamic(&self, x: u32) -> i16 {
        let l = |idx: i32| self.gain_lut[(idx + 6) as usize] as i32;
        if x <= 2 {
            return l(x as i32) as i16;
        }
        let interval = msb32(x) as i32;
        let bidx = 4 * interval - 6;
        let frac = (if interval < 11 {
            x << (11 - interval)
        } else {
            x >> (interval - 11)
        } & 0x3FF) as i32;
        let l0 = l(bidx);
        let l1 = l(bidx + 1);
        let l2 = l(bidx + 2);
        let mut result = (l2 * frac) >> 5;
        result += l1 << 5;
        result *= frac;
        result = (result + (1 << 14)) >> 15;
        result += l0;
        result as i16
    }

    fn shrink(x: u32) -> u32 {
        if x < (2_u32 << PCAN_SNR_BITS) {
            (x.wrapping_mul(x)) >> (2 + 2 * PCAN_SNR_BITS - PCAN_OUTPUT_BITS)
        } else {
            (x >> (PCAN_SNR_BITS - PCAN_OUTPUT_BITS)) - (1_u32 << PCAN_OUTPUT_BITS)
        }
    }

    fn apply(&self, signal: &mut [u32], noise_estimate: &[u32]) {
        for i in 0..signal.len() {
            let gain = self.wide_dynamic(noise_estimate[i]) as u32;
            let snr = (((signal[i] as u64) * (gain as u64)) >> self.snr_shift) as u32;
            signal[i] = Self::shrink(snr);
        }
    }
}

// ---------------------------------------------------------------------------
// Noise reduction (stateful) + log scale.
// ---------------------------------------------------------------------------

struct NoiseReduction {
    even_smoothing: u32,
    odd_smoothing: u32,
    min_signal_remaining: u32,
    estimate: Vec<u32>,
}

impl NoiseReduction {
    fn new() -> Self {
        Self {
            even_smoothing: (EVEN_SMOOTHING * ((1 << NOISE_REDUCTION_BITS) as f32)) as u32,
            odd_smoothing: (ODD_SMOOTHING * ((1 << NOISE_REDUCTION_BITS) as f32)) as u32,
            min_signal_remaining: (MIN_SIGNAL_REMAINING
                * ((1 << NOISE_REDUCTION_BITS) as f32)) as u32,
            estimate: vec![0_u32; NUM_MEL_BINS],
        }
    }

    fn reset(&mut self) {
        self.estimate.iter_mut().for_each(|e| *e = 0);
    }

    fn apply(&mut self, signal: &mut [u32]) {
        for i in 0..signal.len() {
            let smoothing = if (i & 1) == 0 {
                self.even_smoothing
            } else {
                self.odd_smoothing
            };
            let one_minus = (1_u32 << NOISE_REDUCTION_BITS) - smoothing;
            let signal_scaled_up = signal[i] << SMOOTHING_BITS;
            let estimate = (((signal_scaled_up as u64) * (smoothing as u64)
                + (self.estimate[i] as u64) * (one_minus as u64))
                >> NOISE_REDUCTION_BITS) as u32;
            self.estimate[i] = estimate;
            let clamped = if estimate > signal_scaled_up {
                signal_scaled_up
            } else {
                estimate
            };
            let floor = (((signal[i] as u64) * (self.min_signal_remaining as u64))
                >> NOISE_REDUCTION_BITS) as u32;
            let subtracted = (signal_scaled_up - clamped) >> SMOOTHING_BITS;
            signal[i] = if subtracted > floor { subtracted } else { floor };
        }
    }
}

/// `Log2FractionPart` from log_scale.cc.
fn log2_fraction_part(x: u32, log2x: u32) -> u32 {
    let mut frac: i64 = (x as i64) - (1_i64 << log2x);
    if log2x < LOG_SCALE_LOG2 {
        frac <<= LOG_SCALE_LOG2 - log2x;
    } else {
        frac >>= log2x - LOG_SCALE_LOG2;
    }
    let base_seg = (frac >> (LOG_SCALE_LOG2 - LOG_SEGMENTS_LOG2)) as usize;
    let seg_unit = ((1_u32 << LOG_SCALE_LOG2) >> LOG_SEGMENTS_LOG2) as i64;
    let c0 = LOG_LUT[base_seg] as i64;
    let c1 = LOG_LUT[base_seg + 1] as i64;
    let seg_base = seg_unit * (base_seg as i64);
    let rel_pos = ((c1 - c0) * (frac - seg_base)) >> LOG_SCALE_LOG2;
    (frac + c0 + rel_pos) as u32
}

/// `Log` from log_scale.cc.
fn log_fixed(x: u32, scale_shift: u32) -> u32 {
    let integer = msb32(x) - 1;
    let fraction = log2_fraction_part(x, integer);
    let log2 = (integer << LOG_SCALE_LOG2) + fraction;
    let round = LOG_SCALE / 2;
    let loge = ((LOG_COEFF * (log2 as u64) + (round as u64)) >> LOG_SCALE_LOG2) as u32;
    ((loge << scale_shift) + round) >> LOG_SCALE_LOG2
}

/// `LogScaleApply` → uint16 features.
fn log_scale_apply(signal: &[u32], correction_bits: i32) -> Vec<u16> {
    let mut out = vec![0_u16; signal.len()];
    for (i, &raw) in signal.iter().enumerate() {
        let mut value = raw;
        if correction_bits < 0 {
            value >>= -correction_bits;
        } else {
            value <<= correction_bits;
        }
        let value = if value > 1 {
            log_fixed(value, LOG_SCALE_SHIFT)
        } else {
            0
        };
        out[i] = if value < 0xFFFF { value as u16 } else { 0xFFFF };
    }
    out
}

// ---------------------------------------------------------------------------
// Public front-end.
// ---------------------------------------------------------------------------

/// Stateful log-mel front-end producing the `[186, 40]` matrix.
///
/// Construction precomputes the window coefficients, FFT twiddles, mel
/// filterbank and PCAN gain LUT. [`MelFrontend::mel_window`] resets the
/// noise-reduction state and streams a [`MEL_WINDOW_SAMPLES`]-length PCM
/// window through the full bit-exact pipeline.
pub struct MelFrontend {
    coefficients: Vec<i16>,
    fftr: KissFftr,
    filterbank: Filterbank,
    pcan: Pcan,
    correction_bits: i32,
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontend {
    /// Build the front-end (precomputes coefficients, twiddles, filterbank,
    /// PCAN LUT). The Hann window coefficients are computed in `f32` to match
    /// the reference's `cosf` math.
    #[must_use]
    pub fn new() -> Self {
        let coefficients = window_coefficients();
        let correction_bits = (msb32(FFT_SIZE as u32) as i32) - 1 - (FILTERBANK_BITS / 2);
        Self {
            coefficients,
            fftr: KissFftr::new(FFT_SIZE),
            filterbank: Filterbank::new(),
            pcan: Pcan::new(correction_bits),
            correction_bits,
        }
    }

    /// Convert a [`MEL_WINDOW_SAMPLES`]-length PCM window into the
    /// `[186][40]` log-mel feature matrix the model consumes.
    ///
    /// The pipeline is streamed frame-by-frame (the noise-reduction estimate
    /// carries across frames, exactly as the reference does). Shorter inputs
    /// are zero-padded; longer inputs use only the leading
    /// [`MEL_WINDOW_SAMPLES`] samples, so the row count is always exactly
    /// [`NUM_FRAMES`].
    #[must_use]
    pub fn mel_window(&self, pcm: &[i16]) -> Box<[[f32; NUM_MEL_BINS]]> {
        let mut noise = NoiseReduction::new();
        noise.reset();
        let mut out = vec![[0.0_f32; NUM_MEL_BINS]; NUM_FRAMES];
        for (f, row) in out.iter_mut().enumerate() {
            let start = f * HOP_SAMPLES;
            // Window the 30 ms span (Q12 fixed point), tracking max |output|.
            let mut windowed = [0_i16; WINDOW_SAMPLES];
            let mut max_abs: i16 = 0;
            for (s, w) in windowed.iter_mut().enumerate() {
                let sample = pcm.get(start + s).copied().unwrap_or(0);
                let nv = (((sample as i32) * (self.coefficients[s] as i32)) >> WINDOW_BITS) as i16;
                *w = nv;
                let a = if nv < 0 { nv.wrapping_neg() } else { nv };
                if a > max_abs {
                    max_abs = a;
                }
            }
            // FFT input scaling: int input_shift = 15 - MSB32(max_abs).
            let input_shift = 15 - (msb32(max_abs as u32) as i32);
            let mut fft_in = [0_i16; FFT_SIZE];
            for (s, &w) in windowed.iter().enumerate() {
                // (int16_t)((uint16_t)w << input_shift): widen via u16 then truncate.
                let v = ((w as u16) as u32).wrapping_shl(input_shift as u32);
                fft_in[s] = v as u16 as i16;
            }
            let fft = self.fftr.run(&fft_in);
            let energy = self.filterbank.energy(&fft);
            let work = self.filterbank.accumulate(&energy);
            // FilterbankSqrt: output[i] = Sqrt64(work[i+1]) >> input_shift.
            let mut scaled = [0_u32; NUM_MEL_BINS];
            for (i, slot) in scaled.iter_mut().enumerate() {
                *slot = sqrt64(work[i + 1]) >> input_shift;
            }
            noise.apply(&mut scaled);
            self.pcan.apply(&mut scaled, &noise.estimate);
            let logged = log_scale_apply(&scaled, self.correction_bits);
            for (m, cell) in row.iter_mut().enumerate() {
                *cell = (logged[m] as f32) * FEATURE_SCALE;
            }
        }
        out.into_boxed_slice()
    }
}

/// Hann window coefficients quantised to Q[`WINDOW_BITS`] fixed point.
///
/// Computed in `f32` to match the reference's `cosf`/`M_PI` float arithmetic;
/// `f64` here diverges by ±1 LSB on at least one coefficient.
fn window_coefficients() -> Vec<i16> {
    let size = WINDOW_SAMPLES;
    let arg = PI * 2.0 / (size as f32);
    let mut c = Vec::with_capacity(size);
    for i in 0..size {
        let float_value = 0.5 - (0.5 * (arg * (i as f32 + 0.5)).cos());
        c.push((float_value * ((1 << WINDOW_BITS) as f32) + 0.5).floor() as i16);
    }
    c
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
        // Silence must not produce NaN/Inf; the log output is ≥ 0.
        let feat = mel_window(&vec![0_i16; MEL_WINDOW_SAMPLES]);
        for row in feat.iter() {
            for &c in row {
                assert!(c.is_finite(), "feature must be finite");
                assert!(c >= 0.0, "log scale must be non-negative");
            }
        }
    }

    #[test]
    fn mel_window_tone_has_energy() {
        // A 440 Hz tone must put nonzero energy into at least one mel bin
        // (genuinely exercises the full pipeline, not just shape).
        let mut pcm = vec![0_i16; MEL_WINDOW_SAMPLES];
        for (i, s) in pcm.iter_mut().enumerate() {
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
    fn window_coefficients_are_f32_quantized() {
        // The reference quantizes the Hann window in f32. A couple of known
        // values from the reference C frontend (cosf/M_PI, 30 ms @ 16 kHz).
        let c = window_coefficients();
        assert_eq!(c.len(), WINDOW_SAMPLES);
        // First coefficient: 0.5 - 0.5*cos(arg*0.5), arg=2π/480.
        assert_eq!(c[0], 0);
        // Midpoint is the window peak (~4096 = 1<<12).
        assert!(c[WINDOW_SAMPLES / 2] >= 4090, "peak near 1<<12");
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
    /// The committed golden is a genuine `pymicro_features.MicroFrontend`
    /// export (provenance VERIFIED via `contrib/gen_golden_mel.py --verify`,
    /// maxabs = 0). [`mel_window`] is a bit-exact port of that reference
    /// frontend, so this asserts parity to ≤1e-3 (in practice maxabs = 0).
    #[test]
    fn ac2_mel_parity_with_training_golden() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/mel_440hz_8000amp.json"
        ))
        .expect("golden vector must be present");
        let golden: serde_json::Value =
            serde_json::from_str(&raw).expect("golden must be valid JSON");
        let n = golden["n_samples"].as_u64().unwrap() as usize;
        let freq = golden["freq_hz"].as_f64().unwrap();
        let amp = golden["amplitude"].as_f64().unwrap();
        // Build the canonical 440 Hz / amp-8000 buffer with truncate-toward-
        // zero PCM. The sine is computed in f64 to byte-match the golden
        // generator (`int(AMPLITUDE * math.sin(...))`, f64); an f32 sine
        // perturbs occasional samples by ±1 and cascades into a mismatch.
        let mut pcm = vec![0_i16; n];
        for (i, s) in pcm.iter_mut().enumerate() {
            let v = (2.0 * std::f64::consts::PI * freq * (i as f64)
                / (SAMPLE_RATE_HZ as f64))
                .sin()
                * amp;
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
