//! Mic-source abstraction.
//!
//! `MicSource` lets the daemon ingest PCM from any backend (`PipeWire`,
//! file replay, deterministic test stream). Real `PipeWire` / `NoiseTorch`
//! capture lands in a later iteration; the [`NullSource`] here gives
//! tests and `cargo check` something to compile against today.

use crate::errors::AudioError;
use tokio::sync::mpsc;

/// Fixed sample rate of the canonical mic stream (PRD §2.3 step 2).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// One PCM frame from the capture device.
///
/// Frames are always 16 kHz mono i16, the canonical format consumers
/// downstream of resampling expect. Size is left flexible because
/// `PipeWire`'s natural quantum is set by the graph.
#[derive(Debug, Clone)]
pub struct PcmFrame {
    /// Capture-device wall-clock timestamp of the first sample, in
    /// epoch milliseconds.
    pub ts_ms: u64,
    /// Little-endian i16 samples, 16 kHz mono.
    pub samples: Vec<i16>,
}

/// Stream metadata reported by a source.
#[derive(Debug, Clone, Copy)]
pub struct SourceMeta {
    /// Sample rate in Hz. Sources that natively differ MUST resample.
    pub sample_rate: u32,
    /// Channel count. Sources that natively differ MUST downmix.
    pub channels: u16,
}

impl SourceMeta {
    /// The canonical 16 kHz mono format.
    #[must_use]
    pub const fn canonical() -> Self {
        Self {
            sample_rate: SAMPLE_RATE_HZ,
            channels: 1,
        }
    }
}

/// Abstraction over a capture device.
///
/// Implementations push frames on a tokio mpsc; the daemon owns the
/// receiver and fans frames out to wake/VAD/socket consumers. The
/// returned future is `Send + 'static` so the daemon can spawn it
/// on a multi-thread runtime.
pub trait MicSource: Send + 'static {
    /// Describe the stream this source will produce.
    fn meta(&self) -> SourceMeta;

    /// Begin capture, pushing frames on `tx` until `tx` is dropped or
    /// an unrecoverable error occurs.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Capture`] when the underlying device
    /// rejects open / read.
    fn run(
        self,
        tx: mpsc::Sender<PcmFrame>,
    ) -> impl std::future::Future<Output = Result<(), AudioError>> + Send;
}

/// A test/bootstrap source that emits a finite sequence of silent
/// frames. Useful for wiring up `cargo check` and exercising the
/// fanout topology without a real audio device present.
pub struct NullSource {
    /// How many frames to emit before exiting cleanly.
    pub frames: usize,
    /// Samples per frame.
    pub frame_size: usize,
}

impl Default for NullSource {
    fn default() -> Self {
        // 320 samples = 20 ms at 16 kHz.
        Self {
            frames: 0,
            frame_size: 320,
        }
    }
}

impl MicSource for NullSource {
    fn meta(&self) -> SourceMeta {
        SourceMeta::canonical()
    }

    // We cannot use `async fn` here: the trait declares a `Send`-bound
    // future explicitly (so the daemon can spawn it on a multi-thread
    // runtime), and `async fn` in trait impls does not yet allow
    // attaching `+ Send`. Suppress the manual-async-fn lint.
    #[allow(
        clippy::manual_async_fn,
        reason = "Send-bound on the trait's returned future forces explicit `impl Future`"
    )]
    fn run(
        self,
        tx: mpsc::Sender<PcmFrame>,
    ) -> impl std::future::Future<Output = Result<(), AudioError>> + Send {
        async move {
            let started = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0_u64, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
            for i in 0..self.frames {
                let i_u64 = u64::try_from(i).unwrap_or(u64::MAX);
                let frame = PcmFrame {
                    ts_ms: started.saturating_add(i_u64.saturating_mul(20)),
                    samples: vec![0_i16; self.frame_size],
                };
                if tx.send(frame).await.is_err() {
                    // Receiver hung up — clean termination, not an error.
                    break;
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn null_source_emits_requested_frames() {
        let (tx, mut rx) = mpsc::channel::<PcmFrame>(8);
        let src = NullSource {
            frames: 4,
            frame_size: 160,
        };
        let h = tokio::spawn(async move { src.run(tx).await });
        let mut got = 0_usize;
        while let Some(frame) = rx.recv().await {
            assert_eq!(frame.samples.len(), 160);
            got += 1;
        }
        let res = h.await.ok().and_then(Result::ok);
        assert!(res.is_some(), "task should succeed");
        assert_eq!(got, 4);
    }

    #[test]
    fn canonical_meta_is_16k_mono() {
        let m = SourceMeta::canonical();
        assert_eq!(m.sample_rate, SAMPLE_RATE_HZ);
        assert_eq!(m.channels, 1);
    }
}
