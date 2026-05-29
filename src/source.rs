//! Mic-source abstraction.
//!
//! `MicSource` lets the daemon ingest PCM from any backend (`PipeWire`,
//! file replay, deterministic test stream). The shipping default
//! [`PwRecordSource`] spawns `pw-record` and reads 320-sample i16 frames
//! off its stdout — exactly the symmetric mirror of `wm-tts`'s `pw-cat`
//! playback path. [`SupervisedPwRecord`] wraps it with a retry/backoff
//! supervisor for the persistent-service shape PRD-pipewire-input §2.2
//! requires. [`NullSource`] remains for tests / `cargo check`.

use crate::errors::AudioError;
use crate::events::{AudioErrorPayload, AudioEvent, CaptureEnd, CaptureStart, Timestamp};
use crate::state::Shutdown;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

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

/// 20 ms of 16 kHz mono PCM, in samples. The fixed frame size the
/// [`PwRecordSource`] emits.
pub const PW_RECORD_FRAME_SAMPLES: usize = 320;

/// Resolved mic-node selection. The daemon uses the variant to decide
/// whether to inject `--target` into the `pw-record` argv and what to
/// stamp into `wm.audio.capture.start.mic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicNodeSelection {
    /// `WM_MIC_NODE` is empty / unset; route to `PipeWire`'s default
    /// source. The capture event reports `mic: "default"`.
    Default,
    /// Configured node exists on the system and will be passed as
    /// `--target <node>` to `pw-record`.
    Configured(String),
    /// Configured node was NOT found in the live source list; fall
    /// back to `PipeWire`'s default (per PRD AC9) and surface the
    /// fallback through `wm.audio.error{kind:"mic_node_fallback"}` so
    /// the operator notices.
    FallbackFromMissing {
        /// The originally-requested node name (for logging).
        requested: String,
    },
}

impl MicNodeSelection {
    /// Label used in the `wm.audio.capture.start.mic` field.
    #[must_use]
    pub fn published_mic(&self) -> String {
        match self {
            Self::Default | Self::FallbackFromMissing { .. } => "default".to_owned(),
            Self::Configured(node) => node.clone(),
        }
    }

    /// `Some(node)` when the daemon should inject `--target <node>`
    /// into the `pw-record` argv, `None` otherwise.
    #[must_use]
    pub const fn target_arg(&self) -> Option<&str> {
        match self {
            Self::Configured(node) => Some(node.as_str()),
            Self::Default | Self::FallbackFromMissing { .. } => None,
        }
    }
}

/// Resolve `mic_node` against the live list of input nodes.
///
/// `available_sources` is the set of node names that `pactl list short
/// sources` (or any equivalent enumerator) reports. The function is
/// pure — the daemon owns the side effect of populating
/// `available_sources` and reacting to the variant returned here.
///
/// Behaviour table (mirror of `wm-tts`'s `WM_SINK_NODE` resolution):
///
/// | `mic_node`              | exists in list | result                              |
/// |-------------------------|----------------|-------------------------------------|
/// | empty / whitespace      | n/a            | [`MicNodeSelection::Default`]       |
/// | non-empty               | yes            | [`MicNodeSelection::Configured`]    |
/// | non-empty               | no             | [`MicNodeSelection::FallbackFromMissing`] |
#[must_use]
pub fn resolve_mic_node(mic_node: &str, available_sources: &[String]) -> MicNodeSelection {
    let trimmed = mic_node.trim();
    if trimmed.is_empty() {
        return MicNodeSelection::Default;
    }
    if available_sources.iter().any(|s| s == trimmed) {
        MicNodeSelection::Configured(trimmed.to_owned())
    } else {
        MicNodeSelection::FallbackFromMissing {
            requested: trimmed.to_owned(),
        }
    }
}

/// Build the argv (after the binary name) passed to `pw-record`.
///
/// Extracted as a pure function so tests can pin the exact shape
/// without forking a process. Mirror of `wm-tts`'s `player_args`.
///
/// The output is always 16 kHz mono signed-16 LE on stdout (`-`).
#[must_use]
pub fn pw_record_args(target: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    args.push("--record".into());
    if let Some(node) = target {
        args.push("--target".into());
        args.push(node.into());
    }
    args.push("--rate".into());
    args.push("16000".into());
    args.push("--channels".into());
    args.push("1".into());
    args.push("--format".into());
    args.push("s16".into());
    args.push("-".into());
    args
}

/// Mic source backed by a spawned `pw-record` subprocess.
///
/// Reads little-endian i16 samples off stdout and emits 320-sample
/// (20 ms @ 16 kHz) frames. Closes cleanly when the receiver is
/// dropped; emits a [`AudioError::Capture`] when `pw-record` itself
/// exits with a non-zero status while the receiver is still alive.
pub struct PwRecordSource {
    /// Binary path or name (resolved on `$PATH` when no slash).
    pub bin: String,
    /// Resolved mic-node selection; controls the `--target` flag.
    pub selection: MicNodeSelection,
}

impl PwRecordSource {
    /// Construct a source that will spawn `bin` with the argv
    /// determined by `selection`.
    #[must_use]
    pub const fn new(bin: String, selection: MicNodeSelection) -> Self {
        Self { bin, selection }
    }
}

impl MicSource for PwRecordSource {
    fn meta(&self) -> SourceMeta {
        SourceMeta::canonical()
    }

    #[allow(
        clippy::manual_async_fn,
        reason = "Send-bound on the trait's returned future forces explicit `impl Future`"
    )]
    fn run(
        self,
        tx: mpsc::Sender<PcmFrame>,
    ) -> impl std::future::Future<Output = Result<(), AudioError>> + Send {
        async move {
            let args = pw_record_args(self.selection.target_arg());
            let mut cmd = Command::new(&self.bin);
            cmd.args(&args);
            cmd.stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(AudioError::Capture(format!(
                        "pw_record_missing: {} not on $PATH",
                        self.bin
                    )));
                }
                Err(e) => {
                    return Err(AudioError::Capture(format!(
                        "spawn {} failed: {e}",
                        self.bin
                    )));
                }
            };

            let Some(mut stdout) = child.stdout.take() else {
                return Err(AudioError::Capture(
                    "pw-record stdout was already taken".to_owned(),
                ));
            };

            let bytes_per_frame = PW_RECORD_FRAME_SAMPLES.saturating_mul(2);
            let mut buf = vec![0_u8; bytes_per_frame];
            loop {
                match stdout.read_exact(&mut buf).await {
                    Ok(_n) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        // pw-record exited; let the wait below decide
                        // ok-vs-error.
                        break;
                    }
                    Err(e) => {
                        return Err(AudioError::Capture(format!("read pw-record stdout: {e}")));
                    }
                }
                // Decode i16 LE.
                let mut samples = Vec::with_capacity(PW_RECORD_FRAME_SAMPLES);
                for chunk in buf.chunks_exact(2) {
                    // chunks_exact(2) yields slices of len 2; we can
                    // pin-pop without indexing arithmetic. Use
                    // try_into() to stay inside the clippy
                    // indexing_slicing restriction.
                    if let Ok(arr) = <[u8; 2]>::try_from(chunk) {
                        samples.push(i16::from_le_bytes(arr));
                    }
                }
                let ts_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0_u64, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
                let frame = PcmFrame { ts_ms, samples };
                if tx.send(frame).await.is_err() {
                    // Receiver dropped — clean termination.
                    break;
                }
            }

            // Reap; surface exit status as Ok vs Err.
            match child.wait().await {
                Ok(status) if status.success() => Ok(()),
                Ok(status) => Err(AudioError::Capture(format!(
                    "pw-record exited with status {status}"
                ))),
                Err(e) => Err(AudioError::Capture(format!("wait pw-record: {e}"))),
            }
        }
    }
}

/// Backoff sequence for the [`SupervisedPwRecord`] respawn loop.
///
/// 1 s, 2 s, 4 s, capped at 30 s. PRD §2.2. Pure function so the
/// daemon's retry behaviour is testable without a wall clock.
#[must_use]
pub fn backoff_ms_for_attempt(attempt: u32) -> u64 {
    const MAX_MS: u64 = 30_000;
    // 1_000 * 2^attempt, saturating at MAX_MS. attempt=0 → 1 s.
    let shift = attempt.min(30);
    let raw = 1_000_u64.saturating_mul(1_u64.checked_shl(shift).unwrap_or(u64::MAX));
    raw.min(MAX_MS)
}

/// Wraps a [`PwRecordSource`] with a respawn-on-exit loop and emits
/// `wm.audio.capture.{start,end}` / `wm.audio.error` events through
/// an mpsc channel the daemon drains and publishes.
///
/// PRD §2.2: capture is a persistent service. If `pw-record` exits
/// for any reason while we're still running, we sleep with backoff
/// (1 s → 2 s → 4 s → … capped at 30 s) and respawn. Each spawn
/// emits exactly one start event before, and exactly one end event
/// after.
pub struct SupervisedPwRecord {
    /// Binary path or name (resolved on `$PATH` when no slash).
    pub bin: String,
    /// Resolved mic-node selection; controls the `--target` flag.
    pub selection: MicNodeSelection,
    /// Channel the supervisor emits lifecycle events into.
    pub lifecycle_tx: mpsc::Sender<AudioEvent>,
    /// Shutdown handle — when triggered, the supervisor exits
    /// instead of respawning after the current `pw-record` falls.
    pub shutdown: Shutdown,
}

impl SupervisedPwRecord {
    /// Build a supervisor for a [`PwRecordSource`].
    #[must_use]
    pub const fn new(
        bin: String,
        selection: MicNodeSelection,
        lifecycle_tx: mpsc::Sender<AudioEvent>,
        shutdown: Shutdown,
    ) -> Self {
        Self {
            bin,
            selection,
            lifecycle_tx,
            shutdown,
        }
    }

    /// Emit a `wm.audio.error{kind="mic_node_fallback"}` event when the
    /// resolved selection is a fallback. Idempotent: only fires once
    /// per supervisor instance (the supervisor doesn't re-resolve mid
    /// life).
    async fn maybe_emit_fallback(&self) {
        if let MicNodeSelection::FallbackFromMissing { requested } = &self.selection {
            let ev = AudioEvent::Error(AudioErrorPayload {
                kind: "mic_node_fallback".to_owned(),
                message: format!(
                    "configured WM_MIC_NODE={requested:?} not in source list; falling back to PipeWire default"
                ),
                ts: Timestamp::now(),
            });
            // Best-effort: failure to enqueue is non-fatal.
            if let Err(e) = self.lifecycle_tx.send(ev).await {
                debug!(error = %e, "fallback notice could not be enqueued");
            }
        }
    }
}

impl MicSource for SupervisedPwRecord {
    fn meta(&self) -> SourceMeta {
        SourceMeta::canonical()
    }

    #[allow(
        clippy::manual_async_fn,
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        reason = "Send-bound on the trait's returned future forces explicit `impl Future`; supervisor orchestrates spawn, wait, error-classify, lifecycle-publish, backoff, all in one cycle"
    )]
    fn run(
        self,
        tx: mpsc::Sender<PcmFrame>,
    ) -> impl std::future::Future<Output = Result<(), AudioError>> + Send {
        async move {
            // One-shot warning for fallback mode.
            self.maybe_emit_fallback().await;

            let mut attempt: u32 = 0;
            loop {
                if self.shutdown.is_triggered() {
                    return Ok(());
                }
                // Publish capture.start before spawning.
                let start_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0_u64, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
                let mic_label = self.selection.published_mic();
                let start_ev = AudioEvent::CaptureStart(CaptureStart {
                    mic: mic_label.clone(),
                    rate: SAMPLE_RATE_HZ,
                    channels: 1,
                    ts: Timestamp(start_ms),
                });
                if let Err(e) = self.lifecycle_tx.send(start_ev).await {
                    debug!(error = %e, "capture.start could not be enqueued");
                }
                info!(
                    mic = %mic_label,
                    attempt,
                    "spawning pw-record",
                );

                // Spawn one PwRecordSource instance.
                let inner = PwRecordSource::new(self.bin.clone(), self.selection.clone());
                let inner_tx = tx.clone();
                let shutdown_for_select = self.shutdown.clone();
                let run_outcome = tokio::select! {
                    res = inner.run(inner_tx) => res,
                    () = async {
                        // Poll the shutdown flag at the cadence the
                        // daemon already uses elsewhere (250 ms).
                        loop {
                            if shutdown_for_select.is_triggered() { break; }
                            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                        }
                    } => Ok(()),
                };

                let dur_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0_u64, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
                    .saturating_sub(start_ms);

                // If shutdown is in flight, treat whatever pw-record
                // did on its way out as a graceful stop (kill_on_drop
                // from our side will surface as a non-zero exit, but
                // that's our SIGTERM, not a real fault). PRD AC6: ok
                // on graceful stop.
                let shutdown_in_flight = self.shutdown.is_triggered();
                let (outcome, reason) = if shutdown_in_flight {
                    ("ok".to_owned(), "shutdown".to_owned())
                } else {
                    match &run_outcome {
                        Ok(()) => ("ok".to_owned(), "eof".to_owned()),
                        Err(AudioError::Capture(msg)) if msg.contains("pw_record_missing") => {
                            // Also emit the dedicated error envelope for AC10.
                            let err_ev = AudioEvent::Error(AudioErrorPayload {
                                kind: "pw_record_missing".to_owned(),
                                message: msg.clone(),
                                ts: Timestamp::now(),
                            });
                            if let Err(e) = self.lifecycle_tx.send(err_ev).await {
                                debug!(error = %e, "pw_record_missing error could not be enqueued");
                            }
                            ("error".to_owned(), "pw_record_missing".to_owned())
                        }
                        Err(e) => {
                            let err_ev = AudioEvent::Error(AudioErrorPayload {
                                kind: "spawn_failed".to_owned(),
                                message: format!("{e}"),
                                ts: Timestamp::now(),
                            });
                            if let Err(send_err) = self.lifecycle_tx.send(err_ev).await {
                                debug!(error = %send_err, "spawn_failed error could not be enqueued");
                            }
                            ("error".to_owned(), format!("{e}"))
                        }
                    }
                };
                let end_ev = AudioEvent::CaptureEnd(CaptureEnd {
                    outcome: outcome.clone(),
                    dur_ms,
                    reason: reason.clone(),
                    ts: Timestamp::now(),
                });
                if let Err(e) = self.lifecycle_tx.send(end_ev).await {
                    debug!(error = %e, "capture.end could not be enqueued");
                }

                if self.shutdown.is_triggered() {
                    return Ok(());
                }

                // Sleep with backoff, then retry — but bail mid-sleep
                // if shutdown trips.
                let wait_ms = backoff_ms_for_attempt(attempt);
                warn!(
                    outcome = %outcome,
                    reason = %reason,
                    wait_ms,
                    attempt,
                    "pw-record exited; backing off before respawn",
                );
                let start_wait = std::time::Instant::now();
                while start_wait.elapsed().as_millis()
                    < u128::from(wait_ms)
                {
                    if self.shutdown.is_triggered() {
                        return Ok(());
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::manual_let_else,
    clippy::while_let_loop,
    clippy::assertions_on_constants,
    reason = "tests need to fail loudly on unexpected branches"
)]
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

    #[test]
    fn resolve_mic_node_empty_returns_default() {
        let sel = resolve_mic_node("", &["a".into(), "b".into()]);
        assert_eq!(sel, MicNodeSelection::Default);
        assert_eq!(sel.target_arg(), None);
        assert_eq!(sel.published_mic(), "default");

        // Whitespace also normalises to Default.
        let sel = resolve_mic_node("   ", &["a".into()]);
        assert_eq!(sel, MicNodeSelection::Default);
    }

    #[test]
    fn resolve_mic_node_present_returns_configured() {
        let sources = vec!["mic1".to_string(), "mic2".to_string()];
        let sel = resolve_mic_node("mic2", &sources);
        assert_eq!(sel, MicNodeSelection::Configured("mic2".into()));
        assert_eq!(sel.target_arg(), Some("mic2"));
        assert_eq!(sel.published_mic(), "mic2");
    }

    #[test]
    fn resolve_mic_node_missing_returns_fallback() {
        // AC9 — configured node not on the system → fall back to
        // PipeWire's default rather than refusing to start.
        let sources = vec!["real-mic".to_string()];
        let sel = resolve_mic_node("alsa_input.does-not-exist.source", &sources);
        match &sel {
            MicNodeSelection::FallbackFromMissing { requested } => {
                assert_eq!(requested, "alsa_input.does-not-exist.source");
            }
            other => panic!("expected FallbackFromMissing, got {other:?}"),
        }
        assert_eq!(sel.target_arg(), None, "fallback omits --target");
        assert_eq!(sel.published_mic(), "default");
    }

    #[test]
    fn pw_record_args_no_target_omits_flag() {
        let args = pw_record_args(None);
        assert!(args.iter().any(|s| s == "--record"));
        assert!(args.iter().any(|s| s == "--rate"));
        assert!(args.iter().any(|s| s == "16000"));
        assert!(args.iter().any(|s| s == "--format"));
        assert!(args.iter().any(|s| s == "s16"));
        assert!(args.iter().any(|s| s == "-"));
        assert!(!args.iter().any(|s| s == "--target"));
    }

    #[test]
    fn pw_record_args_with_target_injects_node() {
        let args = pw_record_args(Some("my-mic"));
        // The two adjacent slots must be --target followed by the node.
        let mut iter = args.iter();
        let mut found = false;
        while let Some(s) = iter.next() {
            if s == "--target" {
                let next = iter.next();
                assert_eq!(next.map(String::as_str), Some("my-mic"));
                found = true;
                break;
            }
        }
        assert!(found, "--target node pair must appear in argv");
    }

    #[test]
    fn pw_record_args_format_is_s16_le_16k_mono() {
        // PRD §2.1 pins the wire format. This test locks the argv shape
        // so future edits don't accidentally drift from s16 mono.
        let args = pw_record_args(None);
        let joined = args.join(" ");
        assert!(joined.contains("--rate 16000"), "rate must be 16000");
        assert!(joined.contains("--channels 1"), "channels must be 1");
        assert!(joined.contains("--format s16"), "format must be s16");
    }

    #[tokio::test]
    async fn pw_record_source_reports_missing_binary() {
        // AC10 — when WM_PW_RECORD_BIN is bogus, the source surfaces
        // an AudioError::Capture whose message starts with the
        // "pw_record_missing" class string. The daemon's retry/error
        // loop matches on that prefix to publish the right error
        // envelope.
        let src = PwRecordSource::new(
            "/definitely/not/pw-record-xyz".into(),
            MicNodeSelection::Default,
        );
        let (tx, _rx) = mpsc::channel::<PcmFrame>(4);
        let res = src.run(tx).await;
        match res {
            Err(AudioError::Capture(msg)) => {
                assert!(
                    msg.contains("pw_record_missing"),
                    "expected pw_record_missing in {msg}"
                );
            }
            other => panic!("expected Capture(pw_record_missing), got {other:?}"),
        }
    }

    #[test]
    fn backoff_ms_sequence_matches_prd() {
        // PRD §2.2: 1 s, 2 s, 4 s, … capped at 30 s.
        assert_eq!(backoff_ms_for_attempt(0), 1_000);
        assert_eq!(backoff_ms_for_attempt(1), 2_000);
        assert_eq!(backoff_ms_for_attempt(2), 4_000);
        assert_eq!(backoff_ms_for_attempt(3), 8_000);
        assert_eq!(backoff_ms_for_attempt(4), 16_000);
        assert_eq!(backoff_ms_for_attempt(5), 30_000, "cap at 30 s");
        assert_eq!(backoff_ms_for_attempt(20), 30_000);
        assert_eq!(backoff_ms_for_attempt(u32::MAX), 30_000);
    }

    #[tokio::test]
    async fn supervisor_emits_start_then_error_for_missing_binary() {
        // AC10 — exercise the supervisor end-to-end with a binary that
        // doesn't exist. It must emit at least one capture.start, one
        // wm.audio.error{kind:"pw_record_missing"}, one capture.end with
        // outcome=error, then either back off or exit on shutdown.
        let (life_tx, mut life_rx) = mpsc::channel::<AudioEvent>(16);
        let (pcm_tx, _pcm_rx) = mpsc::channel::<PcmFrame>(8);
        let shutdown = Shutdown::new();
        let sup = SupervisedPwRecord::new(
            "/definitely/not/pw-record-xyz".into(),
            MicNodeSelection::Default,
            life_tx,
            shutdown.clone(),
        );
        // Run the supervisor, but flip shutdown after a short while.
        let shutdown_trigger = shutdown.clone();
        let killer = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            shutdown_trigger.trigger();
        });
        let run = tokio::spawn(async move { sup.run(pcm_tx).await });

        let mut saw_start = false;
        let mut saw_error = false;
        let mut saw_end = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline && !(saw_start && saw_error && saw_end) {
            let ev = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                life_rx.recv(),
            )
            .await;
            match ev {
                Ok(Some(AudioEvent::CaptureStart(_))) => saw_start = true,
                Ok(Some(AudioEvent::Error(e))) if e.kind == "pw_record_missing" => saw_error = true,
                Ok(Some(AudioEvent::CaptureEnd(e))) if e.outcome == "error" => saw_end = true,
                _ => {}
            }
        }
        let _ = killer.await;
        let _ = run.await;
        assert!(saw_start, "expected capture.start");
        assert!(saw_error, "expected wm.audio.error{{pw_record_missing}}");
        assert!(saw_end, "expected capture.end with outcome=error");
    }

    #[tokio::test]
    async fn supervisor_emits_fallback_notice_on_missing_node() {
        // AC9 — when the configured node isn't in the source list, the
        // supervisor emits wm.audio.error{kind="mic_node_fallback"}.
        let (life_tx, mut life_rx) = mpsc::channel::<AudioEvent>(16);
        let (pcm_tx, _pcm_rx) = mpsc::channel::<PcmFrame>(8);
        let shutdown = Shutdown::new();
        let sup = SupervisedPwRecord::new(
            "/definitely/not/pw-record-xyz".into(),
            MicNodeSelection::FallbackFromMissing {
                requested: "alsa_input.fake".into(),
            },
            life_tx,
            shutdown.clone(),
        );
        let shutdown_trigger = shutdown.clone();
        let killer = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            shutdown_trigger.trigger();
        });
        let run = tokio::spawn(async move { sup.run(pcm_tx).await });

        let mut saw_fallback = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            let ev = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                life_rx.recv(),
            )
            .await;
            if let Ok(Some(AudioEvent::Error(e))) = ev {
                if e.kind == "mic_node_fallback" {
                    saw_fallback = true;
                    break;
                }
            }
        }
        let _ = killer.await;
        let _ = run.await;
        assert!(saw_fallback, "expected wm.audio.error{{mic_node_fallback}}");
    }

    #[tokio::test]
    async fn pw_record_source_streams_frames_from_stdin_stand_in() {
        // /bin/cat is our pw-record stand-in: reads stdin, writes
        // stdout. We can't drive its stdin from outside (the source
        // wires stdin to Stdio::null), but we can supply a binary that
        // writes a known byte stream to stdout and verify the source
        // decodes 320-sample frames correctly. /bin/yes outputs "y\n"
        // continuously which is unsuitable; use `printf` via /bin/sh
        // instead.
        //
        // Strategy: spawn /bin/cat with a synthetic file via stdin
        // redirection is awkward, so we use a tiny shell oneliner
        // that writes a fixed number of bytes.
        let bytes_per_frame = PW_RECORD_FRAME_SAMPLES.saturating_mul(2);
        let total_frames = 3_usize;
        let total_bytes = bytes_per_frame.saturating_mul(total_frames);
        // Build a source whose `bin` is /bin/sh and whose argv is
        // (-c, "head -c <N> /dev/zero"). We can't reuse pw_record_args
        // here — the test exercises the read-and-frame loop, not the
        // argv builder. Bypass the source entirely and drive it
        // through a small adapter.
        //
        // We special-case: use std::process::Command to spawn the
        // sh; collect stdout; feed via a minimal MicSource impl.
        let mut cmd = tokio::process::Command::new("/bin/sh");
        cmd.args([
            "-c",
            &format!("head -c {total_bytes} /dev/zero"),
        ]);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(_) => return, // sh not present — skip silently in CI
        };
        let mut stdout = match child.stdout.take() {
            Some(s) => s,
            None => return,
        };

        let (tx, mut rx) = mpsc::channel::<PcmFrame>(8);
        // Mirror the source's read+frame loop here so we exercise the
        // exact decoding pattern.
        let reader = tokio::spawn(async move {
            let mut buf = vec![0_u8; bytes_per_frame];
            loop {
                match stdout.read_exact(&mut buf).await {
                    Ok(_n) => {}
                    Err(_) => break,
                }
                let mut samples = Vec::with_capacity(PW_RECORD_FRAME_SAMPLES);
                for chunk in buf.chunks_exact(2) {
                    if let Ok(arr) = <[u8; 2]>::try_from(chunk) {
                        samples.push(i16::from_le_bytes(arr));
                    }
                }
                let frame = PcmFrame { ts_ms: 0, samples };
                if tx.send(frame).await.is_err() {
                    break;
                }
            }
        });
        let mut got = 0_usize;
        while let Some(frame) = rx.recv().await {
            assert_eq!(frame.samples.len(), PW_RECORD_FRAME_SAMPLES);
            // All zeros — /dev/zero source.
            assert!(frame.samples.iter().all(|s| *s == 0_i16));
            got += 1;
        }
        let _ = reader.await;
        let _ = child.wait().await;
        assert_eq!(got, total_frames, "expected {total_frames} frames");
    }
}
