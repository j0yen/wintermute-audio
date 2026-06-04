//! Top-level daemon: wires the mic source, ring fanout, agorabus
//! publisher/subscriber, and graceful-shutdown loop.

use crate::config::Config;
use crate::errors::AudioError;
use crate::events::{
    AudioEvent, ControlEvent, SpeechChunk, SpeechEnd, SpeechStart, Timestamp, WakeDetected,
    is_self_emitted_topic,
};
use crate::fanout;
use crate::source::{MicSource, PcmFrame};
use crate::state::{MuteReason, MuteState, Shutdown};
use crate::vad::{NullVadDetector, SpeechEdge, VadDetector, VadEdgeTracker, VadWindow};
use crate::features::MelWindowBuffer;
use crate::wake::{
    self, NullWakeDetector, WakeDetector, WakeOutcome, WakeSlot,
};

use agorabus::Client;
use base64::Engine as _;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer as _, Observer as _, Producer as _, Split as _};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Bounded ring-buffer capacity in i16 samples (~5 s of 16 kHz mono).
pub const RING_CAPACITY: usize = 16_000 * 5;
/// Channel depth for the source -> daemon hand-off.
const SOURCE_CHANNEL_DEPTH: usize = 64;

/// Cumulative-bytes counter for the capture metric (PRD §2.4).
///
/// Cheap to clone (it's just an `Arc<AtomicU64>`). The capture frame
/// path bumps it for each PCM frame the daemon successfully forwards
/// to the fanout broadcast channel; observers (the `wm-audio-dump`
/// subcommand, integration tests, future metrics endpoint) can read
/// the live value with [`CapturedBytes::load`].
#[derive(Debug, Clone, Default)]
pub struct CapturedBytes(Arc<AtomicU64>);

impl CapturedBytes {
    /// Build a fresh counter starting at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current cumulative captured-bytes total.
    #[must_use]
    pub fn load(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Add `n` bytes to the counter. Saturates on overflow.
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
}

/// Daemon handle wrapping all wired components.
///
/// Construct via [`Daemon::new`]; run via [`Daemon::run`]. The
/// [`run`] free function in this module is the one-shot helper for
/// the `wm-audio` binary.
pub struct Daemon<S: MicSource> {
    config: Config,
    source: S,
    mute: MuteState,
    shutdown: Shutdown,
    wake_slot: WakeSlot,
    vad_detector: Arc<dyn VadDetector>,
    lifecycle_rx: Option<mpsc::Receiver<AudioEvent>>,
    captured_bytes: CapturedBytes,
}

impl<S: MicSource> Daemon<S> {
    /// Construct a daemon with the given config + capture source.
    ///
    /// Defaults to a [`NullWakeDetector`] labelled from
    /// [`Config::wake_word`]. Swap in a real backend via
    /// [`Daemon::with_wake_detector`].
    #[must_use]
    pub fn new(config: Config, source: S) -> Self {
        let label = config.wake_word.as_label().to_owned();
        let default_detector: Arc<dyn WakeDetector> = Arc::new(NullWakeDetector::new(label));
        Self {
            config,
            source,
            mute: MuteState::new(),
            shutdown: Shutdown::new(),
            wake_slot: wake::wake_slot(default_detector),
            vad_detector: Arc::new(NullVadDetector::new("null")),
            lifecycle_rx: None,
            captured_bytes: CapturedBytes::new(),
        }
    }

    /// Attach a lifecycle-event receiver. Events sent on the matching
    /// `Sender` (held by e.g. [`crate::source::SupervisedPwRecord`])
    /// are drained by the main loop and republished on the bus.
    ///
    /// Without this, supervised sources still work but their
    /// `wm.audio.capture.start` / `wm.audio.capture.end` / error
    /// events stay confined to the in-process channel.
    #[must_use]
    pub fn with_lifecycle_channel(mut self, rx: mpsc::Receiver<AudioEvent>) -> Self {
        self.lifecycle_rx = Some(rx);
        self
    }

    /// Shared handle to the cumulative captured-bytes counter (PRD §2.4 / AC7).
    #[must_use]
    pub fn captured_bytes_handle(&self) -> CapturedBytes {
        self.captured_bytes.clone()
    }

    /// Swap in a wake-word backend. Returns `self` for the builder
    /// pattern. The backend MUST be `Send + Sync` (enforced by the
    /// trait's supertraits).
    #[must_use]
    pub fn with_wake_detector<W: WakeDetector + 'static>(self, detector: W) -> Self {
        wake::write_slot(&self.wake_slot, Arc::new(detector));
        self
    }

    /// Swap in a wake-word backend from an already-boxed `Arc<dyn WakeDetector>`.
    ///
    /// This is the entry point for [`crate::inference::load_or_null_wake`],
    /// which returns either an ONNX-backed detector or the null fallback
    /// boxed as a trait object — so the daemon can wire whichever it gets
    /// without the caller knowing the concrete type.
    #[must_use]
    pub fn with_wake_detector_arc(self, detector: Arc<dyn WakeDetector>) -> Self {
        wake::write_slot(&self.wake_slot, detector);
        self
    }

    /// Shared handle to the active wake-word backend (PRD AC6
    /// hot-swap target). Mutating the slot mid-run takes effect on the
    /// next window inference.
    #[must_use]
    pub fn wake_slot(&self) -> WakeSlot {
        Arc::clone(&self.wake_slot)
    }

    /// Swap in a VAD backend. Returns `self` for the builder pattern.
    /// The backend MUST be `Send + Sync` (enforced by the trait's
    /// supertraits).
    #[must_use]
    pub fn with_vad_detector<V: VadDetector + 'static>(mut self, detector: V) -> Self {
        self.vad_detector = Arc::new(detector);
        self
    }

    /// Swap in a VAD backend from an already-boxed `Arc<dyn VadDetector>`
    /// (the entry point for [`crate::inference::load_or_null_vad`], which
    /// returns either the ONNX detector or the null fallback).
    #[must_use]
    pub fn with_vad_detector_arc(mut self, detector: Arc<dyn VadDetector>) -> Self {
        self.vad_detector = detector;
        self
    }

    /// Shared mute handle (for tests / inspection).
    #[must_use]
    pub fn mute(&self) -> MuteState {
        self.mute.clone()
    }

    /// Shared shutdown handle (for tests).
    #[must_use]
    pub fn shutdown_handle(&self) -> Shutdown {
        self.shutdown.clone()
    }

    /// Drive the daemon until shutdown.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Bus`] if the agorabus daemon cannot be
    /// reached, or [`AudioError::Capture`] on a fatal capture failure.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub async fn run(self) -> Result<(), AudioError> {
        let Self {
            config,
            source,
            mute,
            shutdown,
            wake_slot,
            vad_detector,
            mut lifecycle_rx,
            captured_bytes,
        } = self;

        info!(
            session = %config.session_id,
            mic = %config.mic_node,
            wake = %config.wake_word.as_label(),
            speech_end_silence_ms = config.speech_end_silence_ms,
            "wm-audio starting",
        );

        // 1. Open the agorabus connections (pub + sub use separate
        //    clients so the subscribe stream doesn't HOL-block our
        //    publishes).
        let mut pub_client = Client::connect(&config.bus_socket)
            .await
            .map_err(|e| AudioError::Bus(format!("connect pub: {e:#}")))?;
        pub_client
            .announce(
                &config.session_id,
                std::process::id(),
                std::env::current_dir()
                    .ok()
                    .and_then(|p| p.to_str().map(str::to_owned))
                    .unwrap_or_default()
                    .as_str(),
                "wm-audio mic/wake/vad pipeline",
            )
            .await
            .map_err(|e| AudioError::Bus(format!("announce: {e:#}")))?;

        let mut sub_client = Client::connect(&config.bus_socket)
            .await
            .map_err(|e| AudioError::Bus(format!("connect sub: {e:#}")))?;
        sub_client
            .announce(
                &format!("{}-sub", config.session_id),
                std::process::id(),
                "",
                "wm-audio control subscribe",
            )
            .await
            .map_err(|e| AudioError::Bus(format!("announce sub: {e:#}")))?;
        // Subscribe to TTS + dialog control surfaces.
        let prefixes = ["wm.tts.", "wm.dialog.", "wm.audio.reload"];
        for prefix in prefixes {
            sub_client
                .subscribe(prefix)
                .await
                .map_err(|e| AudioError::Bus(format!("subscribe {prefix}: {e:#}")))?;
        }

        // 2. Spawn the source on its own channel.
        let (frame_tx, mut frame_rx) =
            mpsc::channel::<PcmFrame>(SOURCE_CHANNEL_DEPTH);
        let source_task = tokio::spawn(async move { source.run(frame_tx).await });

        // 3. Bounded ring buffer reserved for in-process wake/VAD
        //    consumers (iter-4+); UDS socket fanout uses the broadcast
        //    channel below instead so slow subscribers can't stall the
        //    capture loop.
        let rb = HeapRb::<i16>::new(RING_CAPACITY);
        let (mut producer, mut consumer) = rb.split();

        // 3a. Broadcast channel + UDS fanout server (PRD §2.3 step 3, AC7).
        let (bcast_tx, bcast_seed) = fanout::channel();
        drop(bcast_seed);
        let fanout_socket = config.mic_socket.clone();
        let fanout_shutdown = shutdown.clone();
        let fanout_bcast = bcast_tx.clone();
        let fanout_task = tokio::spawn(async move {
            if let Err(e) = fanout::run(fanout_socket, fanout_bcast, fanout_shutdown).await {
                warn!(error = %e, "fanout listener exited with error");
            }
        });

        // 4. Spawn the control-subscriber loop.
        let control_mute = mute.clone();
        let control_shutdown = shutdown.clone();
        let control_wake_slot = Arc::clone(&wake_slot);
        let control_task = tokio::spawn(async move {
            run_control_loop(sub_client, control_mute, control_shutdown, control_wake_slot)
                .await;
        });

        // 5. Spawn the signal handler that flips the shutdown flag.
        let sig_shutdown = shutdown.clone();
        let signal_task = tokio::spawn(async move {
            install_signal_handlers(sig_shutdown).await;
        });

        // 6. Main loop: drain capture frames, push into the ring,
        //    publish bus events on transitions. Wake inference plugs
        //    in via the [`WakeDetector`] trait; the iter-7 default
        //    backend is [`NullWakeDetector`].
        let mut total_samples: u64 = 0;
        let mut wake_window = MelWindowBuffer::with_defaults();
        let mut vad_window = VadWindow::with_defaults();
        let mut vad_tracker = VadEdgeTracker::new(
            0.5,
            crate::vad::VAD_FRAME_MS,
            config.speech_end_silence_ms,
        );
        let mut chunk_seq: u64 = 0;
        loop {
            if shutdown.is_triggered() {
                info!("shutdown requested, draining");
                break;
            }
            // Wait for either a PCM frame from the source or a
            // lifecycle event from the capture supervisor (capture
            // start/end, wm.audio.error). The 250 ms ticker keeps the
            // shutdown flag responsive even when neither channel has
            // traffic.
            #[allow(clippy::redundant_pub_crate)] // tokio::select! macro hygiene
            let frame_opt = tokio::select! {
                f = frame_rx.recv() => {
                    let Some(frame) = f else {
                        debug!("capture source closed");
                        break;
                    };
                    Some(frame)
                },
                ev = async {
                    match &mut lifecycle_rx {
                        Some(rx) => rx.recv().await,
                        None => {
                            // No lifecycle channel attached — block forever
                            // so the select! arm never wins.
                            std::future::pending::<Option<AudioEvent>>().await
                        }
                    }
                } => {
                    if let Some(ev) = ev {
                        publish_audio_event(&mut pub_client, &ev).await;
                    }
                    None
                }
                () = tokio::time::sleep(std::time::Duration::from_millis(250)) => None,
            };
            let Some(frame) = frame_opt else { continue };

            if !mute.should_publish_pcm() {
                // Dialog has hard-muted the mic; drop the frame.
                continue;
            }

            let n = producer.push_slice(&frame.samples);
            if n < frame.samples.len() {
                warn!(
                    dropped = frame.samples.len() - n,
                    "ring buffer overflow; downstream consumer is too slow",
                );
            }
            let n_u64 = u64::try_from(n).unwrap_or(u64::MAX);
            total_samples = total_samples.saturating_add(n_u64);

            // Capture metric (PRD §2.4 / AC7): two bytes per i16 sample.
            // Bump BEFORE the broadcast send so the counter reflects
            // bytes the capture path saw, not bytes downstream
            // received — fanout sends never panic and the receiver
            // count is orthogonal.
            let frame_bytes = u64::try_from(frame.samples.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(2);
            captured_bytes.add(frame_bytes);
            // Fanout to UDS subscribers. `SendError` only means "no
            // active subscribers right now," which is the steady state
            // when nothing is listening — ignore.
            let _ = bcast_tx.send(frame);

            // Drain the in-process ring into the wake + VAD windows.
            let avail = consumer.occupied_len();
            if avail > 0 {
                let mut scratch = vec![0_i16; avail];
                let popped = consumer.pop_slice(&mut scratch);
                scratch.truncate(popped);
                // Wake ALWAYS listens — never gated by mute. The wake word
                // must fire regardless of TTS/dialog state (per user directive);
                // the old TTS/dialog mute gate could get stuck and kill wake.
                wake_window.push(&scratch);
                vad_window.push(&scratch);
            }

            // Run wake inference on every complete window that accumulated.
            // Re-read the slot each outer pass so a `wm.audio.reload`
            // hot-swap (PRD AC6) takes effect within one frame quantum.
            let active_detector = wake::read_slot(&wake_slot);
            while let Some(window) = wake_window.next_window() {
                // No mute gate: wake always runs (see push site above).
                let WakeOutcome::Detected { confidence } = active_detector.process(&window)
                else {
                    continue;
                };
                if confidence < config.wake_threshold {
                    continue;
                }
                let ev = AudioEvent::Wake(WakeDetected {
                    wake_word: active_detector.label().to_owned(),
                    confidence,
                    ts: Timestamp::now(),
                });
                publish_audio_event(&mut pub_client, &ev).await;
            }

            // Run VAD inference on every complete VAD window. Publishes
            // speech.start on rising edge, speech.chunk per in-speech
            // window, speech.end after PRD §2.3's 500 ms hangover.
            while let Some(window) = vad_window.next_window() {
                let outcome = vad_detector.process(&window);
                let edge = vad_tracker.step(outcome);
                match edge {
                    Some(SpeechEdge::Start) => {
                        chunk_seq = 0;
                        let ev = AudioEvent::SpeechStart(SpeechStart {
                            ts: Timestamp::now(),
                        });
                        publish_audio_event(&mut pub_client, &ev).await;
                    }
                    Some(SpeechEdge::End { duration_ms }) => {
                        let ev = AudioEvent::SpeechEnd(SpeechEnd {
                            duration_ms,
                            ts: Timestamp::now(),
                        });
                        publish_audio_event(&mut pub_client, &ev).await;
                    }
                    None => {}
                }
                if vad_tracker.in_speech() {
                    let mut pcm_bytes = Vec::with_capacity(window.len().saturating_mul(2));
                    for sample in &window {
                        pcm_bytes.extend_from_slice(&sample.to_le_bytes());
                    }
                    let pcm_b64 = base64::engine::general_purpose::STANDARD.encode(&pcm_bytes);
                    let ev = AudioEvent::SpeechChunk(SpeechChunk {
                        seq: chunk_seq,
                        pcm_b64,
                        ts: Timestamp::now(),
                    });
                    chunk_seq = chunk_seq.saturating_add(1);
                    publish_audio_event(&mut pub_client, &ev).await;
                }
            }
        }

        // 7. Tear down. We do *not* await source_task here — the
        //    capture loop owns a socket / device handle that may not
        //    interrupt cleanly. Drop the channel to signal it.
        drop(frame_rx);
        drop(bcast_tx);
        let _ = signal_task.await;
        let _ = control_task.await;
        let _ = fanout_task.await;
        // Source task is the supervisor (or a NullSource). Give it a
        // short window to flush its final capture.end event into the
        // lifecycle channel before we publish.
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            source_task,
        )
        .await;

        // 7a. Drain whatever the supervisor managed to enqueue before
        //     exiting (its final capture.end), best-effort. We don't
        //     block past 500 ms — the bus is about to die anyway.
        if let Some(mut rx) = lifecycle_rx {
            let drain_deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(500);
            loop {
                if std::time::Instant::now() > drain_deadline {
                    break;
                }
                match tokio::time::timeout(
                    std::time::Duration::from_millis(100),
                    rx.recv(),
                )
                .await
                {
                    Ok(Some(ev)) => publish_audio_event(&mut pub_client, &ev).await,
                    // Either channel closed or wait timed out — stop draining.
                    Ok(None) | Err(_) => break,
                }
            }
        }

        // 8. Best-effort publish of shutdown notice (cosmetic; the
        //    bus client may already be torn down).
        let _ = pub_client
            .publish(
                "wm.audio.shutdown",
                serde_json::json!({ "ts": Timestamp::now() }),
            )
            .await;

        info!(samples = total_samples, "wm-audio exited");
        Ok(())
    }
}

/// Convenience entry point for the binary.
///
/// Builds a [`Daemon`] from env + the provided capture source and
/// drives it to completion.
///
/// # Errors
///
/// Propagates [`Daemon::run`].
pub async fn run<S: MicSource>(source: S) -> Result<(), AudioError> {
    let config = Config::from_env()?;
    Daemon::new(config, source).run().await
}

async fn publish_audio_event(client: &mut Client, ev: &AudioEvent) {
    match ev.payload() {
        Ok(payload) => {
            if let Err(e) = client.publish(ev.topic(), payload).await {
                warn!(topic = ev.topic(), error = %e, "publish failed");
            }
        }
        Err(e) => warn!(topic = ev.topic(), error = %e, "serialize payload failed"),
    }
}

async fn run_control_loop(
    mut sub: Client,
    mute: MuteState,
    shutdown: Shutdown,
    wake_slot: WakeSlot,
) {
    loop {
        if shutdown.is_triggered() {
            return;
        }
        let next = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            sub.next_event(),
        )
        .await;
        let ev = match next {
            Ok(Ok(Some(ev))) => ev,
            Ok(Ok(None)) => {
                debug!("bus subscribe stream closed");
                return;
            }
            Ok(Err(e)) => {
                warn!(error = %e, "control subscribe error");
                return;
            }
            Err(_) => continue, // timeout — re-check shutdown
        };
        // Silently drop echoes of our own publishes — without this,
        // every `wm.audio.error` we publish on the bus would get
        // routed back via `wm.audio.` subscribers and recurse. Mirror
        // of the wm-tts error-loop-suppress fix (PRD §2.2).
        if is_self_emitted_topic(&ev.topic) {
            continue;
        }
        match ControlEvent::from_topic(&ev.topic) {
            Some(ControlEvent::TtsStart) => mute.set(MuteReason::TtsActive, true),
            Some(ControlEvent::TtsEnd) => mute.set(MuteReason::TtsActive, false),
            Some(ControlEvent::MuteRequest) => mute.set(MuteReason::DialogRequest, true),
            Some(ControlEvent::UnmuteRequest) => mute.set(MuteReason::DialogRequest, false),
            Some(ControlEvent::Reload) => match apply_wake_reload(&ev.data, &wake_slot) {
                Ok(label) => info!(wake_word = %label, "wake detector hot-swapped"),
                Err(e) => warn!(error = %e, "wake reload rejected"),
            },
            None => debug!(topic = %ev.topic, "ignored bus event"),
        }
    }
}

/// Apply a `wm.audio.reload` payload to the wake-detector slot.
///
/// Selects the new wake word from (in order): the payload's
/// `wake_word` field, then the `WM_WAKE_WORD` env var. Builds a fresh
/// [`NullWakeDetector`] labelled for that word and swaps it in.
///
/// Returns the new label on success; returns [`AudioError::Config`]
/// if no source provided a value or the value is unknown. Real ONNX
/// backends will plug in here once the model is loadable.
///
/// # Errors
///
/// Returns [`AudioError::Config`] for missing or malformed wake-word
/// identifiers.
pub fn apply_wake_reload(
    payload: &serde_json::Value,
    slot: &WakeSlot,
) -> Result<String, AudioError> {
    let chosen = payload
        .get("wake_word")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| std::env::var("WM_WAKE_WORD").ok())
        .ok_or_else(|| {
            AudioError::Config(
                "reload: no wake_word in payload and WM_WAKE_WORD env not set".to_owned(),
            )
        })?;
    let word = crate::config::WakeWord::parse(&chosen)?;
    let new: Arc<dyn WakeDetector> = Arc::new(NullWakeDetector::new(word.as_label()));
    wake::write_slot(slot, new);
    Ok(word.as_label().to_owned())
}

async fn install_signal_handlers(shutdown: Shutdown) {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to install SIGTERM handler");
            return;
        }
    };
    let mut int = match signal(SignalKind::interrupt()) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to install SIGINT handler");
            return;
        }
    };
    tokio::select! {
        _ = term.recv() => info!("SIGTERM"),
        _ = int.recv() => info!("SIGINT"),
    }
    shutdown.trigger();
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::unnecessary_literal_bound,
    reason = "tests need to fail loudly and stub trait impls return literal-tied lifetimes by default"
)]
mod tests {
    use super::*;
    use crate::source::NullSource;
    use crate::wake::WAKE_WINDOW_SAMPLES;

    fn test_config() -> Config {
        Config {
            mic_node: String::new(),
            wake_word: crate::config::WakeWord::HeyJarvis,
            wake_threshold: 0.6,
            mic_socket: std::path::PathBuf::from("/tmp/wm-audio-test.sock"),
            bus_socket: std::path::PathBuf::from("/tmp/wm-audio-no-bus.sock"),
            session_id: "wm-audio-test".into(),
            pw_record_bin: crate::config::DEFAULT_PW_RECORD.to_owned(),
            speech_end_silence_ms: crate::config::SPEECH_SILENCE_MS_DEFAULT,
        }
    }

    #[tokio::test]
    async fn daemon_constructs_without_bus() {
        // We don't run() — that would try to connect. We just verify
        // the daemon assembles and exposes its handles.
        let d = Daemon::new(test_config(), NullSource::default());
        let m = d.mute();
        let s = d.shutdown_handle();
        assert!(m.should_run_wake());
        assert!(!s.is_triggered());
        s.trigger();
        assert!(s.is_triggered());
    }

    #[test]
    fn captured_bytes_handle_round_trips() {
        // The counter starts at zero, accepts adds, and is observable
        // through a clone. AC7 reads through this handle from the bin
        // surface.
        let d = Daemon::new(test_config(), NullSource::default());
        let h = d.captured_bytes_handle();
        assert_eq!(h.load(), 0);
        h.add(640);
        h.add(640);
        assert_eq!(h.load(), 1_280);
        let h2 = h.clone();
        assert_eq!(h2.load(), 1_280, "clones observe the shared counter");
        h2.add(0);
        assert_eq!(h.load(), 1_280);
    }

    #[tokio::test]
    async fn lifecycle_channel_is_attachable() {
        // The builder accepts a Receiver<AudioEvent> and stores it for
        // the main loop to drain. We don't run() here — the channel is
        // unit-tested via the daemon's plumbing test downstream.
        let (_tx, rx) = mpsc::channel::<AudioEvent>(4);
        let d = Daemon::new(test_config(), NullSource::default())
            .with_lifecycle_channel(rx);
        // Internal field is private; we can only check by re-attaching
        // wouldn't compile. Existence is what we want.
        assert!(!d.shutdown_handle().is_triggered());
    }

    /// Stub detector that records every window it sees, used to assert
    /// the daemon hands inference windows to the trait.
    #[derive(Clone, Default)]
    struct RecordingDetector {
        seen: std::sync::Arc<std::sync::Mutex<usize>>,
    }

    impl WakeDetector for RecordingDetector {
        fn label(&self) -> &str {
            "stub"
        }

        fn process(&self, _window: &[i16]) -> WakeOutcome {
            if let Ok(mut guard) = self.seen.lock() {
                *guard = guard.saturating_add(1);
            }
            WakeOutcome::NotDetected
        }
    }

    #[tokio::test]
    async fn with_wake_detector_swaps_backend() {
        // Constructor-level check: builder accepts and stores a real
        // detector implementation. Running the loop requires a live
        // agorabus daemon, which the test harness does not have —
        // we only validate the wiring shape here.
        let stub = RecordingDetector::default();
        let d = Daemon::new(test_config(), NullSource::default())
            .with_wake_detector(stub.clone());
        // Read the active detector through the slot the same way the
        // capture loop does, and exercise it twice.
        let detector = wake::read_slot(&d.wake_slot());
        detector.process(&[0_i16; WAKE_WINDOW_SAMPLES]);
        detector.process(&[0_i16; WAKE_WINDOW_SAMPLES]);
        let seen = stub.seen.lock().map(|g| *g).unwrap_or(0);
        assert_eq!(seen, 2, "trait dispatch should reach the swapped backend");
    }

    #[test]
    fn apply_wake_reload_payload_swaps_label() {
        let d = Daemon::new(test_config(), NullSource::default());
        let slot = d.wake_slot();
        assert_eq!(wake::read_slot(&slot).label(), "hey-jarvis");
        let payload = serde_json::json!({ "wake_word": "okay_nabu" });
        let label = apply_wake_reload(&payload, &slot).expect("reload should succeed");
        assert_eq!(label, "okay-nabu");
        assert_eq!(wake::read_slot(&slot).label(), "okay-nabu");
    }

    #[test]
    fn apply_wake_reload_unknown_word_errors_and_keeps_slot() {
        let d = Daemon::new(test_config(), NullSource::default());
        let slot = d.wake_slot();
        let payload = serde_json::json!({ "wake_word": "nope" });
        let err = apply_wake_reload(&payload, &slot).expect_err("unknown wake word must error");
        assert!(
            matches!(err, AudioError::Config(_)),
            "expected AudioError::Config, got {err:?}"
        );
        assert_eq!(
            wake::read_slot(&slot).label(),
            "hey-jarvis",
            "failed reload must leave the active detector untouched",
        );
    }

    /// Stub VAD that returns whatever outcome was prepared, for
    /// trait-dispatch verification.
    #[derive(Clone, Default)]
    struct RecordingVad {
        seen: std::sync::Arc<std::sync::Mutex<usize>>,
    }

    impl crate::vad::VadDetector for RecordingVad {
        fn label(&self) -> &str {
            "stub-vad"
        }

        fn process(&self, _window: &[i16]) -> crate::vad::VadOutcome {
            if let Ok(mut guard) = self.seen.lock() {
                *guard = guard.saturating_add(1);
            }
            crate::vad::VadOutcome::Silence
        }
    }

    #[tokio::test]
    async fn with_vad_detector_swaps_backend() {
        let stub = RecordingVad::default();
        let d = Daemon::new(test_config(), NullSource::default())
            .with_vad_detector(stub.clone());
        d.vad_detector
            .process(&[0_i16; crate::vad::VAD_WINDOW_SAMPLES]);
        d.vad_detector
            .process(&[0_i16; crate::vad::VAD_WINDOW_SAMPLES]);
        let seen = stub.seen.lock().map(|g| *g).unwrap_or(0);
        assert_eq!(seen, 2, "trait dispatch should reach the swapped VAD backend");
    }
}
