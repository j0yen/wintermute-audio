//! Top-level daemon: wires the mic source, ring fanout, agorabus
//! publisher/subscriber, and graceful-shutdown loop.

use crate::config::Config;
use crate::errors::AudioError;
use crate::events::{AudioEvent, ControlEvent, Timestamp, WakeDetected};
use crate::fanout;
use crate::source::{MicSource, PcmFrame};
use crate::state::{MuteReason, MuteState, Shutdown};
use crate::wake::{NullWakeDetector, WakeDetector, WakeOutcome, WakeWindow};

use agorabus::Client;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer as _, Observer as _, Producer as _, Split as _};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Bounded ring-buffer capacity in i16 samples (~5 s of 16 kHz mono).
pub const RING_CAPACITY: usize = 16_000 * 5;
/// Channel depth for the source -> daemon hand-off.
const SOURCE_CHANNEL_DEPTH: usize = 64;

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
    wake_detector: Arc<dyn WakeDetector>,
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
        Self {
            config,
            source,
            mute: MuteState::new(),
            shutdown: Shutdown::new(),
            wake_detector: Arc::new(NullWakeDetector::new(label)),
        }
    }

    /// Swap in a wake-word backend. Returns `self` for the builder
    /// pattern. The backend MUST be `Send + Sync` (enforced by the
    /// trait's supertraits).
    #[must_use]
    pub fn with_wake_detector<W: WakeDetector + 'static>(mut self, detector: W) -> Self {
        self.wake_detector = Arc::new(detector);
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
    #[allow(clippy::too_many_lines)]
    pub async fn run(self) -> Result<(), AudioError> {
        let Self {
            config,
            source,
            mute,
            shutdown,
            wake_detector,
        } = self;

        info!(
            session = %config.session_id,
            mic = %config.mic_node,
            wake = %config.wake_word.as_label(),
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
        let control_task = tokio::spawn(async move {
            run_control_loop(sub_client, control_mute, control_shutdown).await;
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
        let mut wake_window = WakeWindow::with_defaults();
        let mut wake_was_active = mute.should_run_wake();
        loop {
            if shutdown.is_triggered() {
                info!("shutdown requested, draining");
                break;
            }
            let frame = match tokio::time::timeout(
                std::time::Duration::from_millis(250),
                frame_rx.recv(),
            )
            .await
            {
                Ok(Some(f)) => f,
                Ok(None) => {
                    debug!("capture source closed");
                    break;
                }
                Err(_) => continue, // timeout — re-check shutdown flag
            };

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

            // Fanout to UDS subscribers. `SendError` only means "no
            // active subscribers right now," which is the steady state
            // when nothing is listening — ignore.
            let _ = bcast_tx.send(frame);

            // Drain the in-process ring into the wake window. On a
            // TTS-mute edge, flush the buffer so the next inference
            // does not run on stale audio from before un-mute.
            let avail = consumer.occupied_len();
            if avail > 0 {
                let wake_active = mute.should_run_wake();
                if wake_active && !wake_was_active {
                    wake_window.clear();
                }
                wake_was_active = wake_active;

                let mut scratch = vec![0_i16; avail];
                let popped = consumer.pop_slice(&mut scratch);
                scratch.truncate(popped);
                if wake_active {
                    wake_window.push(&scratch);
                }
            }

            // Run inference on every complete window that accumulated.
            while let Some(window) = wake_window.next_window() {
                if !mute.should_run_wake() {
                    continue;
                }
                let WakeOutcome::Detected { confidence } = wake_detector.process(&window) else {
                    continue;
                };
                if confidence < config.wake_threshold {
                    continue;
                }
                let ev = AudioEvent::Wake(WakeDetected {
                    wake_word: wake_detector.label().to_owned(),
                    confidence,
                    ts: Timestamp::now(),
                });
                match ev.payload() {
                    Ok(payload) => {
                        if let Err(e) = pub_client.publish(ev.topic(), payload).await {
                            warn!(error = %e, "publish wake failed");
                        }
                    }
                    Err(e) => warn!(error = %e, "serialize wake payload failed"),
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
        let _ = source_task.await;

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

async fn run_control_loop(mut sub: Client, mute: MuteState, shutdown: Shutdown) {
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
        match ControlEvent::from_topic(&ev.topic) {
            Some(ControlEvent::TtsStart) => mute.set(MuteReason::TtsActive, true),
            Some(ControlEvent::TtsEnd) => mute.set(MuteReason::TtsActive, false),
            Some(ControlEvent::MuteRequest) => mute.set(MuteReason::DialogRequest, true),
            Some(ControlEvent::UnmuteRequest) => mute.set(MuteReason::DialogRequest, false),
            Some(ControlEvent::Reload) => {
                info!("reload requested; iter-3 will hot-swap ONNX here");
            }
            None => debug!(topic = %ev.topic, "ignored bus event"),
        }
    }
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
        // Use the stored detector to demonstrate the trait object is
        // callable through the daemon's clone of the Arc.
        d.wake_detector.process(&[0_i16; WAKE_WINDOW_SAMPLES]);
        d.wake_detector.process(&[0_i16; WAKE_WINDOW_SAMPLES]);
        let seen = stub.seen.lock().map(|g| *g).unwrap_or(0);
        assert_eq!(seen, 2, "trait dispatch should reach the swapped backend");
    }
}
