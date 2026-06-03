//! Integration test for `Daemon`'s wake-event publishing path.
//!
//! Spins up an in-process `agorabus` daemon on a temp socket, points
//! the wm-audio daemon at it, drives a [`NullSource`] through a
//! [`ScriptedWake`] detector that fires [`WakeOutcome::Detected`] on a
//! specific window index, and verifies the subscriber receives exactly
//! one `wm.audio.wake` event with the configured wake word and the
//! detector's confidence. Locks down PRD §2.3 step 3 wake publish path
//! end-to-end against a real bus — mirrors the `vad_bus_smoke` /
//! `reload_bus_smoke` patterns.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::as_conversions,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::missing_assert_message,
    clippy::missing_errors_doc,
    clippy::float_arithmetic,
    clippy::float_cmp
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agorabus::{Client, DaemonConfig, run_daemon};
use tokio::time::timeout;

use wintermute_audio::config::{Config, WakeWord};
use wintermute_audio::daemon::Daemon;
use wintermute_audio::features::{MEL_STRIDE_SAMPLES, MEL_WINDOW_SAMPLES};
use wintermute_audio::source::NullSource;
use wintermute_audio::wake::{WakeDetector, WakeOutcome};

/// `NullSource` frame count that drains enough PCM through the daemon's
/// [`wintermute_audio::features::MelWindowBuffer`] for the scripted detector
/// to fire on its 3rd `process` call.
///
/// The v0.7.0 mel front-end sized the wake window at
/// [`MEL_WINDOW_SAMPLES`] (30 240 samples / 1.89 s) advanced by
/// [`MEL_STRIDE_SAMPLES`] (2 560 samples / 160 ms) — far larger than the old
/// 1 280-sample window these tests were originally written against. Three
/// complete windows therefore need
/// `MEL_WINDOW_SAMPLES + 2 * MEL_STRIDE_SAMPLES` = 35 360 samples; with a
/// 320-sample `NullSource` frame that is 110.5 frames. We round up and add
/// margin so the detector reliably reaches call 3.
const WAKE_SMOKE_FRAME_SIZE: usize = 320;
const WAKE_SMOKE_FRAMES: usize = {
    let needed = MEL_WINDOW_SAMPLES + 2 * MEL_STRIDE_SAMPLES;
    // ceil-div by frame size, then a small margin for the warmup window.
    needed.div_ceil(WAKE_SMOKE_FRAME_SIZE) + 20
};

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Same private-subdir trick as vad_bus_smoke: agorabus chmods the
    // socket's parent to 0700 on bind, which silently goes wrong if
    // pointed at /tmp directly.
    let dir = std::env::temp_dir().join(format!("wm-audio-test-{pid}-{nanos}"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{tag}.{ext}"))
}

/// Fires `WakeOutcome::Detected { confidence }` on its `fire_on_call`
/// invocation (1-indexed), `NotDetected` on every other call. Lets the
/// test pin down exactly one wake event in the stream so the
/// subscriber-side assertion is unambiguous.
struct ScriptedWake {
    counter: AtomicUsize,
    fire_on_call: usize,
    confidence: f32,
}

impl WakeDetector for ScriptedWake {
    fn label(&self) -> &'static str {
        "scripted-wake"
    }

    fn process(&self, _window: &[i16]) -> WakeOutcome {
        let n = self.counter.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        if n == self.fire_on_call {
            WakeOutcome::Detected {
                confidence: self.confidence,
            }
        } else {
            WakeOutcome::NotDetected
        }
    }
}

async fn run_wake_lifecycle() -> Result<(), String> {
    // 1. Spawn an in-process agorabus on a unique temp socket.
    let bus_sock = tmp_path("bus", "sock");
    let _ = std::fs::remove_file(&bus_sock);
    let bus_cfg = DaemonConfig {
        socket_path: bus_sock.clone(),
        heartbeat_timeout: Duration::from_secs(60),
        broadcast_capacity: 1024,
        drain_grace_ms: agorabus::DEFAULT_DRAIN_GRACE_MS,
        drain_resume_hint_ms: agorabus::DEFAULT_DRAIN_RESUME_HINT_MS,
    };
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (bus_shutdown_tx, bus_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let bus_task = tokio::spawn(async move {
        let _ = run_daemon(bus_cfg, Some(ready_tx), bus_shutdown_rx).await;
    });
    timeout(Duration::from_secs(2), ready_rx)
        .await
        .map_err(|_| "bus never signalled ready".to_string())?
        .map_err(|e| format!("bus ready_tx dropped: {e}"))?;

    // 2. Subscribe before the daemon starts emitting so the broadcast
    //    channel can't race past us.
    let mut subscriber = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("subscriber connect: {e:#}"))?;
    subscriber
        .announce("wake-bus-smoke-sub", std::process::id(), "", "test-subscriber")
        .await
        .map_err(|e| format!("subscriber announce: {e:#}"))?;
    subscriber
        .subscribe("wm.audio.wake")
        .await
        .map_err(|e| format!("subscriber subscribe: {e:#}"))?;

    // 3. Build a Config bypassing env. Threshold 0.6 — script the
    //    detector's confidence above this so the publish gate passes.
    let mic_sock = tmp_path("mic", "sock");
    let _ = std::fs::remove_file(&mic_sock);
    let config = Config {
        mic_node: String::new(),
        wake_word: WakeWord::HeyJarvis,
        wake_threshold: 0.6,
        mic_socket: mic_sock.clone(),
        bus_socket: bus_sock.clone(),
        session_id: format!("wm-audio-wake-smoke-{}", std::process::id()),
        pw_record_bin: wintermute_audio::DEFAULT_PW_RECORD.to_owned(),
        speech_end_silence_ms: wintermute_audio::SPEECH_SILENCE_MS_DEFAULT,
    };

    // 4. NullSource frame math (v0.7.0 mel front-end geometry):
    //    - Wake window is MEL_WINDOW_SAMPLES (30 240 samples / 1.89 s),
    //      advanced by MEL_STRIDE_SAMPLES (2 560 samples / 160 ms).
    //    - NullSource emits WAKE_SMOKE_FRAME_SIZE (320) samples per frame.
    //    - WAKE_SMOKE_FRAMES is sized to drain >=3 complete windows so the
    //      detector reaches its 3rd `process` call; both NotDetected and
    //      Detected paths are exercised before the event lands.
    let source = NullSource {
        frames: WAKE_SMOKE_FRAMES,
        frame_size: WAKE_SMOKE_FRAME_SIZE,
    };
    let daemon = Daemon::new(config, source).with_wake_detector(ScriptedWake {
        counter: AtomicUsize::new(0),
        fire_on_call: 3,
        confidence: 0.9,
    });

    let daemon_task = tokio::spawn(async move { daemon.run().await });

    // 5. Drain subscriber. Collect every wake event until either the
    //    bus closes, a tight per-event poll times out (meaning no more
    //    are coming), or the overall deadline trips. Collecting >1 is
    //    the bug we want this test to catch — ScriptedWake fires once.
    let mut events: Vec<(String, f64)> = Vec::new();
    let collect_deadline = Duration::from_secs(10);
    let per_event_quiet = Duration::from_millis(800);
    let collected = timeout(collect_deadline, async {
        loop {
            match timeout(per_event_quiet, subscriber.next_event()).await {
                Ok(Ok(Some(ev))) => {
                    if ev.topic != "wm.audio.wake" {
                        return Err(format!("unexpected topic: {}", ev.topic));
                    }
                    let wake_word = ev
                        .data
                        .get("wake_word")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "payload missing wake_word".to_string())?
                        .to_owned();
                    let conf = ev
                        .data
                        .get("confidence")
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| "payload missing confidence".to_string())?;
                    events.push((wake_word, conf));
                }
                Ok(Ok(None)) | Err(_) => break, // bus closed or quiet long enough
                Ok(Err(e)) => return Err(format!("next_event: {e:#}")),
            }
            if events.len() > 4 {
                // Safety valve so a runaway publish loop doesn't hang
                // forever; the count assertion below still fires.
                break;
            }
        }
        Ok::<Vec<(String, f64)>, String>(events)
    })
    .await;

    // 6. Tear down regardless of test outcome so the process exits.
    let _ = bus_shutdown_tx.send(());
    let _ = bus_task.await;
    let daemon_outcome = timeout(Duration::from_secs(3), daemon_task).await;
    let _ = std::fs::remove_file(&bus_sock);
    let _ = std::fs::remove_file(&mic_sock);
    let _ = daemon_outcome;

    let observed = collected
        .map_err(|_| "timed out collecting events overall".to_string())??;

    // 7. Assertions. Exactly one wake, wake_word matches the configured
    //    label (Daemon stamps the active detector's .label() — for the
    //    scripted detector that's "scripted-wake"), confidence round-
    //    trips as ~0.9 (f32 → JSON → f64; equality is within 1e-6).
    if observed.len() != 1 {
        return Err(format!(
            "expected exactly 1 wake event, got {}: {observed:?}",
            observed.len()
        ));
    }
    let (label, conf) = &observed[0];
    if label != "scripted-wake" {
        return Err(format!(
            "expected wake_word=scripted-wake, got {label:?}"
        ));
    }
    if (conf - 0.9).abs() > 1e-6 {
        return Err(format!(
            "expected confidence ≈ 0.9, got {conf}"
        ));
    }
    Ok(())
}

#[test]
fn wake_publishes_through_real_bus() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(async {
        run_wake_lifecycle().await.expect("wake bus lifecycle");
    });
}

/// Detector variant that stamps an [`Instant`] into a shared cell on the
/// call that fires. Lets the timing test compute
/// `event_received - fired_at`, isolating the detect→publish→subscribe
/// path from daemon startup latency.
struct TimedWake {
    counter: AtomicUsize,
    fire_on_call: usize,
    confidence: f32,
    fired_at: Arc<Mutex<Option<Instant>>>,
}

impl WakeDetector for TimedWake {
    fn label(&self) -> &'static str {
        "timed-wake"
    }

    fn process(&self, _window: &[i16]) -> WakeOutcome {
        let n = self.counter.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        if n == self.fire_on_call {
            // Record the fire instant *before* returning so the test
            // measures from the earliest possible point — anything in
            // the daemon's publish path (event encode, pub_client
            // send, bus broadcast) is included in the AC3 budget.
            if let Ok(mut guard) = self.fired_at.lock() {
                *guard = Some(Instant::now());
            }
            WakeOutcome::Detected {
                confidence: self.confidence,
            }
        } else {
            WakeOutcome::NotDetected
        }
    }
}

/// PRD AC3: wake → `wm.audio.wake` event published latency: <200 ms.
///
/// Distinct from `wake_publishes_through_real_bus`, which verifies the
/// publish *path* end-to-end. This one measures the detect→subscribe-
/// receive latency under PRD AC3's 200 ms budget. Mirrors the iter-15
/// AC6 reload-timing test against the wake topic.
async fn run_wake_timing() -> Result<Duration, String> {
    let bus_sock = tmp_path("bus", "sock");
    let _ = std::fs::remove_file(&bus_sock);
    let bus_cfg = DaemonConfig {
        socket_path: bus_sock.clone(),
        heartbeat_timeout: Duration::from_secs(60),
        broadcast_capacity: 1024,
        drain_grace_ms: agorabus::DEFAULT_DRAIN_GRACE_MS,
        drain_resume_hint_ms: agorabus::DEFAULT_DRAIN_RESUME_HINT_MS,
    };
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (bus_shutdown_tx, bus_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let bus_task = tokio::spawn(async move {
        let _ = run_daemon(bus_cfg, Some(ready_tx), bus_shutdown_rx).await;
    });
    timeout(Duration::from_secs(2), ready_rx)
        .await
        .map_err(|_| "bus never signalled ready".to_string())?
        .map_err(|e| format!("bus ready_tx dropped: {e}"))?;

    // Subscribe BEFORE the daemon spawns so the bus has our session in
    // its broadcast table when the daemon publishes. The Client buffers
    // incoming events internally, so even if the publish lands before
    // we await next_event(), the message survives and the receive
    // timestamp is captured at the next-event resume point.
    let mut subscriber = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("subscriber connect: {e:#}"))?;
    subscriber
        .announce(
            "wake-timing-sub",
            std::process::id(),
            "",
            "test-subscriber",
        )
        .await
        .map_err(|e| format!("subscriber announce: {e:#}"))?;
    subscriber
        .subscribe("wm.audio.wake")
        .await
        .map_err(|e| format!("subscriber subscribe: {e:#}"))?;

    let mic_sock = tmp_path("mic", "sock");
    let _ = std::fs::remove_file(&mic_sock);
    let config = Config {
        mic_node: String::new(),
        wake_word: WakeWord::HeyJarvis,
        wake_threshold: 0.6,
        mic_socket: mic_sock.clone(),
        bus_socket: bus_sock.clone(),
        session_id: format!("wm-audio-wake-timing-{}", std::process::id()),
        pw_record_bin: wintermute_audio::DEFAULT_PW_RECORD.to_owned(),
        speech_end_silence_ms: wintermute_audio::SPEECH_SILENCE_MS_DEFAULT,
    };

    let fired_at: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    let detector = TimedWake {
        counter: AtomicUsize::new(0),
        fire_on_call: 3,
        confidence: 0.9,
        fired_at: Arc::clone(&fired_at),
    };

    let source = NullSource {
        frames: WAKE_SMOKE_FRAMES,
        frame_size: WAKE_SMOKE_FRAME_SIZE,
    };
    let daemon = Daemon::new(config, source).with_wake_detector(detector);
    let daemon_task = tokio::spawn(async move { daemon.run().await });

    // Drain until the first wake event lands. Capture Instant::now()
    // at the resume point — that's the subscriber-side receive time.
    // Since we only subscribed to wm.audio.wake and the daemon emits
    // exactly one wake (ScriptedWake-style detector that fires once),
    // the first event off the channel IS the one we care about.
    let received_at = timeout(Duration::from_secs(10), async {
        match subscriber.next_event().await {
            Ok(Some(ev)) if ev.topic == "wm.audio.wake" => {
                Ok::<Instant, String>(Instant::now())
            }
            Ok(Some(ev)) => Err(format!("unexpected topic: {}", ev.topic)),
            Ok(None) => Err("subscriber closed before wake event".into()),
            Err(e) => Err(format!("next_event: {e:#}")),
        }
    })
    .await;

    let _ = bus_shutdown_tx.send(());
    let _ = bus_task.await;
    let _ = timeout(Duration::from_secs(3), daemon_task).await;
    let _ = std::fs::remove_file(&bus_sock);
    let _ = std::fs::remove_file(&mic_sock);

    let receive_instant = received_at
        .map_err(|_| "timed out waiting for wake event".to_string())??;
    let fire_instant = fired_at
        .lock()
        .map_err(|e| format!("fired_at mutex poisoned: {e}"))?
        .ok_or_else(|| "detector never recorded fire instant".to_string())?;
    Ok(receive_instant.saturating_duration_since(fire_instant))
}

#[test]
fn wake_publish_within_two_hundred_ms_ac3() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(async {
        let elapsed = run_wake_timing().await.expect("wake timing lifecycle");
        assert!(
            elapsed < Duration::from_millis(200),
            "AC3 violation: wake publish took {elapsed:?}, budget is <200ms",
        );
        // Sanity-check the measurement is plausible. The detect→publish
        // path runs through real bus broadcast, so sub-microsecond is
        // physically implausible and would indicate the timestamps
        // collapsed somehow.
        assert!(
            elapsed >= Duration::from_micros(1),
            "AC3 measurement implausible: {elapsed:?}",
        );
    });
}
