//! Integration test for `Daemon`'s VAD edge publishing path.
//!
//! Spins up an in-process `agorabus` daemon on a temp socket, points
//! the wm-audio daemon at it, drives a `NullSource` through a
//! `ScriptedVadDetector` that asserts a rising edge then enough silence
//! to confirm a falling edge, and verifies the subscriber receives one
//! `wm.audio.speech.start`, ≥1 `wm.audio.speech.chunk`, and one
//! `wm.audio.speech.end`. Locks down the PRD §2.3 contract end-to-end
//! against a real bus.

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
    clippy::float_arithmetic
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use agorabus::{Client, DaemonConfig, run_daemon};
use tokio::time::timeout;

use wintermute_audio::config::{Config, WakeWord};
use wintermute_audio::daemon::Daemon;
use wintermute_audio::source::NullSource;
use wintermute_audio::vad::{VadDetector, VadOutcome};

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Wrap in a private subdir: the agorabus daemon chmods the socket's
    // parent to 0700 on bind, which silently fails (or worse, would be
    // wrong) if pointed at /tmp directly.
    let dir = std::env::temp_dir().join(format!("wm-audio-test-{pid}-{nanos}"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{tag}.{ext}"))
}

/// Returns `Speech { probability: 0.9 }` for the first `speech_windows`
/// calls, `Silence` thereafter. Drives the daemon's VAD path through a
/// rising edge → mid-utterance chunks → 500 ms hangover → falling edge.
struct ScriptedVad {
    counter: AtomicUsize,
    speech_windows: usize,
}

impl VadDetector for ScriptedVad {
    fn label(&self) -> &'static str {
        "scripted"
    }

    fn process(&self, _window: &[i16]) -> VadOutcome {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        if n < self.speech_windows {
            VadOutcome::Speech { probability: 0.9 }
        } else {
            VadOutcome::Silence
        }
    }
}

async fn run_speech_lifecycle() -> Result<(), String> {
    // 1. Spawn an in-process agorabus on a unique temp socket.
    let bus_sock = tmp_path("bus", "sock");
    let _ = std::fs::remove_file(&bus_sock);
    let bus_cfg = DaemonConfig {
        socket_path: bus_sock.clone(),
        heartbeat_timeout: Duration::from_secs(60),
        broadcast_capacity: 1024,
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

    // 2. Subscribe a client to the speech prefix before the daemon starts
    //    emitting, so the broadcast channel can't race past us.
    let mut subscriber = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("subscriber connect: {e:#}"))?;
    subscriber
        .announce("vad-bus-smoke-sub", std::process::id(), "", "test-subscriber")
        .await
        .map_err(|e| format!("subscriber announce: {e:#}"))?;
    subscriber
        .subscribe("wm.audio.speech.")
        .await
        .map_err(|e| format!("subscriber subscribe: {e:#}"))?;

    // 3. Build a Config bypassing env, pointing at our temp bus + a
    //    unique fanout socket so concurrent test runs don't collide.
    let mic_sock = tmp_path("mic", "sock");
    let _ = std::fs::remove_file(&mic_sock);
    let config = Config {
        mic_node: String::new(),
        wake_word: WakeWord::HeyJarvis,
        wake_threshold: 0.6,
        mic_socket: mic_sock.clone(),
        bus_socket: bus_sock.clone(),
        session_id: format!("wm-audio-vad-smoke-{}", std::process::id()),
        pw_record_bin: wintermute_audio::DEFAULT_PW_RECORD.to_owned(),
    };

    // 4. NullSource frame math:
    //    - VAD window is 512 samples (32 ms at 16 kHz).
    //    - ScriptedVad scripts 2 speech windows, then silence forever.
    //    - VAD tracker hangover is 500 ms → needs ~16 silent windows
    //      after the first silent window before emitting End.
    //    - So we need ≥18 complete VAD windows = ≥9216 samples.
    //    - NullSource emits 320 samples per frame; 50 frames = 16000
    //      samples = ~31 VAD windows. Comfortable margin.
    let source = NullSource {
        frames: 50,
        frame_size: 320,
    };
    let daemon = Daemon::new(config, source).with_vad_detector(ScriptedVad {
        counter: AtomicUsize::new(0),
        speech_windows: 2,
    });

    let daemon_task = tokio::spawn(async move { daemon.run().await });

    // 5. Drain the subscriber until we see SpeechEnd, or time out.
    //    NullSource drains in microseconds, then the daemon main loop's
    //    `frame_rx.recv()` returns None and the loop breaks. All
    //    publishes are awaited inline before that, so the bus has them
    //    by the time we collect.
    let mut starts = 0_u32;
    let mut chunks = 0_u32;
    let mut end_duration_ms: Option<u32> = None;
    let collect_deadline = Duration::from_secs(10);
    let collected = timeout(collect_deadline, async {
        loop {
            match subscriber.next_event().await {
                Ok(Some(ev)) => match ev.topic.as_str() {
                    "wm.audio.speech.start" => starts += 1,
                    "wm.audio.speech.chunk" => chunks += 1,
                    "wm.audio.speech.end" => {
                        end_duration_ms = ev
                            .data
                            .get("duration_ms")
                            .and_then(serde_json::Value::as_u64)
                            .and_then(|n| u32::try_from(n).ok());
                        return Ok::<(), String>(());
                    }
                    other => return Err(format!("unexpected topic: {other}")),
                },
                Ok(None) => return Err("subscriber EOF before SpeechEnd".into()),
                Err(e) => return Err(format!("next_event: {e:#}")),
            }
        }
    })
    .await;

    // 6. Tear down regardless of test outcome so the test process exits.
    let _ = bus_shutdown_tx.send(());
    let _ = bus_task.await;
    let daemon_outcome = timeout(Duration::from_secs(3), daemon_task).await;

    // Best-effort cleanup of UDS files. The bus task removes its own on
    // shutdown; the fanout server in the daemon also removes its own.
    let _ = std::fs::remove_file(&bus_sock);
    let _ = std::fs::remove_file(&mic_sock);

    let _ = daemon_outcome;
    collected
        .map_err(|_| "timed out collecting events before SpeechEnd arrived".to_string())??;

    // 7. Assertions. Counts are pinned to the deterministic VAD-tracker
    //    arithmetic: 1 rising-edge Start, 2 in-speech chunks + 15
    //    PendingEnd chunks = 17 chunks, 1 falling-edge End with
    //    duration_ms = 64 (2 speech windows × 32 ms).
    if starts != 1 {
        return Err(format!("expected exactly 1 SpeechStart, got {starts}"));
    }
    if chunks == 0 {
        return Err(format!("expected at least 1 SpeechChunk, got {chunks}"));
    }
    let duration = end_duration_ms.ok_or_else(|| "SpeechEnd missing duration_ms".to_string())?;
    if duration != 64 {
        return Err(format!(
            "expected SpeechEnd.duration_ms = 64 (2 speech windows × 32 ms), got {duration}"
        ));
    }
    Ok(())
}

#[test]
fn vad_publishes_start_chunk_end_through_real_bus() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(async {
        run_speech_lifecycle().await.expect("VAD bus lifecycle");
    });
}
