//! Integration test for the `wm.audio.reload` hot-swap path (PRD AC6).
//!
//! Spins up an in-process `agorabus` daemon on a temp socket, points
//! the wm-audio daemon at it, grabs a handle to the wake slot, then
//! publishes a `wm.audio.reload` event with a new `wake_word`. Verifies
//! the daemon's control loop swaps the active detector in the slot.
//!
//! The src-level `apply_wake_reload_*` unit tests exercise the helper
//! directly; this test pins down the bus-side wire end-to-end.

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
    clippy::missing_errors_doc
)]

use std::path::PathBuf;
use std::time::Duration;

use agorabus::{Client, DaemonConfig, run_daemon};
use serde_json::json;
use tokio::time::timeout;

use wintermute_audio::config::{Config, WakeWord};
use wintermute_audio::daemon::Daemon;
use wintermute_audio::source::NullSource;
use wintermute_audio::wake;

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Same private-subdir trick as vad_bus_smoke: the agorabus daemon
    // chmods the socket's parent to 0700 on bind, which silently goes
    // wrong if pointed at /tmp directly.
    let dir = std::env::temp_dir().join(format!("wm-audio-test-{pid}-{nanos}"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{tag}.{ext}"))
}

async fn run_reload_lifecycle() -> Result<(), String> {
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

    // 2. Build a publisher client. Subscriptions are not needed; we
    //    only publish wm.audio.reload from this side.
    let mut publisher = Client::connect(&bus_sock)
        .await
        .map_err(|e| format!("publisher connect: {e:#}"))?;
    publisher
        .announce(
            "wake-reload-smoke-pub",
            std::process::id(),
            "",
            "test-publisher",
        )
        .await
        .map_err(|e| format!("publisher announce: {e:#}"))?;

    // 3. Daemon config bypassing env, pointing at our temp bus + a
    //    unique fanout socket.
    let mic_sock = tmp_path("mic", "sock");
    let _ = std::fs::remove_file(&mic_sock);
    let config = Config {
        mic_node: String::new(),
        wake_word: WakeWord::HeyJarvis,
        wake_threshold: 0.6,
        mic_socket: mic_sock.clone(),
        bus_socket: bus_sock.clone(),
        session_id: format!("wm-audio-reload-smoke-{}", std::process::id()),
    };

    // 4. Enough frames that the daemon's main loop is still alive while
    //    we publish + poll. 200 frames × 320 samples = 64 000 samples
    //    ≈ 4 s at 16 kHz; NullSource emits with no real sleep, so this
    //    drains quickly, but the control task keeps running until the
    //    daemon shuts down regardless.
    let source = NullSource {
        frames: 200,
        frame_size: 320,
    };
    let daemon = Daemon::new(config, source);
    let slot = daemon.wake_slot();

    // Pre-condition: the default detector label is the configured wake.
    let pre = wake::read_slot(&slot).label().to_owned();
    if pre != "hey-jarvis" {
        return Err(format!(
            "pre-condition: default detector label should be hey-jarvis, got {pre}"
        ));
    }

    let daemon_task = tokio::spawn(async move { daemon.run().await });

    // 5. Give the daemon time to connect its control sub_client and
    //    subscribe to wm.audio.reload. The connect/announce/subscribe
    //    calls inside daemon.run() are awaited, so once they complete
    //    the broadcast is in place — but we cannot await the spawned
    //    task without consuming it, so we publish in a small retry
    //    loop. Each publish is cheap and the apply_wake_reload helper
    //    is idempotent for repeated identical payloads.
    let observed = timeout(Duration::from_secs(5), async {
        loop {
            let _ = publisher
                .publish("wm.audio.reload", json!({ "wake_word": "okay_nabu" }))
                .await;
            // Give the daemon's 500ms control-loop timeout a chance to
            // see the broadcast, then poll a few times.
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let label = wake::read_slot(&slot).label().to_owned();
                if label != pre {
                    return label;
                }
            }
        }
    })
    .await;

    // 6. Tear down regardless of outcome so the test process can exit.
    let _ = bus_shutdown_tx.send(());
    let _ = bus_task.await;
    let _ = timeout(Duration::from_secs(3), daemon_task).await;
    let _ = std::fs::remove_file(&bus_sock);
    let _ = std::fs::remove_file(&mic_sock);

    let new_label = observed.map_err(|_| {
        format!("wake slot label never changed from {pre} within 5s after publishing reload")
    })?;
    if new_label != "okay-nabu" {
        return Err(format!(
            "expected wake slot to swap to 'okay-nabu', got '{new_label}'"
        ));
    }
    Ok(())
}

#[test]
fn reload_event_swaps_wake_detector_label_through_real_bus() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(async {
        run_reload_lifecycle()
            .await
            .expect("reload hot-swap lifecycle");
    });
}
