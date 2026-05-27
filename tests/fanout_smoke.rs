//! Integration test for `wintermute_audio::fanout`.
//!
//! AC7 from PRD-wintermute-audio.md: two simultaneous `mic.sock`
//! subscribers consume the same PCM stream. We verify the byte-for-byte
//! identical stream property (full 60-minute checksum lives outside CI).

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::as_conversions,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::missing_assert_message,
    clippy::float_arithmetic,
    clippy::similar_names
)]

use std::path::PathBuf;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::net::UnixStream;
use tokio::time::{sleep, timeout};

use wintermute_audio::fanout::{self, channel};
use wintermute_audio::source::PcmFrame;
use wintermute_audio::state::Shutdown;

fn tmp_sock(tag: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("wm-audio-{tag}-{pid}-{nanos}.sock"))
}

async fn wait_for_socket(path: &std::path::Path, attempts: u32) -> bool {
    for _ in 0..attempts {
        if path.exists() {
            return true;
        }
        sleep(Duration::from_millis(20)).await;
    }
    false
}

async fn run_two_subscriber_test() -> Result<(), String> {
    let sock = tmp_sock("fanout-two");
    let _ = std::fs::remove_file(&sock);

    let (tx, seed) = channel();
    drop(seed);
    let shutdown = Shutdown::new();

    let server_tx = tx.clone();
    let server_sock = sock.clone();
    let server_shutdown = shutdown.clone();
    let server = tokio::spawn(async move {
        let _ = fanout::run(server_sock, server_tx, server_shutdown).await;
    });

    if !wait_for_socket(&sock, 100).await {
        shutdown.trigger();
        let _ = server.await;
        return Err(format!("fanout socket never bound at {}", sock.display()));
    }

    let mut c1 = match timeout(Duration::from_secs(2), UnixStream::connect(&sock)).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return Err(format!("connect c1: {e}")),
        Err(_) => return Err("connect c1 timed out".into()),
    };
    let mut c2 = match timeout(Duration::from_secs(2), UnixStream::connect(&sock)).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return Err(format!("connect c2: {e}")),
        Err(_) => return Err("connect c2 timed out".into()),
    };

    // Allow the server accept loop to register both subscribers before
    // we begin publishing. Broadcasts that fire before `subscribe()` are
    // not delivered to that subscriber.
    sleep(Duration::from_millis(200)).await;

    // Push 4 frames * 320 samples = 1280 samples per subscriber.
    let frames = 4_u16;
    let samples_per_frame = 320_usize;
    for i in 0..frames {
        let payload: Vec<i16> = (0..samples_per_frame)
            .map(|s| {
                let raw = (u32::from(i) << 8) | (s as u32 & 0xFF);
                (raw as i16).wrapping_add(0)
            })
            .collect();
        let frame = PcmFrame {
            ts_ms: u64::from(i),
            samples: payload,
        };
        tx.send(frame).map_err(|e| format!("bcast send: {e}"))?;
    }

    let want_bytes = usize::from(frames) * samples_per_frame * 2;
    let mut buf1 = vec![0_u8; want_bytes];
    let mut buf2 = vec![0_u8; want_bytes];

    let read1 = timeout(Duration::from_secs(3), c1.read_exact(&mut buf1)).await;
    let read2 = timeout(Duration::from_secs(3), c2.read_exact(&mut buf2)).await;
    let n1 = read1
        .map_err(|_| "c1 read timed out".to_string())?
        .map_err(|e| format!("c1 read: {e}"))?;
    let n2 = read2
        .map_err(|_| "c2 read timed out".to_string())?
        .map_err(|e| format!("c2 read: {e}"))?;
    if n1 != want_bytes || n2 != want_bytes {
        return Err(format!(
            "short read: c1={n1} c2={n2} want={want_bytes}"
        ));
    }
    if buf1 != buf2 {
        // Find first divergence for the error report.
        let diff = buf1
            .iter()
            .zip(buf2.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        return Err(format!(
            "subscriber streams diverge at byte {diff}: c1[{}]={} c2[{}]={}",
            diff, buf1[diff], diff, buf2[diff]
        ));
    }

    // Sanity-check the wire format: the second byte of every 640-byte
    // frame block encodes the frame index (high byte of the i16 little-
    // endian sample value our generator uses).
    for i in 0..usize::from(frames) {
        let high_byte = buf1[i * samples_per_frame * 2 + 1];
        if high_byte != u8::try_from(i).unwrap_or(0) {
            return Err(format!(
                "frame {i} high-byte mismatch: got {high_byte}, want {i}"
            ));
        }
    }

    shutdown.trigger();
    let _ = timeout(Duration::from_secs(2), server).await;
    Ok(())
}

#[tokio::test]
async fn two_subscribers_receive_identical_pcm_bytes() {
    let r = run_two_subscriber_test().await;
    assert!(r.is_ok(), "fanout AC7 smoke: {r:?}");
}

async fn run_three_subscriber_long_stream_test() -> Result<(), String> {
    let sock = tmp_sock("fanout-three");
    let _ = std::fs::remove_file(&sock);

    let (tx, seed) = channel();
    drop(seed);
    let shutdown = Shutdown::new();

    let server_tx = tx.clone();
    let server_sock = sock.clone();
    let server_shutdown = shutdown.clone();
    let server = tokio::spawn(async move {
        let _ = fanout::run(server_sock, server_tx, server_shutdown).await;
    });

    if !wait_for_socket(&sock, 100).await {
        shutdown.trigger();
        let _ = server.await;
        return Err(format!("fanout socket never bound at {}", sock.display()));
    }

    let mut clients = Vec::with_capacity(3);
    for label in ["c1", "c2", "c3"] {
        let s = match timeout(Duration::from_secs(2), UnixStream::connect(&sock)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(format!("connect {label}: {e}")),
            Err(_) => return Err(format!("connect {label} timed out")),
        };
        clients.push(s);
    }

    // Give the accept loop time to register all three subscribers on the
    // broadcast channel before any sends fire. Same gate the 2-subscriber
    // test uses; 3 connections need a touch more headroom.
    sleep(Duration::from_millis(300)).await;

    // 256 frames * 320 samples = 81920 samples = ~5.12 s of 16 kHz audio.
    // AC7 calls for 60-minute streams; full duration lives outside CI but
    // a multi-second run exercises broadcast capacity, per-subscriber
    // writer task drain, and FIFO order across many sends.
    let frames = 256_u32;
    let samples_per_frame = 320_usize;
    for i in 0..frames {
        // Use a 16-bit-friendly counter that wraps deterministically so
        // the per-frame high-byte check at the end still anchors order.
        let frame_tag = (i & 0xFF) as u16;
        let payload: Vec<i16> = (0..samples_per_frame)
            .map(|s| {
                let raw = (u32::from(frame_tag) << 8) | (s as u32 & 0xFF);
                raw as i16
            })
            .collect();
        let frame = PcmFrame {
            ts_ms: u64::from(i),
            samples: payload,
        };
        tx.send(frame).map_err(|e| format!("bcast send: {e}"))?;
    }

    let want_bytes = (frames as usize) * samples_per_frame * 2;
    let mut buf1 = vec![0_u8; want_bytes];
    let mut buf2 = vec![0_u8; want_bytes];
    let mut buf3 = vec![0_u8; want_bytes];
    let mut c1 = clients.remove(0);
    let mut c2 = clients.remove(0);
    let mut c3 = clients.remove(0);

    // Read all three subscribers concurrently so a slow subscriber can't
    // serialize the test wall-clock. Each gets its own bounded read.
    let (r1, r2, r3) = tokio::join!(
        timeout(Duration::from_secs(8), c1.read_exact(&mut buf1)),
        timeout(Duration::from_secs(8), c2.read_exact(&mut buf2)),
        timeout(Duration::from_secs(8), c3.read_exact(&mut buf3)),
    );
    let n1 = r1
        .map_err(|_| "c1 read timed out".to_string())?
        .map_err(|e| format!("c1 read: {e}"))?;
    let n2 = r2
        .map_err(|_| "c2 read timed out".to_string())?
        .map_err(|e| format!("c2 read: {e}"))?;
    let n3 = r3
        .map_err(|_| "c3 read timed out".to_string())?
        .map_err(|e| format!("c3 read: {e}"))?;
    if n1 != want_bytes || n2 != want_bytes || n3 != want_bytes {
        return Err(format!(
            "short read: c1={n1} c2={n2} c3={n3} want={want_bytes}"
        ));
    }

    // Byte-equal across all three subscribers. Pairwise check pinpoints
    // which pair diverges if any do.
    let bufs = [&buf1, &buf2, &buf3];
    for (a_idx, b_idx) in [(0, 1), (0, 2), (1, 2)] {
        if bufs[a_idx] != bufs[b_idx] {
            let diff = bufs[a_idx]
                .iter()
                .zip(bufs[b_idx].iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(format!(
                "subscriber c{} vs c{} diverge at byte {diff} (out of {want_bytes})",
                a_idx + 1,
                b_idx + 1
            ));
        }
    }

    // FIFO order check: per-frame high-byte cycles 0..=255 then wraps.
    for i in 0..(frames as usize) {
        let high_byte = buf1[i * samples_per_frame * 2 + 1];
        let want = (i & 0xFF) as u8;
        if high_byte != want {
            return Err(format!(
                "frame {i} order/identity mismatch: got high-byte {high_byte}, want {want}"
            ));
        }
    }

    shutdown.trigger();
    let _ = timeout(Duration::from_secs(2), server).await;
    Ok(())
}

#[tokio::test]
async fn three_subscribers_receive_identical_pcm_bytes_over_long_stream() {
    let r = run_three_subscriber_long_stream_test().await;
    assert!(r.is_ok(), "fanout AC7 long-stream smoke: {r:?}");
}

#[tokio::test]
async fn fanout_listener_cleans_socket_on_shutdown() {
    let sock = tmp_sock("fanout-clean");
    let _ = std::fs::remove_file(&sock);

    let (tx, seed) = channel();
    drop(seed);
    let shutdown = Shutdown::new();
    let server_sock = sock.clone();
    let server_shutdown = shutdown.clone();
    let server = tokio::spawn(async move {
        let _ = fanout::run(server_sock, tx, server_shutdown).await;
    });

    assert!(
        wait_for_socket(&sock, 100).await,
        "socket should bind before shutdown"
    );

    shutdown.trigger();
    let _ = timeout(Duration::from_secs(3), server).await;

    // Listener removes the socket file on exit.
    assert!(
        !sock.exists(),
        "fanout should remove its socket on shutdown, still present at {}",
        sock.display()
    );
}
