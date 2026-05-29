//! Integration smoke test for the `wm-audio-dump` bin.
//!
//! Spins up a UDS server that mimics the daemon's fanout socket, writes a
//! known PCM payload, then invokes the installed binary against it and
//! checks the resulting WAV file's header + data length.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::UnixListener;
use tokio::process::Command;

#[tokio::test]
async fn dump_writes_wav_with_correct_header_and_payload() {
    let tmp = unique_tempdir("wm-audio-dump-it-1");
    tokio::fs::create_dir_all(&tmp).await.expect("mkdir tmp");
    let sock_path = tmp.join("mic.sock");
    let out_path = tmp.join("out.wav");

    let listener = UnixListener::bind(&sock_path).expect("bind UDS");

    // Server: accept one client, push exactly one second of canned PCM
    // (16000 samples * 2 bytes = 32000 bytes), then drop. dump should
    // see EOF and exit before its 2s deadline.
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut payload = Vec::with_capacity(32_000);
        for i in 0_u32..16_000 {
            let s = ((i & 0x7fff) as i16).to_le_bytes();
            payload.extend_from_slice(&s);
        }
        stream.write_all(&payload).await.expect("write canned pcm");
        // closing the connection on drop signals EOF to the client.
    });

    let bin = env!("CARGO_BIN_EXE_wm-audio-dump");
    let output = Command::new(bin)
        .arg(&out_path)
        .env("WM_MIC_SOCKET", &sock_path)
        .env("WM_DUMP_DURATION_MS", "2000")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .expect("spawn wm-audio-dump");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "dump bin exit: status={status:?} stderr={stderr}",
        status = output.status,
    );
    let _ = tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server task join timed out");

    let bytes = tokio::fs::read(&out_path).await.expect("read out.wav");
    assert!(bytes.len() >= 44, "WAV header missing: {} bytes", bytes.len());
    assert_eq!(&bytes[0..4], b"RIFF", "RIFF magic");
    assert_eq!(&bytes[8..12], b"WAVE", "WAVE magic");
    assert_eq!(&bytes[12..16], b"fmt ", "fmt chunk id");
    assert_eq!(&bytes[36..40], b"data", "data chunk id");

    // PCM format = 1 (LE u16 @ offset 20).
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1, "PCM format");
    // Channels = 1.
    assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "channels");
    // Sample rate = 16000.
    assert_eq!(
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        16_000,
        "sample rate",
    );
    // Bits per sample = 16.
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16, "bits");
    // Declared data size matches actual payload length.
    let declared = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    assert_eq!(declared as usize, bytes.len() - 44, "declared data size");
    // We sent exactly 32000 PCM bytes; dump should have captured all of them.
    assert_eq!(bytes.len(), 44 + 32_000, "header + 1s of PCM");

    let _ = tokio::fs::remove_file(&out_path).await;
    let _ = tokio::fs::remove_file(&sock_path).await;
    let _ = tokio::fs::remove_dir(&tmp).await;
}

#[tokio::test]
async fn dump_fails_clearly_when_socket_is_missing() {
    let tmp = unique_tempdir("wm-audio-dump-it-2");
    tokio::fs::create_dir_all(&tmp).await.expect("mkdir tmp");
    let bogus_sock = tmp.join("nope.sock");
    let out_path = tmp.join("out.wav");

    let bin = env!("CARGO_BIN_EXE_wm-audio-dump");
    let output = Command::new(bin)
        .arg(&out_path)
        .env("WM_MIC_SOCKET", &bogus_sock)
        .env("WM_DUMP_DURATION_MS", "500")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .expect("spawn wm-audio-dump");

    assert!(!output.status.success(), "exit on missing socket");
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("connect mic socket"),
        "error mentions connect step: {err}",
    );

    let _ = tokio::fs::remove_file(&out_path).await;
    let _ = tokio::fs::remove_dir(&tmp).await;
}

fn unique_tempdir(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("{label}-{pid}-{nanos}"));
    assert!(!Path::new(&dir).exists(), "tempdir collision: {}", dir.display());
    dir
}
