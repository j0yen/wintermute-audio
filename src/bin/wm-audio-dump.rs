//! `wm-audio-dump` — capture N seconds of PCM from the `wm-audio` fanout
//! socket and write a 16 kHz mono WAV file. Useful for debugging false
//! wake fires (PRD §7) without bringing up STT.
//!
//! Usage:
//!
//! ```text
//! wm-audio-dump <output.wav>
//! ```
//!
//! Environment:
//!
//! * `WM_MIC_SOCKET` — UDS path of the fanout socket
//!   (default `$XDG_RUNTIME_DIR/wintermute/mic.sock`).
//! * `WM_DUMP_DURATION_MS` — capture duration in milliseconds
//!   (default 10000).
//!
//! The output file gets a minimal 44-byte PCM WAV header (mono, 16 kHz,
//! 16-bit signed LE) followed by the raw samples read from the socket.
//! Exits early on EOF or after the deadline, whichever comes first.

#![allow(clippy::indexing_slicing, clippy::print_stderr)]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::net::UnixStream;

const SAMPLE_RATE_HZ: u32 = 16_000;
const NUM_CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;
const DEFAULT_DURATION_MS: u64 = 10_000;
const READ_CHUNK_BYTES: usize = 4096;
const WAV_HEADER_BYTES: usize = 44;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match run().await {
        Ok(bytes) => {
            eprintln!("wm-audio-dump: wrote {bytes} PCM bytes");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("wm-audio-dump: {e:#}");
            ExitCode::from(1)
        }
    }
}

async fn run() -> Result<u64> {
    let out_path = parse_output_path()?;
    let socket_path = mic_socket_path();
    let duration = Duration::from_millis(parse_duration_ms());

    let mut stream = UnixStream::connect(&socket_path)
        .await
        .with_context(|| format!("connect mic socket {}", socket_path.display()))?;

    let mut file = File::create(&out_path)
        .await
        .with_context(|| format!("create output file {}", out_path.display()))?;
    // Reserve a 44-byte slot for the WAV header; backfilled at the end.
    file.write_all(&[0_u8; WAV_HEADER_BYTES]).await?;

    let mut pcm_bytes: u64 = 0;
    let mut buf = vec![0_u8; READ_CHUNK_BYTES];
    let deadline = tokio::time::Instant::now() + duration;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let read = tokio::time::timeout(remaining, stream.read(&mut buf)).await;
        match read {
            Ok(Ok(0)) | Err(_) => break,
            Ok(Ok(n)) => {
                file.write_all(&buf[..n]).await?;
                pcm_bytes = pcm_bytes.saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
            }
            Ok(Err(e)) => return Err(e.into()),
        }
    }

    file.flush().await?;
    let header = wav_header(pcm_bytes);
    file.seek(std::io::SeekFrom::Start(0)).await?;
    file.write_all(&header).await?;
    file.flush().await?;
    Ok(pcm_bytes)
}

fn parse_output_path() -> Result<PathBuf> {
    let mut args = env::args_os();
    let _ = args.next();
    let out = args
        .next()
        .context("usage: wm-audio-dump <output.wav>")?;
    Ok(PathBuf::from(out))
}

fn mic_socket_path() -> PathBuf {
    if let Ok(p) = env::var("WM_MIC_SOCKET") {
        return PathBuf::from(p);
    }
    let runtime = env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| String::from("/tmp"));
    PathBuf::from(runtime).join("wintermute").join("mic.sock")
}

fn parse_duration_ms() -> u64 {
    env::var("WM_DUMP_DURATION_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DURATION_MS)
}

fn wav_header(data_bytes: u64) -> [u8; WAV_HEADER_BYTES] {
    let mut h = [0_u8; WAV_HEADER_BYTES];
    let data_size = u32::try_from(data_bytes).unwrap_or(u32::MAX);
    let chunk_size = data_size.saturating_add(36);
    let byte_rate = SAMPLE_RATE_HZ
        .saturating_mul(u32::from(NUM_CHANNELS))
        .saturating_mul(u32::from(BITS_PER_SAMPLE) / 8);
    let block_align = NUM_CHANNELS.saturating_mul(BITS_PER_SAMPLE / 8);
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&chunk_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16_u32.to_le_bytes());
    h[20..22].copy_from_slice(&1_u16.to_le_bytes());
    h[22..24].copy_from_slice(&NUM_CHANNELS.to_le_bytes());
    h[24..28].copy_from_slice(&SAMPLE_RATE_HZ.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());
    h
}
