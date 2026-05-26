//! UDS PCM fanout — one canonical capture stream, N socket subscribers.
//!
//! Implements PRD §2.3 step 3 (socket fanout) and acceptance criterion
//! AC7 (two simultaneous `mic.sock` subscribers consuming identical PCM).
//!
//! The daemon's main loop publishes each captured [`PcmFrame`] on a
//! [`tokio::sync::broadcast`] channel. This module owns the [`UnixListener`]
//! at `$XDG_RUNTIME_DIR/wintermute/mic.sock`; every accepted connection
//! gets its own broadcast receiver and an async writer task that serialises
//! the frame's little-endian i16 samples onto the socket.
//!
//! Slow subscribers that fall further than [`BROADCAST_CAPACITY`] frames
//! behind get `RecvError::Lagged` and we drop them — better than letting
//! one stalled consumer back-pressure the capture loop.

use crate::source::PcmFrame;
use crate::state::Shutdown;

use std::io::ErrorKind;
use std::path::PathBuf;

use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast::{self, Receiver, Sender, error::RecvError};
use tracing::{debug, info, warn};

/// Capacity of the PCM frame broadcast channel, in frames.
///
/// At a 20 ms quantum this is roughly 5 s of buffering — matches the
/// per-consumer ring sized by [`crate::daemon::RING_CAPACITY`].
pub const BROADCAST_CAPACITY: usize = 256;

/// Build the broadcast channel the main loop fans out into.
///
/// The returned [`Receiver`] is a seed that callers MAY drop; subscribers
/// are minted via [`Sender::subscribe`] when each UDS client connects.
#[must_use]
pub fn channel() -> (Sender<PcmFrame>, Receiver<PcmFrame>) {
    broadcast::channel(BROADCAST_CAPACITY)
}

/// Run the UDS fanout server until [`Shutdown`] is triggered.
///
/// Binds the listener at `socket_path` (creating the parent directory
/// and removing any stale socket file first), accepts connections, and
/// spawns a per-connection writer task.
///
/// # Errors
///
/// Returns the underlying [`std::io::Error`] if the socket cannot be
/// bound. Per-connection write errors are logged and the task exits;
/// the listener stays up.
pub async fn run(
    socket_path: PathBuf,
    bcast: Sender<PcmFrame>,
    shutdown: Shutdown,
) -> std::io::Result<()> {
    if let Some(parent) = socket_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                warn!(error = %e, path = %parent.display(), "could not create socket parent dir");
            }
        }
    }
    match tokio::fs::remove_file(&socket_path).await {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => warn!(
            error = %e,
            path = %socket_path.display(),
            "could not remove stale socket",
        ),
    }
    let listener = UnixListener::bind(&socket_path)?;
    info!(path = %socket_path.display(), "fanout listening");

    loop {
        if shutdown.is_triggered() {
            break;
        }
        let accept = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            listener.accept(),
        )
        .await;
        let stream = match accept {
            Ok(Ok((s, _addr))) => s,
            Ok(Err(e)) => {
                warn!(error = %e, "fanout accept error");
                continue;
            }
            Err(_) => continue,
        };
        let rx = bcast.subscribe();
        tokio::spawn(async move {
            match serve_client(stream, rx).await {
                Ok(()) => debug!("fanout client disconnected cleanly"),
                Err(e) => debug!(error = %e, "fanout client closed"),
            }
        });
    }

    // Best-effort cleanup of the socket file on shutdown.
    if let Err(e) = tokio::fs::remove_file(&socket_path).await {
        if e.kind() != ErrorKind::NotFound {
            debug!(error = %e, "fanout socket cleanup");
        }
    }
    Ok(())
}

async fn serve_client(
    mut stream: UnixStream,
    mut rx: Receiver<PcmFrame>,
) -> std::io::Result<()> {
    loop {
        let frame = match rx.recv().await {
            Ok(f) => f,
            Err(RecvError::Closed) => return Ok(()),
            Err(RecvError::Lagged(n)) => {
                warn!(dropped = n, "fanout subscriber lagged; dropping");
                return Ok(());
            }
        };
        // Serialise i16 little-endian (canonical PCM wire format).
        let mut bytes: Vec<u8> = Vec::with_capacity(frame.samples.len() * 2);
        for sample in &frame.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        stream.write_all(&bytes).await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn channel_subscribers_each_see_frame() {
        let (tx, mut rx1) = channel();
        let mut rx2 = tx.subscribe();
        let frame = PcmFrame {
            ts_ms: 1,
            samples: vec![1, 2, 3, 4],
        };
        let sent = tx.send(frame).is_ok();
        assert!(sent, "broadcast send with two live receivers");

        let got1 = rx1.recv().await.ok().map(|f| f.samples);
        let got2 = rx2.recv().await.ok().map(|f| f.samples);
        assert_eq!(got1, Some(vec![1, 2, 3, 4]));
        assert_eq!(got2, Some(vec![1, 2, 3, 4]));
    }

    #[tokio::test]
    async fn send_with_no_subscribers_is_an_error_but_not_fatal() {
        let (tx, rx) = channel();
        drop(rx);
        let frame = PcmFrame {
            ts_ms: 1,
            samples: vec![0; 8],
        };
        // Documented contract: ignore SendError when nobody is listening.
        assert!(tx.send(frame).is_err());
    }
}
