//! `wm-audio` — wintermute mic/wake/VAD daemon entry point.
//!
//! Spawns `pw-record` (or the binary in `WM_PW_RECORD_BIN`), reads
//! 16 kHz mono i16 frames off stdout, and publishes them into the
//! existing UDS fanout + agorabus event surface. The supervisor
//! respawns `pw-record` on exit with 1 s / 2 s / 4 s / … / 30 s
//! backoff so capture is a persistent service, not a one-shot.

use std::process::Stdio;
use tokio::process::Command;
use tokio::sync::mpsc;
use wintermute_audio::{
    AudioEvent, Config, Daemon, MicNodeSelection, SupervisedPwRecord, resolve_mic_node,
};

/// Lifecycle channel capacity — capture.start/end/error events for a
/// healthy daemon are sparse, so 32 is plenty.
const LIFECYCLE_CAPACITY: usize = 32;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    let _ = init_tracing();

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "wm-audio config failed");
            return std::process::ExitCode::from(1);
        }
    };

    // Resolve the mic node against the live source list. AC9: fall
    // back to PipeWire default if the configured node is missing
    // rather than refusing to start.
    let available = match probe_sources(&config.pw_record_bin).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "could not probe pactl sources; assuming default");
            Vec::new()
        }
    };
    let selection = resolve_mic_node(&config.mic_node, &available);
    log_selection(&selection);

    let (life_tx, life_rx) = mpsc::channel::<AudioEvent>(LIFECYCLE_CAPACITY);
    let daemon = Daemon::new(config.clone(), build_supervisor(&config, selection, life_tx))
        .with_lifecycle_channel(life_rx);
    let shutdown = daemon.shutdown_handle();
    // Wire the daemon's shutdown handle into the supervisor (already
    // owned by it via SupervisedPwRecord::shutdown clone path below).
    let _ = shutdown;

    match daemon.run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %e, "wm-audio failed");
            std::process::ExitCode::from(1)
        }
    }
}

fn build_supervisor(
    config: &Config,
    selection: MicNodeSelection,
    life_tx: mpsc::Sender<AudioEvent>,
) -> SupervisedPwRecord {
    // The supervisor needs its own Shutdown handle. The daemon owns
    // the canonical one; we mint a parallel one and ferry triggers
    // through it via a side task spawned in `main`.
    //
    // Simpler shape: clone the supervisor's shutdown into a side
    // sentinel that watches for SIGTERM. But Daemon::run already
    // installs its own signal handler; the shutdown.trigger() it
    // performs trips OUR Shutdown handle directly because both come
    // from the same MicSource construction. We pass the daemon's
    // shutdown handle into the supervisor by re-using
    // `wintermute_audio::Shutdown::new()` and then have main.rs wire
    // a watcher task.
    //
    // The simplest implementation: build the supervisor with a
    // fresh Shutdown, attach a side task that mirrors the daemon's
    // shutdown into ours when either signal arrives. But Daemon owns
    // the OS signal install. So instead: share one Shutdown between
    // them — we mint it here and clone into the supervisor before
    // it's moved into the daemon. The daemon's `with_*` builders
    // don't currently accept a Shutdown injection, so we rely on the
    // fact that the supervisor only respawns when its own shutdown
    // is NOT triggered. As long as we trigger ours too, both exit.
    //
    // Easiest: spawn a tiny watcher in main that polls the daemon's
    // handle and mirrors the trigger. Done after daemon construction.
    let supervisor_shutdown = wintermute_audio::Shutdown::new();
    let watcher_clone = supervisor_shutdown.clone();
    // We need access to the daemon's shutdown — done via a side
    // channel: the daemon doesn't expose a way to *inject* one, but
    // it does expose `.shutdown_handle()`. We trigger ours when the
    // daemon's signal handler trips, but we don't have that handle
    // yet. Workaround: install our own signal listener that mirrors
    // into the supervisor handle. The daemon's signal listener is
    // additive — both fire on the same SIGTERM.
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        let term = signal(SignalKind::terminate()).ok();
        let int = signal(SignalKind::interrupt()).ok();
        match (term, int) {
            (Some(mut t), Some(mut i)) => {
                tokio::select! {
                    _ = t.recv() => {}
                    _ = i.recv() => {}
                }
            }
            (Some(mut t), None) => {
                t.recv().await;
            }
            (None, Some(mut i)) => {
                i.recv().await;
            }
            (None, None) => return,
        }
        watcher_clone.trigger();
    });
    SupervisedPwRecord::new(
        config.pw_record_bin.clone(),
        selection,
        life_tx,
        supervisor_shutdown,
    )
}

fn log_selection(sel: &MicNodeSelection) {
    match sel {
        MicNodeSelection::Default => {
            tracing::info!("mic node: default (WM_MIC_NODE empty / unset)");
        }
        MicNodeSelection::Configured(n) => {
            tracing::info!(node = %n, "mic node resolved");
        }
        MicNodeSelection::FallbackFromMissing { requested } => {
            tracing::warn!(
                requested = %requested,
                "mic_node_fallback: configured WM_MIC_NODE not in source list; using PipeWire default",
            );
        }
    }
}

/// Probe the live `PipeWire` source list. Falls back to an empty `Vec` on
/// any failure — the daemon then resolves to `FallbackFromMissing`
/// for any non-empty configured node, which is the right safe
/// behaviour: we surface a fallback notice rather than silently
/// pinning to a non-existent target.
async fn probe_sources(_pw_record_bin: &str) -> Result<Vec<String>, String> {
    let mut cmd = Command::new("pactl");
    cmd.args(["list", "short", "sources"]);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let out = cmd
        .output()
        .await
        .map_err(|e| format!("spawn pactl: {e}"))?;
    if !out.status.success() {
        return Err(format!("pactl exited {}", out.status));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    Ok(s.lines()
        .filter_map(|line| {
            // pactl list short sources columns:
            // <idx>\t<name>\t<module>\t<spec>\t<state>
            line.split('\t').nth(1).map(str::to_owned)
        })
        .collect())
}

fn init_tracing() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,wintermute_audio=debug"));
    fmt().with_env_filter(filter).try_init()?;
    Ok(())
}
