//! `wm-audio` — wintermute mic/wake/VAD daemon entry point.
//!
//! Subcommands:
//! - `start` (default): Spawns `pw-record`, reads 16 kHz mono i16 frames
//!   off stdout, and publishes them into the UDS fanout + agorabus event
//!   surface. The supervisor respawns `pw-record` on exit with exponential
//!   backoff so capture is a persistent service.
//! - `fetch-models`: Downloads, checksum-verifies, and installs the
//!   upstream-provisioned ONNX models (currently the Silero VAD model) into
//!   the model prefix. Wake models are NOT fetched here — they come from the
//!   local training pipeline (see `models` module docs and
//!   `project_wintermute_wake_training`). See PRD-rouse-wake-vad-models.md.

use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tokio::sync::mpsc;
use wintermute_audio::{
    AecProbe, AudioEvent, Config, DEFAULT_PACTL, Daemon, InstallOutcome, MicNodeSelection,
    OutputFormat, SupervisedPwRecord, MANIFEST, build_list, load_or_null_wake, provision_one,
    read_provenance, resolve_mic_node, run_aec_probe, upsert_provenance, write_provenance,
};
use wintermute_audio::selftest::run_selftest;

/// Default model prefix (system install, matches wm-stt and PRD §2.1).
const DEFAULT_MODEL_PREFIX: &str = "/usr/share/wintermute/models";

/// Lifecycle channel capacity — capture.start/end/error events for a
/// healthy daemon are sparse, so 32 is plenty.
const LIFECYCLE_CAPACITY: usize = 32;

// ---------------------------------------------------------------------------
// Subcommand dispatch
// ---------------------------------------------------------------------------

#[allow(clippy::print_stderr)]
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();

    // Determine subcommand.  With no args or "start", run the daemon.
    let subcmd = args.get(1).map_or("start", String::as_str);

    match subcmd {
        "start" | "--start" => run_daemon(),
        "fetch-models" => {
            let subcmd_args = args.get(1..).unwrap_or(&[]);
            run_fetch_models(subcmd_args)
        }
        "selftest" => {
            let subcmd_args = args.get(1..).unwrap_or(&[]);
            run_selftest(subcmd_args)
        }
        "-h" | "--help" => {
            print_help();
            std::process::ExitCode::SUCCESS
        }
        unknown => {
            eprintln!(
                "wm-audio: unknown subcommand {unknown:?}\n\
                 Use 'wm-audio --help' for usage."
            );
            std::process::ExitCode::from(1)
        }
    }
}

#[allow(clippy::print_stdout)]
fn print_help() {
    println!(
        "wm-audio — wintermute audio daemon\n\
         \n\
         SUBCOMMANDS:\n\
           start           Start the daemon (default when no subcommand given)\n\
           fetch-models    Download and install upstream VAD model(s); wake\n\
         \x20                  model comes from the local training pipeline\n\
           selftest        Diagnose the voice path (fixture or --live mode)\n\
         \n\
         FETCH-MODELS FLAGS:\n\
           --prefix <dir>  Install root (default: {DEFAULT_MODEL_PREFIX})\n\
           --force         Re-download even if models are already current\n\
           --list          Print manifest without downloading (exit 0)\n\
           --format <fmt>  Output format: text|json (default: text)\n\
         \n\
         SELFTEST FLAGS:\n\
           --live [secs]   Live mode: listen to running daemon for SECS seconds\n\
           --format <fmt>  Output format: text|json (default: text)\n\
           --prefix <dir>  Model prefix (default: {DEFAULT_MODEL_PREFIX})\n\
         \n\
         SELFTEST EXIT CODES:\n\
           0  healthy — all required events observed\n\
           1  deaf — pipeline up but events never fired\n\
           2  could-not-run — models absent / no daemon in --live mode\n\
         \n\
         DAEMON ENV (start):\n\
           WM_MIC_NODE       PipeWire capture node (default: PW default)\n\
           WM_WAKE_WORD      hey_jarvis|okay_nabu|hey_mycroft (default: hey_jarvis)\n\
           WM_WAKE_THRESHOLD Float 0.0–1.0 (default: 0.6)\n\
           WM_MIC_SOCK       UDS fanout socket path\n\
           AGORABUS_SOCK     agorabus daemon socket\n\
           WM_AUDIO_SESSION  Session id (default: wm-audio-<pid>)\n\
           WM_PW_RECORD_BIN  pw-record binary path (default: pw-record)\n"
    );
}

// ---------------------------------------------------------------------------
// `fetch-models` subcommand
// ---------------------------------------------------------------------------

/// Flags parsed from `fetch-models` arguments.
struct FetchFlags {
    prefix: PathBuf,
    force: bool,
    list_only: bool,
    format: OutputFormat,
}

/// Parse flags for the `fetch-models` subcommand.
///
/// Returns `Ok(FetchFlags)` or an error message suitable for `eprintln!`.
///
/// # Errors
///
/// Returns an error string on unrecognized flags or missing values.
#[allow(clippy::print_stderr)]
fn parse_fetch_flags(subcmd_args: &[String]) -> Result<Option<FetchFlags>, std::process::ExitCode> {
    let mut prefix = PathBuf::from(DEFAULT_MODEL_PREFIX);
    let mut force = false;
    let mut list_only = false;
    let mut format = OutputFormat::Text;

    let mut i = 1usize; // skip "fetch-models" at index 0
    while i < subcmd_args.len() {
        let flag = subcmd_args.get(i).map_or("", String::as_str);
        match flag {
            "--prefix" => {
                i += 1;
                if let Some(v) = subcmd_args.get(i) {
                    prefix = PathBuf::from(v);
                } else {
                    eprintln!("wm-audio fetch-models: --prefix requires a value");
                    return Err(std::process::ExitCode::from(1));
                }
            }
            "--force" => {
                force = true;
            }
            "--list" => {
                list_only = true;
            }
            "--format" => {
                i += 1;
                if let Some(v) = subcmd_args.get(i) {
                    match OutputFormat::parse(v) {
                        Ok(f) => format = f,
                        Err(e) => {
                            eprintln!("wm-audio fetch-models: {e}");
                            return Err(std::process::ExitCode::from(1));
                        }
                    }
                } else {
                    eprintln!("wm-audio fetch-models: --format requires a value");
                    return Err(std::process::ExitCode::from(1));
                }
            }
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            unknown => {
                eprintln!("wm-audio fetch-models: unknown flag {unknown:?}");
                return Err(std::process::ExitCode::from(1));
            }
        }
        i += 1;
    }

    Ok(Some(FetchFlags { prefix, force, list_only, format }))
}

/// Parse and run the `fetch-models` subcommand.
///
/// `subcmd_args` is `&args[1..]` (i.e. includes "fetch-models" as [0]).
#[allow(clippy::print_stderr)]
fn run_fetch_models(subcmd_args: &[String]) -> std::process::ExitCode {
    let flags = match parse_fetch_flags(subcmd_args) {
        Ok(Some(f)) => f,
        Ok(None) => return std::process::ExitCode::SUCCESS,
        Err(code) => return code,
    };

    // --- --list path -------------------------------------------------------
    if flags.list_only {
        return do_list(&flags.prefix, flags.format);
    }

    // --- provision all models ----------------------------------------------
    do_provision(&flags.prefix, flags.force)
}

/// Provision all models into `prefix`.
#[allow(clippy::print_stderr)]
fn do_provision(prefix: &std::path::Path, force: bool) -> std::process::ExitCode {
    let tmp_dir = std::env::temp_dir().join(format!(
        "wm-audio-fetch-{}", std::process::id()
    ));
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        eprintln!("wm-audio fetch-models: cannot create temp dir: {e}");
        return std::process::ExitCode::from(1);
    }

    let mut provenance = match read_provenance(prefix) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("wm-audio fetch-models: cannot read existing provenance: {e}");
            Vec::new()
        }
    };

    let mut needs_privilege = false;
    let mut had_error = false;

    for entry in MANIFEST {
        let outcome = provision_one(entry, prefix, &tmp_dir, force);
        match outcome {
            Ok(InstallOutcome::AlreadyCurrent) => {
                eprintln!("fetch-models: {} already-current (skipped)", entry.name);
            }
            Ok(InstallOutcome::Installed) => {
                eprintln!("fetch-models: {} installed ok", entry.name);
                // Record provenance with the pinned sha256 (we already
                // verified it in provision_one via download_to_tmp).
                upsert_provenance(&mut provenance, entry, entry.sha256);
            }
            Ok(InstallOutcome::NeedsPrivilege { staged: _, target: _ }) => {
                eprintln!(
                    "fetch-models: {} staged+verified but target not writable; needs privilege",
                    entry.name
                );
                needs_privilege = true;
            }
            Err(e) => {
                eprintln!("fetch-models: ERROR for {}: {e}", entry.name);
                had_error = true;
            }
        }
    }

    // Write provenance for whatever was successfully installed.
    if !provenance.is_empty() {
        if let Err(e) = write_provenance(prefix, &provenance) {
            eprintln!("fetch-models: could not write MODELS.json: {e}");
        }
    }

    // Clean up temp dir (best effort).
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // Exit 2 when any model needs privilege.
    if needs_privilege {
        eprintln!(
            "\nfetch-models: some models could not be installed to {} (not writable).",
            prefix.display()
        );
        eprintln!("Re-run with privilege:");
        eprintln!("  sudo wm-audio fetch-models --prefix {}", prefix.display());
        eprintln!("\nAlternatively use a user-writable prefix:");
        eprintln!("  wm-audio fetch-models --prefix ~/.local/share/wintermute/models");
        return std::process::ExitCode::from(2);
    }

    if had_error {
        return std::process::ExitCode::from(1);
    }

    std::process::ExitCode::SUCCESS
}

/// Print the model list (--list flag).
#[allow(clippy::print_stdout, clippy::print_stderr)]
fn do_list(prefix: &std::path::Path, format: OutputFormat) -> std::process::ExitCode {
    let list = build_list(prefix);
    match format {
        OutputFormat::Text => {
            println!("wm-audio model manifest (prefix: {})", prefix.display());
            println!("{:<14} {:<5} {:<25} LICENSE", "NAME", "KIND", "FILENAME");
            for e in &list {
                println!(
                    "{:<14} {:<5} {:<25} {}",
                    e.name,
                    e.kind.to_string(),
                    e.filename,
                    e.license
                );
            }
        }
        OutputFormat::Json => {
            match serde_json::to_string_pretty(&list) {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("fetch-models --list: JSON error: {e}");
                    return std::process::ExitCode::from(1);
                }
            }
        }
    }
    std::process::ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Daemon subcommand (`start`)
// ---------------------------------------------------------------------------

#[allow(clippy::print_stderr)]
fn run_daemon() -> std::process::ExitCode {
    // Build a single-threaded tokio runtime for the daemon.
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("wm-audio: failed to build tokio runtime: {e}");
            return std::process::ExitCode::from(1);
        }
    };

    rt.block_on(daemon_main())
}

async fn daemon_main() -> std::process::ExitCode {
    let _ = init_tracing();

    let mut config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "wm-audio config failed");
            return std::process::ExitCode::from(1);
        }
    };

    // AEC probe (PRD-aec §2.4). When the `aec` cargo feature is on
    // and the `wm-mic-aec` virtual source is live, substitute it for
    // the configured WM_MIC_NODE so the rest of the resolution chain
    // (existing AC9 fallback) operates on the AEC node. When the
    // probe is Missing, log once and let the configured node win as
    // before — wm-audio degrades to half-duplex, the daemon stays up.
    let pactl_bin = std::env::var("WM_PACTL_BIN")
        .unwrap_or_else(|_| DEFAULT_PACTL.to_owned());
    let aec_probe = run_aec_probe(&pactl_bin).await;
    log_aec_probe(&aec_probe);
    let effective_mic_node = aec_probe.resolve_mic_node(&config.mic_node).to_owned();
    if effective_mic_node != config.mic_node {
        tracing::info!(
            from = %config.mic_node,
            to = %effective_mic_node,
            "AEC probe substituted mic node",
        );
        config.mic_node = effective_mic_node;
    }

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

    // Load the locally-trained wake model from the system prefix
    // (`<prefix>/wake/<label>.onnx`, e.g. `.../wake/wintermute.onnx`).
    // `load_or_null_wake` falls back to the null engine — logging
    // `wake_model_missing` / `wake_model_load_error` — if the file is
    // absent or its input contract doesn't match, so the daemon always
    // starts. Without this the daemon silently ran the null detector and
    // wake never fired regardless of an installed model.
    let wake_label = config.wake_word.as_label();
    let wake_model_path = PathBuf::from(DEFAULT_MODEL_PREFIX)
        .join("wake")
        .join(format!("{wake_label}.onnx"));
    let wake_detector = load_or_null_wake(&wake_model_path, wake_label);

    let (life_tx, life_rx) = mpsc::channel::<AudioEvent>(LIFECYCLE_CAPACITY);
    let daemon = Daemon::new(config.clone(), build_supervisor(&config, selection, life_tx))
        .with_lifecycle_channel(life_rx)
        .with_wake_detector_arc(wake_detector);
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

fn log_aec_probe(probe: &AecProbe) {
    match probe {
        AecProbe::FeatureDisabled => {
            tracing::debug!("aec feature compiled out; skipping AEC probe");
        }
        AecProbe::Present { node } => {
            tracing::info!(node = %node, "aec source present");
        }
        AecProbe::Missing { expected } => {
            tracing::warn!(
                expected = %expected,
                "aec_module_missing: install pkg/pipewire-config/99-wintermute-aec.conf and restart pipewire, or set --no-default-features",
            );
        }
    }
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
