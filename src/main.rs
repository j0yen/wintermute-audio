//! `wm-audio` — wintermute mic/wake/VAD daemon entry point.
//!
//! Iter-2 wires the skeleton against a [`NullSource`] so the binary
//! compiles and runs without a real capture device. Real `PipeWire`
//! capture lands in iter-3.

use wintermute_audio::{NullSource, run};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    // Best-effort tracing init. Failure here is non-fatal — the
    // daemon still runs, it just won't log structured events.
    let _ = init_tracing();

    // The null source emits zero frames and exits, so without an
    // agorabus daemon on the box this binary will fail at connect.
    // That's deliberate: iter-2 is "the daemon compiles, links, and
    // negotiates with the bus when it's there". `cargo check` covers
    // the build; integration validation is queued for iter-3.
    let source = NullSource::default();
    match run(source).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %e, "wm-audio failed");
            std::process::ExitCode::from(1)
        }
    }
}

fn init_tracing() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,wintermute_audio=debug"));
    fmt().with_env_filter(filter).try_init()?;
    Ok(())
}
