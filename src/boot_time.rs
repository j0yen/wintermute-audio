//! `boot-time` subcommand — reads today's boot JSONL and prints a latency table.
//!
//! Each daemon logs a JSON line via `boot-stage.sh`. This command reads
//! `~/brain/journal/boot-YYYY-MM-DD.jsonl`, anchors on `agorabus.ready` as
//! time-zero, and prints elapsed seconds for every stage.

use std::path::{Path, PathBuf};

/// Result type for boot-time operations.
type Result<T> = std::result::Result<T, String>;

/// A single parsed stage entry from the JSONL log.
#[derive(Debug)]
struct StageEntry {
    /// ISO-8601 timestamp string.
    ts: String,
    /// Stage name (e.g. `wm-audio.ready`).
    stage: String,
}

/// Parse a single JSONL line into a `StageEntry`.
/// Returns `None` on malformed lines (best-effort, skip).
fn parse_line(line: &str) -> Option<StageEntry> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    // Minimal hand-rolled parse to avoid pulling extra deps.
    // Expected: {"ts":"...","stage":"...","pid":N}
    let ts = extract_json_string_field(line, "ts")?;
    let stage = extract_json_string_field(line, "stage")?;
    Some(StageEntry { ts, stage })
}

/// Extract a JSON string field value by key from a flat JSON object string.
/// Returns `None` if field is absent or not a string.
fn extract_json_string_field(json: &str, key: &str) -> Option<String> {
    // Look for `"<key>":"<value>"`
    let needle = format!("\"{key}\":\"");
    let start = json.find(&needle)?;
    let value_start = start + needle.len();
    let rest = json.get(value_start..)?;
    // Find closing quote, handling no escapes (our timestamps/stage names are safe)
    let end = rest.find('"')?;
    Some(rest.get(..end)?.to_owned())
}

/// Parse an ISO-8601 timestamp (with or without milliseconds) to seconds
/// since Unix epoch. Returns `None` on parse failure.
///
/// Handles: `YYYY-MM-DDTHH:MM:SS.mmmZ` and `YYYY-MM-DDTHH:MM:SSZ`.
fn parse_ts_to_secs_f64(ts: &str) -> Option<f64> {
    // Strip trailing Z
    let ts = ts.strip_suffix('Z').unwrap_or(ts);
    // Split on T
    let (date_part, time_part) = ts.split_once('T')?;
    let mut date_parts = date_part.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;

    // Split seconds and optional fractional
    let (time_main, frac_secs) = if let Some((main, frac)) = time_part.split_once('.') {
        let frac_str = format!("{frac:0<3}"); // pad to 3 digits
        let millis: f64 = frac_str.get(..3)?.parse::<u64>().ok()? as f64 / 1000.0;
        (main, millis)
    } else {
        (time_part, 0.0_f64)
    };

    let mut time_parts = time_main.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next()?.parse().ok()?;

    // Compute days since Unix epoch (1970-01-01) via Gregorian proleptic calendar
    let epoch_days = gregorian_to_epoch_days(year, month, day)?;
    let total_secs = epoch_days * 86400
        + hour * 3600
        + minute * 60
        + second;

    Some(total_secs as f64 + frac_secs)
}

/// Convert a Gregorian date to days since Unix epoch (1970-01-01).
fn gregorian_to_epoch_days(year: i64, month: i64, day: i64) -> Option<i64> {
    if month < 1 || month > 12 || day < 1 || day > 31 {
        return None;
    }
    // Algorithm: Julian Day Number then subtract JDN of 1970-01-01 (2440588)
    let a = (14 - month) / 12;
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    let jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    Some(jdn - 2_440_588)
}

/// Returns the default journal file path for today.
///
/// # Errors
///
/// Returns an error string if the home directory cannot be determined.
pub fn default_journal_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").map_err(|_| "HOME env var not set".to_owned())?;
    let today = today_date_string();
    Ok(PathBuf::from(home)
        .join("brain")
        .join("journal")
        .join(format!("boot-{today}.jsonl")))
}

/// Returns today's date as `YYYY-MM-DD` using the system `date` command.
fn today_date_string() -> String {
    std::process::Command::new("date")
        .arg("+%F")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "1970-01-01".to_owned())
}

/// Run the `boot-time` subcommand.
///
/// `args` is `&argv[1..]` (i.e. includes "boot-time" as [0]).
///
/// # Errors
///
/// Returns an error string if the journal file cannot be read or parsed.
#[allow(clippy::print_stdout, clippy::print_stderr)]
pub fn run_boot_time(args: &[String]) -> std::process::ExitCode {
    // Parse optional --journal-file flag
    let journal_path = match parse_journal_path(args) {
        Ok(Some(p)) => p,
        Ok(None) => {
            // --help was printed
            return std::process::ExitCode::SUCCESS;
        }
        Err(code) => return code,
    };

    match run_boot_time_inner(&journal_path) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wm-audio boot-time: {e}");
            std::process::ExitCode::from(1)
        }
    }
}

/// Parse the `--journal-file` flag from `boot-time` args.
///
/// Returns `Ok(Some(path))` with the resolved path, or
/// `Ok(None)` if `--help` was seen (caller should exit 0),
/// or `Err(code)` on bad flags.
#[allow(clippy::print_stdout, clippy::print_stderr)]
fn parse_journal_path(
    args: &[String],
) -> std::result::Result<Option<PathBuf>, std::process::ExitCode> {
    let mut journal_path: Option<PathBuf> = None;
    let mut i = 1usize; // skip "boot-time" at [0]
    while i < args.len() {
        match args.get(i).map_or("", String::as_str) {
            "--journal-file" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    journal_path = Some(PathBuf::from(v));
                } else {
                    eprintln!("wm-audio boot-time: --journal-file requires a value");
                    return Err(std::process::ExitCode::from(1));
                }
            }
            "-h" | "--help" => {
                println!(
                    "wm-audio boot-time — print per-stage boot latency table\n\
                     \n\
                     USAGE:\n\
                       wm-audio boot-time [--journal-file <path>]\n\
                     \n\
                     FLAGS:\n\
                       --journal-file <path>  Read from <path> instead of today's boot log\n\
                     \n\
                     EXIT CODES:\n\
                       0  success\n\
                       1  no boot log for today or parse error"
                );
                return Ok(None);
            }
            unknown => {
                eprintln!("wm-audio boot-time: unknown flag {unknown:?}");
                return Err(std::process::ExitCode::from(1));
            }
        }
        i += 1;
    }

    // Default: today's journal file
    let path = match journal_path {
        Some(p) => p,
        None => match default_journal_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("wm-audio boot-time: {e}");
                return Err(std::process::ExitCode::from(1));
            }
        },
    };

    Ok(Some(path))
}

/// Core logic: read the journal file and print the latency table.
#[allow(clippy::print_stdout)]
fn run_boot_time_inner(journal_path: &Path) -> Result<()> {
    let content = match std::fs::read_to_string(journal_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(
                "no boot log for today \u{2014} is wintermute.target running?".to_owned()
            );
        }
        Err(e) => return Err(format!("cannot read {}: {e}", journal_path.display())),
    };

    let mut entries: Vec<StageEntry> = content.lines().filter_map(parse_line).collect();

    if entries.is_empty() {
        return Err("boot log exists but contains no valid stage entries".to_owned());
    }

    // Sort by timestamp string (ISO-8601 sorts lexicographically)
    entries.sort_by(|a, b| a.ts.cmp(&b.ts));

    // Find epoch: first agorabus.ready entry
    let epoch_entry = entries
        .iter()
        .find(|e| e.stage == "agorabus.ready")
        .ok_or_else(|| "no agorabus.ready entry found in boot log".to_owned())?;

    let epoch_secs = parse_ts_to_secs_f64(&epoch_entry.ts)
        .ok_or_else(|| format!("cannot parse epoch timestamp: {}", epoch_entry.ts))?;

    // Compute elapsed for each entry
    let rows: Vec<(&str, f64)> = entries
        .iter()
        .filter_map(|e| {
            parse_ts_to_secs_f64(&e.ts).map(|secs| (e.stage.as_str(), secs - epoch_secs))
        })
        .collect();

    // Print table
    println!("{:<24} {}", "stage", "elapsed");
    for (stage, elapsed) in &rows {
        println!("{stage:<24} {elapsed:.1}s");
    }
    println!("{}", "\u{2500}".repeat(32));

    // voice-armed = time of last stage
    if let Some((_, last_elapsed)) = rows.last() {
        println!(
            "{:<24} {:.1}s  (target: \u{2264}15s)",
            "voice-armed", last_elapsed
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ts_basic() {
        // 2026-06-12T15:09:40Z
        let secs = parse_ts_to_secs_f64("2026-06-12T15:09:40Z");
        assert!(secs.is_some(), "failed to parse basic timestamp");
        let s = secs.unwrap_or(0.0);
        assert!(s > 0.0, "timestamp should be positive");
    }

    #[test]
    fn parse_ts_with_millis() {
        let secs_plain = parse_ts_to_secs_f64("2026-06-12T15:09:40Z").unwrap_or(0.0);
        let secs_millis = parse_ts_to_secs_f64("2026-06-12T15:09:40.500Z").unwrap_or(0.0);
        let diff = secs_millis - secs_plain;
        assert!(
            (diff - 0.5).abs() < 0.001,
            "milliseconds not parsed correctly: diff={diff}"
        );
    }

    #[test]
    fn extract_json_string_basic() {
        let json = r#"{"ts":"2026-06-12T15:09:40.154Z","stage":"wm-audio.ready","pid":649}"#;
        assert_eq!(
            extract_json_string_field(json, "ts").as_deref(),
            Some("2026-06-12T15:09:40.154Z")
        );
        assert_eq!(
            extract_json_string_field(json, "stage").as_deref(),
            Some("wm-audio.ready")
        );
    }

    #[test]
    fn parse_line_valid() {
        let line = r#"{"ts":"2026-06-12T15:09:40.154Z","stage":"wm-audio.ready","pid":649}"#;
        let entry = parse_line(line);
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.stage, "wm-audio.ready");
        assert_eq!(e.ts, "2026-06-12T15:09:40.154Z");
    }

    #[test]
    fn parse_line_empty() {
        assert!(parse_line("").is_none());
        assert!(parse_line("   ").is_none());
    }

    #[test]
    fn boot_time_inner_no_file() {
        let result = run_boot_time_inner(std::path::Path::new("/nonexistent/boot-9999-99-99.jsonl"));
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("no boot log"),
            "expected 'no boot log' message, got: {msg}"
        );
    }

    #[test]
    fn boot_time_inner_synthetic() {
        use std::io::Write as _;
        let tmp = tempfile_path();
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            writeln!(f, r#"{{"ts":"2026-06-12T10:00:00.000Z","stage":"agorabus.ready","pid":1}}"#).unwrap();
            writeln!(f, r#"{{"ts":"2026-06-12T10:00:00.200Z","stage":"wm-tts.ready","pid":2}}"#).unwrap();
            writeln!(f, r#"{{"ts":"2026-06-12T10:00:00.600Z","stage":"wm-audio.ready","pid":3}}"#).unwrap();
            writeln!(f, r#"{{"ts":"2026-06-12T10:00:01.100Z","stage":"wm-brain.ready","pid":4}}"#).unwrap();
        }
        let result = run_boot_time_inner(&tmp);
        // Cleanup
        let _ = std::fs::remove_file(&tmp);
        assert!(result.is_ok(), "boot_time_inner failed: {:?}", result);
    }

    /// Generate a temp file path (not created yet).
    fn tempfile_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "boot-telemetry-test-{}.jsonl",
            std::process::id()
        ))
    }
}
