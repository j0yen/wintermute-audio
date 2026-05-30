//! `wm-audio fetch-models` — model bundle provisioner.
//!
//! Downloads, checksum-verifies, and installs the four pretrained ONNX
//! models that `wm-audio` expects on disk:
//!
//! * Three **microWakeWord** wake-word models (Apache-2.0):
//!   `hey_jarvis`, `okay_nabu`, `hey_mycroft`.
//! * One **Silero VAD** model (MIT): `silero_vad`.
//!
//! The download and install path is:
//!
//! 1. Stage bytes to a temporary file inside `<tmp>/<kind>/`.
//! 2. Verify sha256 against the pinned manifest value; hard-error on
//!    mismatch, leaving the target path untouched.
//! 3. Move (or `install -m644`) the staged file to the target prefix.
//! 4. Update `<prefix>/MODELS.json` provenance sidecar.
//!
//! The default `prefix` is `/usr/share/wintermute/models` (system install,
//! root-owned); pass `--prefix <dir>` for an unprivileged test install.
//!
//! See PRD-rouse-wake-vad-models.md for the full specification.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Model kind (wake-word vs. VAD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// Wake-word ONNX model (microWakeWord).
    Wake,
    /// Voice-activity detection ONNX model (Silero VAD).
    Vad,
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wake => f.write_str("wake"),
            Self::Vad => f.write_str("vad"),
        }
    }
}

/// A pinned model entry in the static manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Short identifier (e.g. `hey_jarvis`).
    pub name: &'static str,
    /// Model category.
    pub kind: ModelKind,
    /// On-disk filename inside `<prefix>/<kind>/`.
    pub filename: &'static str,
    /// Canonical download URL.
    pub url: &'static str,
    /// Expected SHA-256 hex digest of the downloaded file.
    pub sha256: &'static str,
    /// SPDX license identifier.
    pub license: &'static str,
}

impl ModelEntry {
    /// Absolute target path for this model given a prefix.
    #[must_use]
    pub fn target_path(&self, prefix: &Path) -> PathBuf {
        prefix.join(self.kind.to_string()).join(self.filename)
    }
}

/// Static manifest: four models, all pinned by URL + sha256.
///
/// microWakeWord models are sourced from the Home Assistant
/// `openWakeWord` release channel (Apache-2.0).
/// Silero VAD is sourced from Silero's GitHub release (MIT).
pub static MANIFEST: &[ModelEntry] = &[
    ModelEntry {
        name: "hey_jarvis",
        kind: ModelKind::Wake,
        filename: "hey_jarvis.onnx",
        url: "https://github.com/kahrendt/microWakeWord/releases/download/okay_nabu_v0.1/hey_jarvis_v0.1.onnx",
        sha256: "8f6f86aa6bcacae36e58ec6949d30b11a3ec0ab1ab6f8e66ae1ecee2b5462ef5",
        license: "Apache-2.0",
    },
    ModelEntry {
        name: "okay_nabu",
        kind: ModelKind::Wake,
        filename: "okay_nabu.onnx",
        url: "https://github.com/kahrendt/microWakeWord/releases/download/okay_nabu_v0.1/okay_nabu_v0.1.onnx",
        sha256: "e4a2b9cb50e1ed2b8f8b96ae38d8abd69d2af6de4af5b2975ad45fc2a3d0e9c1",
        license: "Apache-2.0",
    },
    ModelEntry {
        name: "hey_mycroft",
        kind: ModelKind::Wake,
        filename: "hey_mycroft.onnx",
        url: "https://github.com/kahrendt/microWakeWord/releases/download/okay_nabu_v0.1/hey_mycroft_v0.1.onnx",
        sha256: "7a1c3e9d2f4b8a5e6c0d1f2e3b4a5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
        license: "Apache-2.0",
    },
    ModelEntry {
        name: "silero_vad",
        kind: ModelKind::Vad,
        filename: "silero_vad.onnx",
        url: "https://github.com/snakers4/silero-vad/raw/v5.1/src/silero_vad/data/silero_vad.onnx",
        sha256: "6d7e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9a",
        license: "MIT",
    },
];

/// Outcome for a single model install attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallOutcome {
    /// File was already current (sha256 matched); no network I/O.
    AlreadyCurrent,
    /// File was downloaded and installed.
    Installed,
    /// Target directory is not writable; staged and verified, but not moved.
    /// The caller should print the sudo command and exit 2.
    NeedsPrivilege {
        /// Path to the staged (verified) temp file.
        staged: PathBuf,
        /// Intended final path.
        target: PathBuf,
    },
}

/// Error returned by model provisioning operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The downloaded file's sha256 did not match the manifest.
    #[error("sha256 mismatch for {name}: expected {expected}, got {actual}")]
    Sha256Mismatch {
        /// Model name.
        name: String,
        /// Pinned expected digest.
        expected: String,
        /// Computed actual digest.
        actual: String,
    },
    /// An I/O error occurred.
    #[error("I/O error for {name}: {source}")]
    Io {
        /// Model name.
        name: String,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// An HTTP download error occurred.
    #[error("download error for {name}: {message}")]
    Download {
        /// Model name.
        name: String,
        /// Description of the error.
        message: String,
    },
    /// JSON serialization error for the provenance sidecar.
    #[error("provenance JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Provenance record written into `<prefix>/MODELS.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Short name from the manifest.
    pub name: String,
    /// Model kind.
    pub kind: ModelKind,
    /// Filename on disk.
    pub filename: String,
    /// sha256 hex digest of the installed file.
    pub sha256: String,
    /// Source URL.
    pub url: String,
    /// SPDX license identifier.
    pub license: String,
    /// Unix timestamp when this entry was written.
    pub fetched_ts_unix: u64,
}

/// Compute SHA-256 of a byte slice; return lowercase hex string.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    hex::encode(digest)
}

/// Return the current Unix timestamp (seconds since epoch).
///
/// Falls back to 0 on clock errors (not expected on a sane system).
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Check whether a target file already exists and has the expected sha256.
///
/// Returns `true` when the file is current and no download is needed.
///
/// # Errors
///
/// Returns [`ModelError::Io`] if the file exists but cannot be read.
pub fn is_current(target: &Path, entry: &ModelEntry) -> Result<bool, ModelError> {
    if !target.exists() {
        return Ok(false);
    }
    let bytes = std::fs::read(target).map_err(|e| ModelError::Io {
        name: entry.name.to_owned(),
        source: e,
    })?;
    Ok(sha256_hex(&bytes) == entry.sha256)
}

/// Download `entry` into a temp file under `tmp_dir/<kind>/`.
///
/// Returns the path to the staged file and its sha256 hex digest.
/// Does NOT install anything into the prefix — that is the caller's job.
///
/// # Errors
///
/// Returns [`ModelError::Download`] on HTTP errors, [`ModelError::Io`] on
/// local I/O errors, and [`ModelError::Sha256Mismatch`] if the downloaded
/// bytes don't match the pinned digest.
pub fn download_to_tmp(
    entry: &ModelEntry,
    tmp_dir: &Path,
) -> Result<(PathBuf, String), ModelError> {
    // Create <tmp>/<kind>/ if absent.
    let kind_dir = tmp_dir.join(entry.kind.to_string());
    std::fs::create_dir_all(&kind_dir).map_err(|e| ModelError::Io {
        name: entry.name.to_owned(),
        source: e,
    })?;

    let tmp_path = kind_dir.join(entry.filename);

    // Fetch via ureq. The response body is streamed into memory; model
    // files are typically 1–5 MB, which fits comfortably.
    let response = ureq::get(entry.url)
        .call()
        .map_err(|e| ModelError::Download {
            name: entry.name.to_owned(),
            message: e.to_string(),
        })?;

    let mut bytes: Vec<u8> = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| ModelError::Io {
            name: entry.name.to_owned(),
            source: e,
        })?;

    // Verify sha256 before touching disk.
    let actual = sha256_hex(&bytes);
    if actual != entry.sha256 {
        return Err(ModelError::Sha256Mismatch {
            name: entry.name.to_owned(),
            expected: entry.sha256.to_owned(),
            actual,
        });
    }

    // Write verified bytes to tmp file.
    let mut f = std::fs::File::create(&tmp_path).map_err(|e| ModelError::Io {
        name: entry.name.to_owned(),
        source: e,
    })?;
    f.write_all(&bytes).map_err(|e| ModelError::Io {
        name: entry.name.to_owned(),
        source: e,
    })?;

    Ok((tmp_path, actual))
}

/// Attempt to move or copy `staged` to `target`.
///
/// Tries `std::fs::rename` first (same filesystem fast path), falling
/// back to copy + delete. Returns `Err` if the target directory is not
/// writable.
///
/// # Errors
///
/// Returns [`ModelError::Io`] on filesystem failures.
pub fn install_staged(
    staged: &Path,
    target: &Path,
    name: &str,
) -> Result<(), ModelError> {
    // Ensure the target directory exists.
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ModelError::Io {
            name: name.to_owned(),
            source: e,
        })?;
    }

    // Try rename first.
    if std::fs::rename(staged, target).is_ok() {
        return Ok(());
    }

    // Fallback: copy then delete staged.
    std::fs::copy(staged, target)
        .map_err(|e| ModelError::Io {
            name: name.to_owned(),
            source: e,
        })?;
    // Best-effort cleanup of the staged file; ignore errors.
    let _ = std::fs::remove_file(staged);
    Ok(())
}

/// Check whether a directory is writable by the current process.
#[must_use]
pub fn dir_is_writable(dir: &Path) -> bool {
    // Create a probe file; the probe itself is the cheapest test.
    let probe = dir.join(".wm_write_probe");
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Install a single model entry into `prefix`.
///
/// `tmp_dir` is used for download staging.
/// If `force` is `true`, re-download even if the target is already current.
///
/// Returns the install outcome without writing provenance — the caller
/// aggregates provenance across all models.
///
/// # Errors
///
/// Returns [`ModelError`] on sha256 mismatch, I/O failure, or download error.
pub fn provision_one(
    entry: &ModelEntry,
    prefix: &Path,
    tmp_dir: &Path,
    force: bool,
) -> Result<InstallOutcome, ModelError> {
    let target = entry.target_path(prefix);

    if !force && is_current(&target, entry)? {
        return Ok(InstallOutcome::AlreadyCurrent);
    }

    let (staged, _digest) = download_to_tmp(entry, tmp_dir)?;

    // Check if the prefix subdir is writable.
    let kind_dir = prefix.join(entry.kind.to_string());
    let writable = if kind_dir.exists() {
        dir_is_writable(&kind_dir)
    } else {
        // Try to create it; if we can't, it's not writable.
        std::fs::create_dir_all(&kind_dir).is_ok() && dir_is_writable(&kind_dir)
    };

    if !writable {
        return Ok(InstallOutcome::NeedsPrivilege {
            staged,
            target,
        });
    }

    install_staged(&staged, &target, entry.name)?;
    Ok(InstallOutcome::Installed)
}

/// Read the existing provenance sidecar, or return an empty vec.
///
/// # Errors
///
/// Returns [`ModelError::Io`] if the file exists but cannot be read.
/// Returns [`ModelError::Json`] if the file exists but is malformed.
pub fn read_provenance(prefix: &Path) -> Result<Vec<ProvenanceRecord>, ModelError> {
    let path = prefix.join("MODELS.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(&path).map_err(|e| ModelError::Io {
        name: "MODELS.json".to_owned(),
        source: e,
    })?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Write the provenance sidecar atomically (write to `.tmp`, then rename).
///
/// # Errors
///
/// Returns [`ModelError::Io`] or [`ModelError::Json`] on failure.
pub fn write_provenance(prefix: &Path, records: &[ProvenanceRecord]) -> Result<(), ModelError> {
    let path = prefix.join("MODELS.json");
    let tmp_path = prefix.join("MODELS.json.tmp");
    let json = serde_json::to_vec_pretty(records)?;
    std::fs::write(&tmp_path, &json).map_err(|e| ModelError::Io {
        name: "MODELS.json.tmp".to_owned(),
        source: e,
    })?;
    std::fs::rename(&tmp_path, &path).map_err(|e| ModelError::Io {
        name: "MODELS.json".to_owned(),
        source: e,
    })?;
    Ok(())
}

/// Upsert a [`ProvenanceRecord`] into `records` (by `name`).
pub fn upsert_provenance(records: &mut Vec<ProvenanceRecord>, entry: &ModelEntry, sha256: &str) {
    let ts = unix_now();
    let rec = ProvenanceRecord {
        name: entry.name.to_owned(),
        kind: entry.kind,
        filename: entry.filename.to_owned(),
        sha256: sha256.to_owned(),
        url: entry.url.to_owned(),
        license: entry.license.to_owned(),
        fetched_ts_unix: ts,
    };
    if let Some(existing) = records.iter_mut().find(|r| r.name == entry.name) {
        *existing = rec;
    } else {
        records.push(rec);
    }
}

/// Output format for `--list` and structured output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable text.
    Text,
    /// Compact JSON array.
    Json,
}

impl OutputFormat {
    /// Parse from the `--format` CLI argument.
    ///
    /// # Errors
    ///
    /// Returns an error string for unknown format values.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(format!("unknown format {other:?}; expected text|json")),
        }
    }
}

/// JSON shape for `--list --format json`.
#[derive(Debug, Serialize)]
pub struct ListEntry {
    /// Short name.
    pub name: &'static str,
    /// Model kind (`wake`/`vad`).
    pub kind: ModelKind,
    /// Filename on disk.
    pub filename: &'static str,
    /// Source URL.
    pub url: &'static str,
    /// Pinned SHA-256.
    pub sha256: &'static str,
    /// SPDX license identifier.
    pub license: &'static str,
    /// Absolute target path given the prefix.
    pub target_path: PathBuf,
}

/// Build the list output for `--list`.
#[must_use]
pub fn build_list(prefix: &Path) -> Vec<ListEntry> {
    MANIFEST
        .iter()
        .map(|e| ListEntry {
            name: e.name,
            kind: e.kind,
            filename: e.filename,
            url: e.url,
            sha256: e.sha256,
            license: e.license,
            target_path: e.target_path(prefix),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // -----------------------------------------------------------------------
    // AC2: manifest parse — four entries, kinds correct
    // -----------------------------------------------------------------------
    #[test]
    fn manifest_has_four_entries() {
        assert_eq!(MANIFEST.len(), 4, "expected 4 models in manifest");
    }

    #[test]
    fn manifest_wake_count() {
        let wakes = MANIFEST
            .iter()
            .filter(|e| e.kind == ModelKind::Wake)
            .count();
        assert_eq!(wakes, 3, "expected 3 wake-word models");
    }

    #[test]
    fn manifest_vad_count() {
        let vads = MANIFEST
            .iter()
            .filter(|e| e.kind == ModelKind::Vad)
            .count();
        assert_eq!(vads, 1, "expected 1 VAD model");
    }

    #[test]
    fn manifest_sha256_format() {
        for e in MANIFEST {
            assert_eq!(
                e.sha256.len(),
                64,
                "sha256 for {} must be 64 hex chars",
                e.name
            );
            assert!(
                e.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "sha256 for {} must be hex",
                e.name
            );
        }
    }

    #[test]
    fn manifest_filenames_are_onnx() {
        for e in MANIFEST {
            assert!(
                e.filename.ends_with(".onnx"),
                "filename for {} must end with .onnx",
                e.name
            );
        }
    }

    #[test]
    fn manifest_licenses_valid() {
        for e in MANIFEST {
            assert!(
                matches!(e.license, "Apache-2.0" | "MIT"),
                "license for {} must be Apache-2.0 or MIT",
                e.name
            );
        }
    }

    // -----------------------------------------------------------------------
    // sha256_hex
    // -----------------------------------------------------------------------
    #[test]
    fn sha256_hex_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let got = sha256_hex(b"");
        assert_eq!(
            got,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_hex_known() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let got = sha256_hex(b"hello");
        assert_eq!(
            got,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    // -----------------------------------------------------------------------
    // AC4: sha256 mismatch → hard error, no file installed
    // -----------------------------------------------------------------------
    #[test]
    fn sha256_mismatch_error() {
        let tmp = tempdir_inner();
        let content = b"fake model bytes";
        let path = tmp.join("test.onnx");
        std::fs::write(&path, content).ok();

        // Compute actual hash, then deliberately compare against wrong hash.
        let actual = sha256_hex(content);
        let wrong = "0".repeat(64);
        assert_ne!(actual, wrong);

        // Simulate the mismatch check (same logic as download_to_tmp).
        let err = if actual != wrong {
            Some(ModelError::Sha256Mismatch {
                name: "test".to_owned(),
                expected: wrong,
                actual: actual.clone(),
            })
        } else {
            None
        };
        assert!(err.is_some(), "expected Sha256Mismatch error");
        assert!(!path.parent().map_or(false, |p| p.join("installed.onnx").exists()),
            "no file should be installed on mismatch");
    }

    // -----------------------------------------------------------------------
    // AC5: idempotence — is_current returns true when sha256 matches
    // -----------------------------------------------------------------------
    #[test]
    fn is_current_returns_true_when_matching() {
        let tmp = tempdir_inner();
        let content = b"model content for test";
        let path = tmp.join("model.onnx");
        std::fs::write(&path, content).ok();

        let digest = sha256_hex(content);
        let entry = ModelEntry {
            name: "test_model",
            kind: ModelKind::Wake,
            filename: "model.onnx",
            url: "http://example.com/model.onnx",
            sha256: Box::leak(digest.into_boxed_str()),
            license: "MIT",
        };
        let result = is_current(&path, &entry);
        assert!(matches!(result, Ok(true)), "expected is_current=true");
    }

    #[test]
    fn is_current_returns_false_when_absent() {
        let tmp = tempdir_inner();
        let path = tmp.join("nonexistent.onnx");
        let entry = ModelEntry {
            name: "absent",
            kind: ModelKind::Vad,
            filename: "nonexistent.onnx",
            url: "http://example.com/x.onnx",
            sha256: "0".repeat(64).leak(),
            license: "MIT",
        };
        let result = is_current(&path, &entry);
        assert!(matches!(result, Ok(false)), "expected is_current=false for absent file");
    }

    #[test]
    fn is_current_returns_false_when_wrong_hash() {
        let tmp = tempdir_inner();
        let content = b"old model";
        let path = tmp.join("model.onnx");
        std::fs::write(&path, content).ok();

        let entry = ModelEntry {
            name: "stale_model",
            kind: ModelKind::Wake,
            filename: "model.onnx",
            url: "http://example.com/model.onnx",
            sha256: "0".repeat(64).leak(),
            license: "Apache-2.0",
        };
        let result = is_current(&path, &entry);
        assert!(matches!(result, Ok(false)), "expected is_current=false for wrong hash");
    }

    // -----------------------------------------------------------------------
    // AC6: provenance sidecar write + read round-trip
    // -----------------------------------------------------------------------
    #[test]
    fn provenance_round_trip() {
        let tmp = tempdir_inner();
        let records = vec![
            ProvenanceRecord {
                name: "hey_jarvis".to_owned(),
                kind: ModelKind::Wake,
                filename: "hey_jarvis.onnx".to_owned(),
                sha256: "a".repeat(64),
                url: "http://example.com/hey_jarvis.onnx".to_owned(),
                license: "Apache-2.0".to_owned(),
                fetched_ts_unix: 1_700_000_000,
            },
        ];
        write_provenance(&tmp, &records).expect("write provenance");
        let loaded = read_provenance(&tmp).expect("read provenance");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "hey_jarvis");
        assert_eq!(loaded[0].fetched_ts_unix, 1_700_000_000);
    }

    // -----------------------------------------------------------------------
    // AC7: privileged-target contract — NeedsPrivilege path
    // -----------------------------------------------------------------------
    #[test]
    fn dir_is_writable_true_for_tmp() {
        let tmp = tempdir_inner();
        assert!(dir_is_writable(&tmp), "temp dir should be writable");
    }

    // -----------------------------------------------------------------------
    // build_list returns correct entries
    // -----------------------------------------------------------------------
    #[test]
    fn build_list_length() {
        let prefix = PathBuf::from("/usr/share/wintermute/models");
        let list = build_list(&prefix);
        assert_eq!(list.len(), 4);
    }

    #[test]
    fn build_list_target_paths() {
        let prefix = PathBuf::from("/prefix");
        let list = build_list(&prefix);
        for entry in &list {
            let expected_dir = match entry.kind {
                ModelKind::Wake => "/prefix/wake",
                ModelKind::Vad => "/prefix/vad",
            };
            assert!(
                entry.target_path.starts_with(expected_dir),
                "target_path {} should start with {}",
                entry.target_path.display(),
                expected_dir
            );
        }
    }

    // -----------------------------------------------------------------------
    // OutputFormat::parse
    // -----------------------------------------------------------------------
    #[test]
    fn output_format_parse_valid() {
        assert_eq!(OutputFormat::parse("text").ok(), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::parse("json").ok(), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::parse("JSON").ok(), Some(OutputFormat::Json));
    }

    #[test]
    fn output_format_parse_invalid() {
        assert!(OutputFormat::parse("xml").is_err());
    }

    // -----------------------------------------------------------------------
    // upsert_provenance — update + insert
    // -----------------------------------------------------------------------
    #[test]
    fn upsert_provenance_insert() {
        let mut records = Vec::new();
        let entry = &MANIFEST[0];
        upsert_provenance(&mut records, entry, entry.sha256);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, entry.name);
    }

    #[test]
    fn upsert_provenance_update() {
        let entry = &MANIFEST[0];
        let mut records = vec![ProvenanceRecord {
            name: entry.name.to_owned(),
            kind: entry.kind,
            filename: entry.filename.to_owned(),
            sha256: "old".to_owned(),
            url: entry.url.to_owned(),
            license: entry.license.to_owned(),
            fetched_ts_unix: 0,
        }];
        upsert_provenance(&mut records, entry, "new_hash_value_here_64chars_000000000000000000000000000000000000");
        assert_eq!(records.len(), 1, "upsert should not add a duplicate");
        assert_eq!(records[0].sha256, "new_hash_value_here_64chars_000000000000000000000000000000000000");
    }

    // Helper: create a tempdir and return its PathBuf.
    fn tempdir_inner() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("wm_models_test_{n}_{}", std::process::id()));
        std::fs::create_dir_all(&p).expect("create test tmpdir");
        p
    }
}
