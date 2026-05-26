//! Shared state: shutdown handle + mute state machine.
//!
//! The daemon has two orthogonal mute conditions:
//!
//! * **TTS-AEC mute** — wake detection is suppressed while
//!   `wm.tts.start..wm.tts.end` is in flight, to mask the AEC tail.
//! * **Dialog request mute** — the entire mic capture is suppressed
//!   (operator-initiated, or dialog policy).
//!
//! Both can be active simultaneously and clear independently. The
//! state machine here collapses them into the boolean a consumer
//! actually cares about (`should_run_wake`, `should_publish_pcm`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Why a mute is in effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuteReason {
    /// TTS playback is in flight; suppress wake.
    TtsActive,
    /// Operator / dialog requested a hard mute.
    DialogRequest,
}

/// Mute state machine.
///
/// Cheap to clone: internal flags are `AtomicBool`s behind an `Arc`.
#[derive(Debug, Clone)]
pub struct MuteState {
    tts_active: Arc<AtomicBool>,
    dialog_mute: Arc<AtomicBool>,
}

impl Default for MuteState {
    fn default() -> Self {
        Self::new()
    }
}

impl MuteState {
    /// All conditions cleared.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tts_active: Arc::new(AtomicBool::new(false)),
            dialog_mute: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set a mute condition.
    pub fn set(&self, reason: MuteReason, on: bool) {
        let cell = match reason {
            MuteReason::TtsActive => &self.tts_active,
            MuteReason::DialogRequest => &self.dialog_mute,
        };
        cell.store(on, Ordering::SeqCst);
    }

    /// Whether wake detection should run right now. Suppressed while
    /// TTS is active or the dialog has hard-muted the mic.
    #[must_use]
    pub fn should_run_wake(&self) -> bool {
        !self.tts_active.load(Ordering::SeqCst)
            && !self.dialog_mute.load(Ordering::SeqCst)
    }

    /// Whether the daemon should publish raw PCM (`speech.chunk`).
    /// Only the dialog-hard-mute blocks PCM; the TTS-AEC mute only
    /// gates wake detection.
    #[must_use]
    pub fn should_publish_pcm(&self) -> bool {
        !self.dialog_mute.load(Ordering::SeqCst)
    }

    /// True iff a dialog hard-mute is currently in effect.
    #[must_use]
    pub fn is_dialog_muted(&self) -> bool {
        self.dialog_mute.load(Ordering::SeqCst)
    }
}

/// Shared shutdown flag. The daemon polls this and integration tests
/// flip it to drive a graceful exit without raising real signals.
#[derive(Debug, Clone, Default)]
pub struct Shutdown {
    flag: Arc<AtomicBool>,
}

impl Shutdown {
    /// New, un-triggered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request shutdown.
    pub fn trigger(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Has shutdown been requested?
    #[must_use]
    pub fn is_triggered(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::redundant_clone)]

    use super::*;

    #[test]
    fn mute_default_lets_everything_run() {
        let m = MuteState::new();
        assert!(m.should_run_wake());
        assert!(m.should_publish_pcm());
        assert!(!m.is_dialog_muted());
    }

    #[test]
    fn tts_mute_only_gates_wake() {
        let m = MuteState::new();
        m.set(MuteReason::TtsActive, true);
        assert!(!m.should_run_wake());
        assert!(
            m.should_publish_pcm(),
            "tts-aec mute should NOT block PCM publish"
        );
        m.set(MuteReason::TtsActive, false);
        assert!(m.should_run_wake());
    }

    #[test]
    fn dialog_mute_gates_everything() {
        let m = MuteState::new();
        m.set(MuteReason::DialogRequest, true);
        assert!(!m.should_run_wake());
        assert!(!m.should_publish_pcm());
        assert!(m.is_dialog_muted());
    }

    #[test]
    fn mutes_compose_independently() {
        let m = MuteState::new();
        m.set(MuteReason::TtsActive, true);
        m.set(MuteReason::DialogRequest, true);
        // Releasing only the dialog mute should leave wake suppressed.
        m.set(MuteReason::DialogRequest, false);
        assert!(!m.should_run_wake());
        assert!(m.should_publish_pcm());
    }

    #[test]
    fn shutdown_handle_flips_once() {
        let s = Shutdown::new();
        let s2 = s.clone();
        assert!(!s.is_triggered());
        s2.trigger();
        assert!(s.is_triggered());
    }
}
