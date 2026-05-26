//! Acceptance test template.
//!
//! For each AC in agent/intent-card.json, the scaffold step (or the
//! orchestrator at intake completion) creates one file
//! `tests/acceptance_<ac_id_lowercase>.rs` from this template, with the
//! AC's `description` and `test` predicate filled in.
//!
//! Read-only after scaffold: the edit-agent must NOT modify acceptance
//! tests. If a test is wrong, write agent/intent_card_amendment_request.json.

#[test]
fn placeholder_ac() {
    // This file is a template; the scaffold step replaces it per AC.
    // Until replaced, this test passes so cargo test does not block the loop.
}
