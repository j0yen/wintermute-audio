#!/usr/bin/env bash
# audit.sh-trace: emits ::audit-start marker for harness instrumentation
# audit.sh — runs the BAD_RUST audit against this project.
# READ-ONLY: the edit-agent must not modify this file.
#
# Delegates to ~/.claude/skills/autobuilder/rules/audit-checks.sh.
# Output: target/autobuilder/receipts/risk-gate.json (the gate's expected
# location — eliminates the previous manual `cp audit.json risk-gate.json`
# step every standalone build needed).
# Exit 0 if no BLOCKING findings, 1 otherwise.

set -euo pipefail
cd "$(dirname "$0")/.."

AUDIT="$HOME/.claude/skills/autobuilder/rules/audit-checks.sh"
mkdir -p target/autobuilder/receipts

if [ ! -x "$AUDIT" ]; then
  echo "audit: skill audit-checks.sh not found at $AUDIT" >&2
  exit 1
fi

"$AUDIT" . > target/autobuilder/receipts/risk-gate.json
