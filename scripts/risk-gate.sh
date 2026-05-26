#!/usr/bin/env bash
# risk-gate.sh — checks that 7 receipts are present and digest-bound.
# READ-ONLY: the edit-agent must not modify this file.

set -euo pipefail
cd "$(dirname "$0")/.."

exec "$HOME/.claude/skills/autobuilder/scripts/risk-gate.sh" .
