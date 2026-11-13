#!/usr/bin/env bash
set -Eeuo pipefail

command -v verda >/dev/null 2>&1 || {
  printf 'verda CLI is not installed. Install it with: brew install verda-cloud/tap/verda-cli\n' >&2
  exit 1
}

if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [INSTANCE_TYPE]\n' "$0" >&2
  exit 2
fi

verda --agent instance-types --gpu -o json
verda --agent ssh-key list -o json

if [[ $# -eq 1 ]]; then
  verda --agent vm availability --type "$1" -o json
  verda --agent images --type "$1" --category ubuntu -o json
  verda --agent cost balance -o json
  verda --agent cost estimate --type "$1" --os-volume 200 -o json
else
  printf '\nRun %s INSTANCE_TYPE after selecting an RTX GPU.\n' "$0"
fi
