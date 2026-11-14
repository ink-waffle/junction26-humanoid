#!/bin/bash
# soft_hand_sim.sh: dev only. The organisers' run_sim.sh with its main swapped for soft_hand_sim.py (hand gains from $HAND_GAINS).
set -eo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
STOCK='MAIN="$REPO/aaltonomous/sim/aalto_sim_main.py"'
OURS='MAIN="$REPO/halal/ops/soft_hand_sim.py"'
grep -qF "$STOCK" "$REPO/aaltonomous/sim/run_sim.sh" || { echo "soft_hand_sim.sh: run_sim.sh no longer has the line $STOCK" >&2; exit 1; }
# run_sim.sh finds the repo from its own path and hard-codes its main: run its text with those two lines replaced
script="$(awk -v repo="REPO=\"$REPO\"" -v stock="$STOCK" -v ours="$OURS" \
    '/^REPO=/ { print repo; next } index($0, stock) { sub(/MAIN=.*/, ours) } { print }' "$REPO/aaltonomous/sim/run_sim.sh")"
exec bash -c "$script" run_sim.sh "$@"
