#!/bin/bash
# Participant helper for the Aalto Isaac sim (in the container or after setup_instance.sh).
#   aalto.sh sim [args]     start the sim headless in the background (TASK, default Aalto-Sort-G129-Dex3-Joint),
#                           wait until the scene service answers; log in $AALTO_LOG (default /tmp/aalto_sim.log)
#   aalto.sh stop           stop it (SIGINT to the python pid, then the camera server)
#   aalto.sh status         pid + scene service ping
#   aalto.sh check          scene check (8 checks, ~1 min)
#   aalto.sh baseline [..]  oracle baseline, rung 1: evaluate --policy oracle --task sort_fixed --trials 3
#   aalto.sh eval [args]    python -m aaltonomous.sim.evaluate [args]   (e.g. --policy ws://localhost:8000)
#   aalto.sh snap [--out D]  save one image per camera to D (default /tmp/aalto_snap), print brightness
#   aalto.sh test           offline unit tests (no sim)
#   aalto.sh py ...         python with the env + PYTHONPATH set
# Env passed through to the sim and clients: TASK, AALTO_DR (train|<yaml>), AALTO_INSTANCE + AALTO_*_PORT/DOMAIN.
set -eo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
LOG="${AALTO_LOG:-/tmp/aalto_sim.log}"
export TASK="${TASK:-Aalto-Sort-G129-Dex3-Joint}" PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$AALTO_INSTANCE" ]]; then   # same defaults as run_sim.sh, so clients find an instance sim
    export AALTO_DDS_DOMAIN="${AALTO_DDS_DOMAIN:-7}" AALTO_CAM_PORT_BASE="${AALTO_CAM_PORT_BASE:-56555}"
    export AALTO_SCENE_PORT="${AALTO_SCENE_PORT:-56580}" AALTO_RGBD_PORT="${AALTO_RGBD_PORT:-56560}"
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate isaac
cd "$REPO"

simpid() { pgrep -f "^python .*aalto_sim_main.py" | head -1; }
ping_scene() { python -m aaltonomous.sim.scene_client ping >/dev/null 2>&1; }

cmd="${1:-help}"; shift || true
case "$cmd" in
sim)
    if [[ -n "$(simpid)" ]]; then echo "sim already running (pid $(simpid))"; exit 0; fi
    # run_sim.sh activates conda from ~/miniforge3; HOME must hold it (true in the image and after setup_instance.sh)
    t0=$SECONDS
    nohup bash aaltonomous/sim/run_sim.sh "$@" >"$LOG" 2>&1 &
    echo "starting $TASK (log $LOG) ..."
    for _ in $(seq 300); do
        if ping_scene; then echo "sim ready after $((SECONDS - t0)) s (pid $(simpid))"; exit 0; fi
        if ! kill -0 $! 2>/dev/null; then echo "sim exited; tail of $LOG:"; tail -30 "$LOG"; exit 1; fi
        sleep 2
    done
    echo "not ready after 10 min; see $LOG"; exit 1 ;;
stop)
    p=$(simpid); [[ -n "$p" ]] && kill -INT "$p" && echo "SIGINT -> $p"
    for _ in $(seq 30); do [[ -z "$(simpid)" ]] && break; sleep 1; done
    pkill -f "tools.image_server_process" 2>/dev/null || true
    [[ -n "$(simpid)" ]] && kill -9 "$(simpid)"; echo stopped ;;
status)
    echo "sim pid: $(simpid)"; ping_scene && echo "scene service: OK" || echo "scene service: no answer" ;;
check)
    python aaltonomous/sim/check_scene.py "$@" ;;
baseline)
    python -m aaltonomous.sim.evaluate --policy oracle --task sort_fixed --trials 3 "$@" ;;
eval)
    python -m aaltonomous.sim.evaluate "$@" ;;
snap)
    python aaltonomous/cloud/snap.py "$@" ;;
test)
    python -m unittest aaltonomous/interface/test_safety.py aaltonomous/interface/test_env.py \
        aaltonomous/scene/test_scene.py aaltonomous/deploy/test_deploy.py aaltonomous/sim/aalto_task/test_layout.py ;;
py)
    exec python "$@" ;;
*)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
