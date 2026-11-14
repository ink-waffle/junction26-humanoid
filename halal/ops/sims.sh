#!/bin/bash
# Isaac sims on the dev instance, one per tag (a|b|c), each with its own DDS domain, ports, cores, log and pid file.
#   sims.sh start <tag> [sim_main args ...]   (env: AALTO_DR, default train; CORES="0-5" replaces the tag's core set;
#                                              HAND_GAINS="kp,kd[,armature]" = dev-only sim with those Dex3 actuator gains)
#   sims.sh stop <tag> | status | env <tag>   (eval "$(sims.sh env b)" points clients at sim b)
set -eo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
TAGS=(a b c)
CLIENT_CORES=12-29   # everything that is not a sim (env.sh pins client shells here)
KIT_THREADS="--/plugins/carb.tasking.plugin/threadCount=4 --/persistent/physics/numThreads=2"

tag_env() {
    case "$1" in
    a) echo "unset AALTO_INSTANCE AALTO_DDS_DOMAIN AALTO_CAM_PORT_BASE AALTO_SCENE_PORT AALTO_RGBD_PORT AALTO_TELEIMAGER_REQ_PORT AALTO_CLOCK_PORT" ;;
    b) echo "export AALTO_INSTANCE=b AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560 AALTO_TELEIMAGER_REQ_PORT=61000; unset AALTO_CLOCK_PORT" ;;
    c) echo "export AALTO_INSTANCE=c AALTO_DDS_DOMAIN=8 AALTO_CAM_PORT_BASE=57555 AALTO_SCENE_PORT=57580 AALTO_RGBD_PORT=57560 AALTO_TELEIMAGER_REQ_PORT=62000; unset AALTO_CLOCK_PORT" ;;
    *) echo "unknown sim tag '$1' (a|b|c)" >&2; exit 1 ;;
    esac
}
tag_cores() { case "$1" in a) echo 0-3 ;; b) echo 4-7 ;; c) echo 8-11 ;; esac; }
pidfile() { echo "/tmp/halal_sim_$1.pid"; }
logfile() { echo "/tmp/halal_sim_$1.log"; }
alive() { local p; p=$(cat "$(pidfile "$1")" 2>/dev/null) && kill -0 "$p" 2>/dev/null; }
ping_scene() { (eval "$(tag_env "$1")"; cd "$REPO" && PYTHONPATH="$REPO" python -m aaltonomous.sim.scene_client ping 2>/dev/null); }

source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate isaac
cmd="${1:-status}"; tag="${2:-}"; shift 2 2>/dev/null || true
case "$cmd" in
start)
    if alive "$tag"; then echo "sim $tag already running (pid $(cat "$(pidfile "$tag")"))"; exit 0; fi
    eval "$(tag_env "$tag")"
    export AALTO_DR="${AALTO_DR-train}" TASK="${TASK:-Aalto-Sort-G129-Dex3-Joint}"
    export OMP_NUM_THREADS=1   # torch's per-core OpenMP pool only spin-waits at num_envs=1 (halal/logs/infra.md)
    cores="${CORES:-$(tag_cores "$tag")}"
    cd "$REPO"
    runner="${HAND_GAINS:+halal/ops/soft_hand_sim.sh}"
    nohup taskset -c "$cores" bash "${runner:-aaltonomous/sim/run_sim.sh}" --kit_args "$KIT_THREADS" "$@" >"$(logfile "$tag")" 2>&1 &
    echo $! >"$(pidfile "$tag")"
    echo "starting sim $tag (DR=${AALTO_DR:-off}, cores $cores, hand gains ${HAND_GAINS:-stock}, log $(logfile "$tag"))"
    for _ in $(seq 400); do
        if ping_scene "$tag" >/dev/null; then echo "sim $tag ready after $SECONDS s"; exit 0; fi
        if ! alive "$tag"; then echo "sim $tag exited:"; tail -30 "$(logfile "$tag")"; exit 1; fi
        sleep 2
    done
    echo "sim $tag not ready after 800 s"; exit 1 ;;
stop)
    p=$(cat "$(pidfile "$tag")" 2>/dev/null) || { echo "no pid file for sim $tag"; exit 0; }
    kids=$(pgrep -P "$p" || true)
    kill -INT "$p" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 "$p" $kids 2>/dev/null || true
    rm -f "$(pidfile "$tag")"; echo "sim $tag stopped" ;;
status)
    for t in "${TAGS[@]}"; do
        if alive "$t"; then echo "sim $t pid $(cat "$(pidfile "$t")"): $(ping_scene "$t" | tr -d '\n ' || echo 'no scene answer')"; else echo "sim $t: down"; fi
    done ;;
env)
    tag_env "$tag" ;;
client-cores)
    echo "$CLIENT_CORES" ;;
*)
    sed -n '2,4p' "$0" ;;
esac
