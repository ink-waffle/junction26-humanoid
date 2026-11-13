#!/bin/bash
# M9 pipeline check without the robot (twinflow): the sim oracle commands the Aalto sim over the SAME DDS topics
# teleop uses (rt/lowcmd debug mode + rt/dex3/*/cmd) while record_demos.py records it -> to_lerobot.py ->
# LeRobotDataset load -> replay the dataset's actions in sim (replay_sim.py) and compare the motion.
#   Needs a running Aalto sim; for a 2nd/3rd instance export its AALTO_DDS_DOMAIN / AALTO_CAM_PORT_BASE /
#   AALTO_SCENE_PORT / AALTO_RGBD_PORT first (as for sim/evaluate.py).
#   TASK=sort_fixed SEED=1000 TL=60 bash aaltonomous/data/e2e_sim.sh [OUT_DIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK="${TASK:-sort_fixed}" SEED="${SEED:-1000}" TL="${TL:-60}"
OUT="${1:-$REPO/runs/demo_e2e/$(date +%Y%m%d_%H%M%S)}"
PY_SIM="$HOME/miniforge3/envs/isaac/bin/python" PY_LR="$HOME/miniforge3/envs/lerobot/bin/python"
export PYTHONPATH="$REPO"
cd "$REPO"; mkdir -p "$OUT"; rm -f "$OUT/keys"; mkfifo "$OUT/keys"

"$PY_SIM" -m aaltonomous.data.record_demos --backend sim --task "$TASK" --card-seed "$SEED" --out "$OUT/raw" \
    --session e2e --operator oracle --notes "sim e2e: sim/evaluate oracle as teleop stand-in" \
    < "$OUT/keys" > "$OUT/record.log" 2>&1 &
REC=$!
exec 3>"$OUT/keys"
sleep 3
"$PY_SIM" -m aaltonomous.sim.evaluate --policy oracle --task "$TASK" --seed "$SEED" --trials 1 --time-limit "$TL" \
    --no-cameras --out "$OUT/eval" > "$OUT/eval.log" 2>&1 &
EVAL=$!
# start recording when the episode layout appears (evaluate first parks all objects, then places the layout)
"$PY_SIM" - <<'EOF'
import time
from aaltonomous.sim.scene_client import SceneClient
sc = SceneClient("127.0.0.1")
active = lambda: sum(o["active"] for o in sc.state()["objects"])
t0 = time.time()
while active() and time.time() - t0 < 120: time.sleep(0.05)
while not active() and time.time() - t0 < 180: time.sleep(0.05)
print("layout placed", time.time() - t0)
EOF
printf n >&3
wait $EVAL || echo "evaluate exit $? (gate result only)"
sleep 1; printf s >&3; sleep 0.5; printf y >&3; sleep 3; printf q >&3; exec 3>&-
wait $REC || true
EP=$(ls -d "$OUT"/raw/e2e/ep* | head -1)
echo "recorded: $EP"; grep -E "saved|WARN|!!" "$OUT/record.log" || true

"$PY_LR" aaltonomous/data/to_lerobot.py "$OUT/raw/e2e" --repo-id local/demo_e2e --root "$OUT/lerobot" \
    > "$OUT/convert.log" 2>&1
"$PY_LR" aaltonomous/data/to_lerobot.py --verify "$OUT/lerobot" --export-episode 0 --export-out "$OUT/ep0.npz" \
    2>&1 | grep -v "^x265\|^\[mp4\|^encoded" | tee "$OUT/verify.log"
"$PY_SIM" -m aaltonomous.data.replay_sim "$OUT/ep0.npz" --json "$OUT/replay.json" --max-clips 100000 2>&1 | grep -v "safety clip" | tail -25
echo "-> $OUT"
